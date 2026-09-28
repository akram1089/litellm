import os
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from integration.spend._daily_activity_fixtures import seed_daily_activity_fixture
from prisma import Prisma
from psycopg import sql

from litellm import constants
from litellm.proxy.management_endpoints.common_daily_activity import get_daily_activity_aggregated
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.daily_activity_repository import DailyActivityDatabase, DailyActivityRepository
from litellm.types.repositories.daily_activity import (
    DailyActivityProxyReads,
    DailyActivityScope,
    DailyActivityTable,
    ExportType,
    KeyMetadataRow,
    SpendLogsWindow,
)


def _scoped_url(url: str, schema: str) -> str:
    parsed: Final = urlsplit(url)
    return urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))


@asynccontextmanager
async def _daily_activity_database() -> AsyncIterator[Prisma]:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(url) as connection:
                seed_daily_activity_fixture(
                    connection,
                    schema=schema,
                    ptu_sentinel_api_key=constants.PTU_SENTINEL_API_KEY,
                )
            database: Final = Prisma(datasource={"url": _scoped_url(url, schema)})
            await database.connect()
            try:
                yield database
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@dataclass(frozen=True, slots=True)
class _PrismaDatabase:
    db: Prisma


@dataclass(frozen=True, slots=True)
class _ProxyReads(DailyActivityProxyReads):
    database: Prisma

    async def global_rollup_reconciled_through(self) -> str | None:
        return None

    async def recover_key_metadata(
        self, resolved: Mapping[str, KeyMetadataRow], api_keys: frozenset[str], window: SpendLogsWindow | None
    ) -> Mapping[str, KeyMetadataRow]:
        user_ids: Final = frozenset(row.user_id for row in resolved.values() if row.user_id)
        user_rows: Final = await find_many_in(self.database.litellm_usertable, "user_id", user_ids) if user_ids else ()
        user_emails: Final = MappingProxyType({row.user_id: row.user_email for row in user_rows if row.user_email})
        return MappingProxyType(
            {
                key: KeyMetadataRow(
                    api_key=row.api_key,
                    key_alias=row.key_alias,
                    team_id=row.team_id,
                    user_id=row.user_id,
                    user_email=row.user_email or user_emails.get(row.user_id),
                    key_exists=row.key_exists,
                    tags=row.tags,
                )
                for key, row in resolved.items()
            }
        )


def _repository(database: Prisma) -> DailyActivityRepository:
    client: Final = cast(DailyActivityDatabase, _PrismaDatabase(database))
    return DailyActivityRepository(client, proxy_reads=_ProxyReads(database))


def _scope(table: DailyActivityTable, entity_id_field: str, entity_id: str) -> DailyActivityScope:
    return DailyActivityScope(
        table=table,
        entity_id_field=entity_id_field,
        entity_ids=(entity_id,),
        exclude_entity_ids=(),
        api_keys=None,
        start_date="2026-06-01",
        end_date="2026-06-01",
        model=None,
        timezone_offset_minutes=None,
    )


@pytest.mark.asyncio
async def test_repository_queries_and_exports_seeded_daily_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _daily_activity_database() as database:
        repository: Final = _repository(database)
        team_scope: Final = _scope(DailyActivityTable.TEAM, "team_id", "team-1")
        monkeypatch.setattr(constants, "USAGE_EXPORT_BATCH_SIZE", 2)
        aggregate: Final = await repository.aggregated(
            team_scope, include_entity_breakdown=True, api_key_limit=3
        )
        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        assert len(totals) == 1
        assert totals[0].spend == 1273.0
        assert totals[0].ptu_flat_cost == 42.0
        assert aggregate.distinct_api_keys == 5
        grouped_keys: Final = frozenset(
            row.api_key for row in aggregate.grouping_rows if row.group_level == 31 and row.api_key
        )
        assert grouped_keys == frozenset(("key-a", "key-b", "key-c"))
        entity_totals: Final = tuple(row for row in aggregate.entity_rows or () if row.api_key_rolled)
        assert len(entity_totals) == 1
        assert entity_totals[0].spend == 1273.0
        assert entity_totals[0].ptu_flat_cost == 42.0

        targeted_model_keys: Final = await repository.model_top_keys(
            team_scope, model_group="model-target", by_model_group=False, limit=3
        )
        assert tuple(row.api_key for row in targeted_model_keys) == ("key-target",)
        popular_model_keys: Final = await repository.model_top_keys(
            team_scope, model_group="model-popular", by_model_group=False, limit=3
        )
        assert tuple(row.api_key for row in popular_model_keys) == ("key-a", "key-b", "key-c")
        assert await repository.search_keys(team_scope, search="target", limit=10) == ("key-target",)
        leakage_keys: Final = await repository.cache_leakage_keys(team_scope, limit=2)
        assert tuple(row.api_key for row in leakage_keys) == ("key-cache", "key-c")

        exports: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY_WITH_KEYS)]
        )
        assert tuple(row.api_key for row in exports) == (
            "key-a",
            "key-b",
            "key-c",
            "key-cache",
            "key-target",
        )
        assert sum(row.spend for row in exports) == 273.0
        assert sum(row.flat_cost for row in exports) == 0.0
        deleted_key_export: Final = next(row for row in exports if row.api_key == "key-target")
        assert (deleted_key_export.key_alias, deleted_key_export.user_id, deleted_key_export.user_email) == (
            "deleted-target",
            "user-1",
            "user@example.com",
        )
        user_exports: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY_WITH_USERS)]
        )
        assert len(user_exports) == 1
        assert (user_exports[0].user_id, user_exports[0].user_email, user_exports[0].spend) == (
            "user-1",
            "user@example.com",
            273.0,
        )
        daily_export: Final = tuple(
            [row async for row in repository.export_rows(team_scope, export_type=ExportType.DAILY)]
        )
        assert len(daily_export) == 1
        assert daily_export[0].spend == 1273.0
        assert daily_export[0].flat_cost == 42.0

        metadata: Final = await repository.key_metadata(frozenset(("key-a", "key-target")), None)
        assert metadata["key-a"].key_exists is True
        assert metadata["key-a"].key_alias == "alias-a"
        assert metadata["key-a"].tags == ("blue", "gold")
        assert metadata["key-a"].user_email == "user@example.com"
        assert metadata["key-target"].key_exists is False
        assert metadata["key-target"].key_alias == "deleted-target"
        assert metadata["key-target"].tags == ("archived",)


@pytest.mark.asyncio
async def test_aggregated_returns_totals_with_a_one_key_limit() -> None:
    async with _daily_activity_database() as database:
        aggregate: Final = await _repository(database).aggregated(
            _scope(DailyActivityTable.TEAM, "team_id", "team-1"),
            include_entity_breakdown=False,
            api_key_limit=1,
        )

        totals: Final = tuple(row for row in aggregate.grouping_rows if row.group_level == 127)
        per_key_rows: Final = tuple(row for row in aggregate.grouping_rows if row.api_key is not None)
        per_key_names: Final = frozenset(row.api_key for row in per_key_rows)
        assert len(totals) == 1
        assert totals[0].spend == 1273.0
        assert len(per_key_rows) == 6
        assert len(per_key_names) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "entity_field", "entity_id", "fixture_name"),
    (
        (DailyActivityTable.USER, "user_id", "user-1", "daily_activity_user.json"),
        (DailyActivityTable.TEAM, "team_id", "team-1", "daily_activity_team.json"),
    ),
)
async def test_aggregated_response_matches_base_golden(
    table: DailyActivityTable, entity_field: str, entity_id: str, fixture_name: str
) -> None:
    async with _daily_activity_database() as database:
        result: Final = await get_daily_activity_aggregated(
            _repository(database),
            _scope(table, entity_field, entity_id),
            entity_metadata_field=MappingProxyType({"team-1": {"team_alias": "Usage Team"}}),
            include_entity_breakdown=True,
        )
        golden_path: Final = Path(__file__).with_name("fixtures") / fixture_name
        assert result.model_dump_json() + "\n" == golden_path.read_text()
