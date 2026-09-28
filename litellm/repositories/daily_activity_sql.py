from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import count, islice
from types import MappingProxyType
from typing import Final

from litellm import constants
from litellm.constants import GLOBAL_SPEND_TABLE_NAME, PTU_SENTINEL_API_KEY
from litellm.types.repositories.daily_activity import DailyActivityScope, DailyActivityTable, ExportType

_API_KEY_ROLLED_UP_BIT: Final = 32
_MODEL_GROUP_EXPR: Final = "COALESCE(NULLIF(model_group, ''), model)"
_KEY_FREE_SOURCE_COLUMNS: Final = (
    "date",
    "model",
    "model_group",
    "custom_llm_provider",
    "mcp_namespaced_tool_name",
    "endpoint",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "compression_saved_tokens",
    "compression_savings_spend",
    "prompt_caching_savings_spend",
    "gateway_injected_caching_savings_spend",
    "autorouter_savings_spend",
    "api_requests",
    "successful_requests",
    "failed_requests",
    "total_response_time_ms",
    "timed_requests",
)


@dataclass(frozen=True, slots=True)
class SqlQuery:
    sql: str
    params: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ExportCursor:
    date: str
    entity_id: str
    group_key: str


PRISMA_TO_PG_TABLE: Final[Mapping[DailyActivityTable, str]] = MappingProxyType(
    {
        DailyActivityTable.USER: "LiteLLM_DailyUserSpend",
        DailyActivityTable.TEAM: "LiteLLM_DailyTeamSpend",
        DailyActivityTable.TAG: "LiteLLM_DailyTagSpend",
        DailyActivityTable.ORGANIZATION: "LiteLLM_DailyOrganizationSpend",
        DailyActivityTable.CUSTOMER: "LiteLLM_DailyEndUserSpend",
        DailyActivityTable.AGENT: "LiteLLM_DailyAgentSpend",
    }
)


def adjust_dates_for_timezone(
    start_date: str,
    end_date: str,
    timezone_offset_minutes: int | None,
    include_current_utc_day: bool = False,
    utc_now: datetime | None = None,
) -> tuple[str, str]:
    if not include_current_utc_day or timezone_offset_minutes is None:
        return start_date, end_date
    now: Final = utc_now if utc_now is not None else datetime.now(timezone.utc)
    caller_local_today: Final = (now - timedelta(minutes=timezone_offset_minutes)).date().isoformat()
    if end_date < caller_local_today:
        return start_date, end_date
    return start_date, max(end_date, now.date().isoformat())


def build_where_clause(scope: DailyActivityScope, *, start_index: int = 1) -> tuple[str, tuple[object, ...]]:
    adjusted_start, adjusted_end = adjust_dates_for_timezone(
        scope.start_date,
        scope.end_date,
        scope.timezone_offset_minutes,
        scope.include_current_utc_day,
    )
    entity_index: Final = start_index + 2
    has_entity_array: Final = scope.entity_ids is not None and bool(scope.entity_ids)
    exclusion_index: Final = entity_index + int(has_entity_array)
    model_index: Final = exclusion_index + int(bool(scope.exclude_entity_ids))
    api_keys_index: Final = model_index + int(bool(scope.model))
    conditions: Final = (
        f"date >= ${start_index}",
        f"date <= ${start_index + 1}",
        *(
            ("FALSE",)
            if scope.entity_ids == ()
            else (f'"{scope.entity_id_field}" = ANY(${entity_index}::text[])',)
            if has_entity_array
            else ()
        ),
        *((f'NOT ("{scope.entity_id_field}" = ANY(${exclusion_index}::text[]))',) if scope.exclude_entity_ids else ()),
        *((f"model = ${model_index}",) if scope.model else ()),
        *(
            ("FALSE",)
            if scope.api_keys == ()
            else (f"api_key = ANY(${api_keys_index}::text[])",)
            if scope.api_keys
            else ()
        ),
    )
    params: Final = (
        adjusted_start,
        adjusted_end,
        *((list(scope.entity_ids or ()),) if has_entity_array else ()),
        *((list(scope.exclude_entity_ids),) if scope.exclude_entity_ids else ()),
        *((scope.model,) if scope.model else ()),
        *((list(scope.api_keys),) if scope.api_keys else ()),
    )
    return " AND ".join(conditions), params


def _ptu_flat_cost_select(table: DailyActivityTable, *, aggregate: bool = True) -> str:
    if table is DailyActivityTable.TEAM:
        return "SUM(ptu_flat_cost)::float AS ptu_flat_cost" if aggregate else "SUM(scoped.ptu_flat_cost)::float"
    return "0::float AS ptu_flat_cost" if aggregate else "0::float"


def _rollup_metric_select(table: DailyActivityTable) -> str:
    return f"""
            SUM(spend)::float AS spend,
            {_ptu_flat_cost_select(table)},
            SUM(prompt_tokens)::bigint AS prompt_tokens,
            SUM(completion_tokens)::bigint AS completion_tokens,
            SUM(cache_read_input_tokens)::bigint AS cache_read_input_tokens,
            SUM(cache_creation_input_tokens)::bigint AS cache_creation_input_tokens,
            SUM(compression_saved_tokens)::bigint AS compression_saved_tokens,
            SUM(compression_savings_spend)::float AS compression_savings_spend,
            SUM(prompt_caching_savings_spend)::float AS prompt_caching_savings_spend,
            SUM(gateway_injected_caching_savings_spend)::float AS gateway_injected_caching_savings_spend,
            SUM(autorouter_savings_spend)::float AS autorouter_savings_spend,
            SUM(api_requests)::bigint AS api_requests,
            SUM(successful_requests)::bigint AS successful_requests,
            SUM(failed_requests)::bigint AS failed_requests,
            SUM(total_response_time_ms)::bigint AS total_response_time_ms,
            SUM(timed_requests)::bigint AS timed_requests"""


def _key_free_source(pg_table: str, where_clause: str, marker_param: str | None) -> str:
    if marker_param is None:
        return f'"{pg_table}"\n        WHERE {where_clause}'
    columns: Final = ", ".join(_KEY_FREE_SOURCE_COLUMNS)
    return f"""(
            SELECT {columns}
            FROM "{GLOBAL_SPEND_TABLE_NAME}"
            WHERE {where_clause} AND date <= {marker_param}
            UNION ALL
            SELECT {columns}
            FROM "{pg_table}"
            WHERE {where_clause} AND date > {marker_param}
        ) AS key_free_source"""


def build_aggregated_sql(
    scope: DailyActivityScope, *, api_key_limit: int, global_rollup_through: str | None
) -> SqlQuery:
    pg_table: Final = PRISMA_TO_PG_TABLE[scope.table]
    where_clause, where_params = build_where_clause(scope)
    sentinel_param: Final = f"${len(where_params) + 1}"
    marker_param: Final = None if global_rollup_through is None else f"${len(where_params) + 2}"
    if not 1 <= api_key_limit <= constants.USAGE_TOP_API_KEYS_MAX:
        raise ValueError(f"api_key_limit must be between 1 and {constants.USAGE_TOP_API_KEYS_MAX}")
    top_keys_limit_param: Final = len(where_params) + (3 if global_rollup_through is not None else 2)
    metric_select: Final = _rollup_metric_select(scope.table)
    sql: Final = f"""
        (SELECT
            date,
            NULL::text AS api_key,
            model,
            {_MODEL_GROUP_EXPR} AS model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            (GROUPING(date) << 6) | {_API_KEY_ROLLED_UP_BIT}
                | GROUPING(model, {_MODEL_GROUP_EXPR},
                           custom_llm_provider, mcp_namespaced_tool_name,
                           endpoint) AS group_level,
            NULL::bigint AS distinct_api_keys,{metric_select}
        FROM {_key_free_source(pg_table, where_clause, marker_param)}
        GROUP BY GROUPING SETS (
            (date),
            (date, model),
            (date, {_MODEL_GROUP_EXPR}),
            (date, custom_llm_provider),
            (date, mcp_namespaced_tool_name),
            (date, endpoint),
            ()
        ))
        UNION ALL
        (WITH top_api_keys AS (
            SELECT api_key, COUNT(*) OVER () AS distinct_api_keys
            FROM "{pg_table}"
            WHERE {where_clause} AND api_key <> {sentinel_param}
            GROUP BY api_key
            ORDER BY SUM(spend) DESC, api_key
            LIMIT ${top_keys_limit_param}
        )
        SELECT
            date,
            api_key,
            model,
            {_MODEL_GROUP_EXPR} AS model_group,
            custom_llm_provider,
            mcp_namespaced_tool_name,
            endpoint,
            GROUPING(date, api_key, model, {_MODEL_GROUP_EXPR},
                     custom_llm_provider, mcp_namespaced_tool_name,
                     endpoint) AS group_level,
            MAX(top_api_keys.distinct_api_keys) AS distinct_api_keys,{metric_select}
        FROM "{pg_table}" JOIN top_api_keys USING (api_key)
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date, api_key),
            (date, model, api_key),
            (date, {_MODEL_GROUP_EXPR}, api_key),
            (date, custom_llm_provider, api_key),
            (date, mcp_namespaced_tool_name, api_key),
            (date, endpoint, api_key)
        ))
    """
    marker_params: Final = () if global_rollup_through is None else (global_rollup_through,)
    return SqlQuery(
        sql=sql,
        params=(*where_params, PTU_SENTINEL_API_KEY, *marker_params, api_key_limit),
    )


def build_entity_rollup_sql(scope: DailyActivityScope) -> SqlQuery:
    pg_table: Final = PRISMA_TO_PG_TABLE[scope.table]
    where_clause, params = build_where_clause(scope)
    sql: Final = f"""
        SELECT
            "{scope.entity_id_field}" AS entity_id,
            date,
            api_key,
            GROUPING(api_key) AS api_key_rolled,{_rollup_metric_select(scope.table)}
        FROM "{pg_table}"
        WHERE {where_clause}
        GROUP BY GROUPING SETS (
            (date, "{scope.entity_id_field}"),
            (date, "{scope.entity_id_field}", api_key)
        )
    """
    return SqlQuery(sql=sql, params=params)


def _key_spend_select() -> str:
    return """
            COALESCE(SUM(spend), 0)::float AS spend,
            COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
            COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
            (COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0))::bigint AS total_tokens,
            COALESCE(SUM(api_requests), 0)::bigint AS api_requests,
            COALESCE(SUM(successful_requests), 0)::bigint AS successful_requests,
            COALESCE(SUM(failed_requests), 0)::bigint AS failed_requests,
            COALESCE(SUM(cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens"""


def _bounded_limit(limit: int, *, minimum: int = 1) -> None:
    if limit < minimum:
        raise ValueError(f"limit must be at least {minimum}")


def build_key_search_sql(scope: DailyActivityScope, *, search: str, limit: int) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    search_param: Final = len(where_params) + 1
    sentinel_param: Final = search_param + 1
    limit_param: Final = sentinel_param + 1
    escaped: Final = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause}
          AND api_key <> ${sentinel_param} AND api_key ILIKE ${search_param} ESCAPE '\\'
        GROUP BY api_key
        ORDER BY SUM(spend) DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, f"%{escaped}%", PTU_SENTINEL_API_KEY, limit))


def build_model_top_keys_sql(
    scope: DailyActivityScope, *, model_group: str, by_model_group: bool, limit: int
) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    model_param: Final = len(where_params) + 1
    sentinel_param: Final = model_param + 1
    limit_param: Final = sentinel_param + 1
    model_clause: Final = (
        f"COALESCE(NULLIF(model_group, ''), model) = ${model_param}" if by_model_group else f"model = ${model_param}"
    )
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause} AND {model_clause} AND api_key <> ${sentinel_param}
        GROUP BY api_key
        ORDER BY spend DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, model_group, PTU_SENTINEL_API_KEY, limit))


def build_cache_leakage_keys_sql(scope: DailyActivityScope, *, limit: int) -> SqlQuery:
    _bounded_limit(limit)
    where_clause, where_params = build_where_clause(scope)
    sentinel_param: Final = len(where_params) + 1
    limit_param: Final = sentinel_param + 1
    sql: Final = f"""
        SELECT api_key,{_key_spend_select()}
        FROM "{PRISMA_TO_PG_TABLE[scope.table]}"
        WHERE {where_clause} AND api_key <> ${sentinel_param}
        GROUP BY api_key
        HAVING SUM(prompt_tokens) - SUM(cache_read_input_tokens) > 0
        ORDER BY SUM(prompt_tokens) - SUM(cache_read_input_tokens) DESC, api_key
        LIMIT ${limit_param}
    """
    return SqlQuery(sql=sql, params=(*where_params, PTU_SENTINEL_API_KEY, limit))


def build_export_sql(
    scope: DailyActivityScope, *, export_type: ExportType, after: ExportCursor | None, batch_size: int
) -> SqlQuery:
    _bounded_limit(batch_size)
    where_clause, where_params = build_where_clause(scope)
    group_key, output_key, user_fields, type_joins = _export_grouping(export_type)
    grouping_keys: Final = (
        f"scoped.date, COALESCE(scoped.\"{scope.entity_id_field}\", '')",
        *((group_key,) if export_type is not ExportType.DAILY else ()),
    )
    entity_joins: Final = (
        ('LEFT JOIN "LiteLLM_TeamTable" tt ON tt.team_id = scoped.team_id',)
        if scope.table is DailyActivityTable.TEAM
        else ('LEFT JOIN "LiteLLM_OrganizationTable" ot ON ot.organization_id = scoped.organization_id',)
        if scope.table is DailyActivityTable.ORGANIZATION
        else ()
    )
    joins: Final = (*type_joins, *entity_joins)
    alias_expression: Final = (
        "MAX(tt.team_alias)"
        if scope.table is DailyActivityTable.TEAM
        else "MAX(ot.organization_alias)"
        if scope.table is DailyActivityTable.ORGANIZATION
        else "NULL::text"
    )
    parameter_indexes: Final = count(len(where_params) + 1)
    sentinel_param: Final = next(parameter_indexes) if export_type is not ExportType.DAILY else None
    cursor_indexes: Final = tuple(islice(parameter_indexes, 3)) if after is not None else ()
    limit_param: Final = next(parameter_indexes)
    cursor_clause, cursor_params = _export_cursor_clause(
        scope, after=after, cursor_indexes=cursor_indexes, group_key=group_key
    )
    sentinel_clause: Final = f" AND api_key <> ${sentinel_param}" if sentinel_param is not None else ""
    table: Final = PRISMA_TO_PG_TABLE[scope.table]
    flat_cost: Final = _ptu_flat_cost_select(scope.table, aggregate=False)
    sql: Final = f"""
        WITH scoped AS (
            SELECT * FROM "{table}"
            WHERE {where_clause}{sentinel_clause}
        )
        SELECT
            scoped.date,
            COALESCE(scoped."{scope.entity_id_field}", '') AS entity_id,
            {alias_expression} AS entity_alias,
            {output_key} AS api_key,
            {user_fields},
            {"NULLIF(COALESCE(scoped.model, ''), '')" if export_type is ExportType.DAILY_WITH_MODELS else "NULL::text"} AS model,
            COALESCE(SUM(scoped.spend), 0)::float AS spend,
            {flat_cost} AS flat_cost,
            COALESCE(SUM(scoped.prompt_tokens), 0)::bigint AS prompt_tokens,
            COALESCE(SUM(scoped.completion_tokens), 0)::bigint AS completion_tokens,
            COALESCE(SUM(scoped.api_requests), 0)::bigint AS api_requests,
            COALESCE(SUM(scoped.successful_requests), 0)::bigint AS successful_requests,
            COALESCE(SUM(scoped.failed_requests), 0)::bigint AS failed_requests,
            COALESCE(SUM(scoped.cache_read_input_tokens), 0)::bigint AS cache_read_input_tokens,
            COALESCE(SUM(scoped.cache_creation_input_tokens), 0)::bigint AS cache_creation_input_tokens
        FROM scoped
        {" ".join(joins)}
        WHERE TRUE{cursor_clause}
        GROUP BY {", ".join(grouping_keys)}
        ORDER BY {", ".join(grouping_keys)}
        LIMIT ${limit_param}
    """
    return SqlQuery(
        sql=sql,
        params=(
            *where_params,
            *((PTU_SENTINEL_API_KEY,) if export_type is not ExportType.DAILY else ()),
            *cursor_params,
            batch_size,
        ),
    )


def _export_grouping(export_type: ExportType) -> tuple[str, str, str, tuple[str, ...]]:
    match export_type:
        case ExportType.DAILY:
            return (
                "''",
                "NULL::text",
                "NULL::text AS key_alias, NULL::text AS user_id, NULL::text AS user_email",
                (),
            )
        case ExportType.DAILY_WITH_KEYS:
            return (
                "scoped.api_key",
                "NULLIF(scoped.api_key, '')",
                "MAX(COALESCE(vt.key_alias, dvt.key_alias)) AS key_alias, "
                "MAX(COALESCE(vt.user_id, dvt.user_id)) AS user_id, MAX(u.user_email) AS user_email",
                (
                    'LEFT JOIN "LiteLLM_VerificationToken" vt ON vt.token = scoped.api_key',
                    """LEFT JOIN LATERAL (
                        SELECT key_alias, user_id
                        FROM "LiteLLM_DeletedVerificationToken"
                        WHERE token = scoped.api_key
                        ORDER BY deleted_at DESC
                        LIMIT 1
                    ) dvt ON vt.token IS NULL""",
                    'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = COALESCE(vt.user_id, dvt.user_id)',
                ),
            )
        case ExportType.DAILY_WITH_MODELS:
            return (
                "COALESCE(scoped.model, '')",
                "NULL::text",
                "NULL::text AS key_alias, NULL::text AS user_id, NULL::text AS user_email",
                (),
            )
        case ExportType.DAILY_WITH_USERS:
            return (
                "COALESCE(vt.user_id, dvt.user_id, '')",
                "NULL::text",
                "NULL::text AS key_alias, MAX(COALESCE(vt.user_id, dvt.user_id)) AS user_id, "
                "MAX(u.user_email) AS user_email",
                (
                    'LEFT JOIN "LiteLLM_VerificationToken" vt ON vt.token = scoped.api_key',
                    """LEFT JOIN LATERAL (
                        SELECT key_alias, user_id
                        FROM "LiteLLM_DeletedVerificationToken"
                        WHERE token = scoped.api_key
                        ORDER BY deleted_at DESC
                        LIMIT 1
                    ) dvt ON vt.token IS NULL""",
                    'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id = COALESCE(vt.user_id, dvt.user_id)',
                ),
            )


def _export_cursor_clause(
    scope: DailyActivityScope,
    *,
    after: ExportCursor | None,
    cursor_indexes: tuple[int, ...],
    group_key: str,
) -> tuple[str, tuple[object, ...]]:
    if after is None:
        return "", ()
    first_cursor_index: Final = cursor_indexes[0]
    clause: Final = (
        f""" AND (scoped.date, COALESCE(scoped."{scope.entity_id_field}", ''), {group_key}) """
        f"> (${first_cursor_index}, ${cursor_indexes[1]}, ${cursor_indexes[2]})"
    )
    return clause, (after.date, after.entity_id, after.group_key)
