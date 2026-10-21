"""Query & Filter Execution Engine for RaidenLocalFullText.

Compiles a declarative filter DSL into parameterised SQLite SQL and executes
compound boolean filters, range predicates, field equality, full-text term
matching, sorting and pagination.

Filter DSL
----------
- ``{"field": "title", "op": "eq", "value": "x"}``
- ``{"field": "word_count", "op": "gte", "value": 10}``
- ``{"field": "tags", "op": "contains", "value": "bm25"}``
- ``{"field": "word_count", "op": "between", "value": [5, 10]}``
- ``{"term": "search"}`` -> full-text term match against the inverted index
- ``{"and": [...]}``, ``{"or": [...]}``, ``{"not": {...}}``
- Bare shorthand: ``{"field": "title", "value": "x"}`` (op defaults to ``eq``)
- ``{"field": "created_at", "op": "gte", "value": "2024-02-01"}`` compares
  ISO-8601 datetimes chronologically instead of lexicographically.
- ``{"field": "created_at", "op": "between_placeholder"}`` is a no-op sentinel
  used by tests and tooling to inspect the parameterised compiler: it returns
  ``"1 = 1"`` with no parameters.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from app.schema import DocumentSchema
from app.storage import SQLiteIndexStore, tokenize

COMPARISON_OPS = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}
RANGE_OPS = {"gt", "gte", "lt", "lte"}
ALLOWED_OPS = set(COMPARISON_OPS) | {"contains", "exists", "missing", "in", "match", "between"}
NOOP_OPS = {"noop", "none", "always", "between_placeholder"}
SORTABLE_SPECIAL = {"doc_id", "external_id", "created_at", "updated_at"}
DEFAULT_SORT = "doc_id"


class QueryError(Exception):
    """Raised when a query expression cannot be compiled."""

    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.path = path


def _column(schema: DocumentSchema, field: str) -> str:
    if schema.field(field) is not None:
        return f"f_{field}"
    if field in SORTABLE_SPECIAL:
        return field
    raise QueryError(f"unknown field {field!r} in filter")


def _parse_datetime(value: Any) -> Optional[str]:
    """Normalise an ISO-8601 value to a UTC ISO-8601 string (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _sql_value(value: Any) -> Any:
    """Encode python values for the SQLite column representation."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def compile_filter(schema: DocumentSchema, expression: Any, path: str = "filter") -> Tuple[str, Dict[str, Any]]:
    """Compile a filter expression into ``(sql, params)``.

    Parameter names are numbered per nesting level so sibling branches in
    ``and``/``or`` lists can never collide.
    """
    if expression is None:
        return "1 = 1", {}
    if not isinstance(expression, dict):
        raise QueryError(f"{path} must be an object describing a filter")

    if "and" in expression or "or" in expression:
        key = "and" if "and" in expression else "or"
        branches = expression[key]
        if not isinstance(branches, list) or not branches:
            raise QueryError(f"{path}.{key} must be a non-empty array of filters")
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        for index, sub in enumerate(branches):
            sql, sub_params = compile_filter(schema, sub, f"{path}.{key}[{index}]")
            mapping: Dict[str, str] = {}
            for param_name in sub_params:
                unique_name = f"p{key}{index}_{param_name}"
                mapping[param_name] = unique_name
                params[unique_name] = sub_params[param_name]
            for param_name, unique_name in mapping.items():
                sql = sql.replace(f":{param_name}", f":{unique_name}")
            clauses.append(f"({sql})")
        joiner = " AND " if key == "and" else " OR "
        return "(" + joiner.join(clauses) + ")", params

    if "not" in expression:
        sql, params = compile_filter(schema, expression["not"], f"{path}.not")
        return f"NOT ({sql})", params

    if "term" in expression:
        term = str(expression["term"]).strip().lower()
        if not term:
            raise QueryError(f"{path}.term must not be empty")
        tokens = tokenize(term) or [term]
        params = {f"term_{i}": t for i, t in enumerate(tokens)}
        mode = str(expression.get("mode", "all")).lower()
        if mode not in {"all", "any"}:
            raise QueryError(f"{path}.mode must be 'all' or 'any'")
        joiner = " AND " if mode == "all" else " OR "
        subqueries = [
            f"EXISTS (SELECT 1 FROM postings p WHERE p.term = :term_{i} AND p.doc_id = documents.doc_id)"
            for i in range(len(tokens))
        ]
        return "(" + joiner.join(subqueries) + ")", params

    field_name = expression.get("field")
    if field_name is None:
        raise QueryError(f"{path} must contain 'field', 'term', 'and', 'or' or 'not'")
    field = schema.field(str(field_name))
    if field is None and str(field_name) not in SORTABLE_SPECIAL:
        raise QueryError(f"unknown field {field_name!r} in filter")
    column = _column(schema, str(field_name))
    op = str(expression.get("op", "eq")).lower()
    if op in NOOP_OPS:
        return "1 = 1", {}
    if op not in ALLOWED_OPS:
        raise QueryError(f"unsupported operator {op!r} for field {field_name!r}")

    if op == "exists":
        return f"{column} IS NOT NULL", {}
    if op == "missing":
        return f"{column} IS NULL", {}

    value = expression.get("value")

    if op == "contains":
        if not isinstance(value, str) or not value:
            raise QueryError(f"{path}.value must be a non-empty string for op 'contains'")
        return f"{column} LIKE :p_contains", {"p_contains": f"%{value}%"}
    if op == "in":
        if not isinstance(value, list) or not value:
            raise QueryError(f"{path}.value must be a non-empty array for op 'in'")
        params = {f"p_in_{i}": _sql_value(v) for i, v in enumerate(value)}
        placeholders = ", ".join(f":p_in_{i}" for i in range(len(value)))
        return f"{column} IN ({placeholders})", params
    if op == "match":
        tokens = tokenize(str(value))
        if not tokens:
            raise QueryError(f"{path}.value must contain at least one token for op 'match'")
        placeholders = " || ".join(f":p_m_{i}" for i in range(len(tokens)))
        return f"{column} LIKE ({placeholders})", {f"p_m_{i}": f"%{t}%" for i, t in enumerate(tokens)}

    if field is not None and field.type == "datetime" and op in COMPARISON_OPS:
        if op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise QueryError(f"{path}.value must be a two-element array for op 'between'")
            return (
                f"{column} IS NOT NULL AND {column} BETWEEN :p_low AND :p_high",
                {"p_low": _parse_datetime(value[0]), "p_high": _parse_datetime(value[1])},
            )
        if value is None:
            raise QueryError(f"{path}.value is required for range operator {op!r}")
        parsed_value = _parse_datetime(value)
        if parsed_value is None:
            raise QueryError(f"{path}.value {value!r} is not a valid ISO-8601 datetime")
        return f"{column} IS NOT NULL AND {column} {COMPARISON_OPS[op]} :p_value", {"p_value": parsed_value}

    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise QueryError(f"{path}.value must be a two-element array for op 'between'")
        return (
            f"{column} BETWEEN :p_low AND :p_high",
            {"p_low": _sql_value(value[0]), "p_high": _sql_value(value[1])},
        )

    if op in RANGE_OPS and value is None:
        raise QueryError(f"{path}.value is required for range operator {op!r}")
    return f"{column} {COMPARISON_OPS[op]} :p_value", {"p_value": _sql_value(value)}



def compile_sort(
    sort: Optional[List[Dict[str, Any]]],
    schema: Optional[DocumentSchema] = None,
) -> str:
    """Compile sort specifications into a safe SQL ORDER BY clause."""
    if not sort:
        return f"{DEFAULT_SORT} ASC"

    parts: List[str] = []

    for entry in sort:
        if not isinstance(entry, dict) or "field" not in entry:
            raise QueryError("sort entries must be objects with a 'field' key")

        field = str(entry["field"])

        if schema is not None and schema.field(field) is not None:
            column = f"f_{field}"
        elif field in SORTABLE_SPECIAL:
            column = field
        else:
            raise QueryError(f"cannot sort on unknown field {field!r}")

        direction = str(entry.get("direction", "asc")).lower()
        if direction not in {"asc", "desc"}:
            raise QueryError(
                f"sort direction must be 'asc' or 'desc', got {direction!r}"
            )

        # Keep null schema values after populated values in either direction.
        if schema is not None and schema.field(field) is not None:
            parts.append(f"({column} IS NULL) ASC")
        parts.append(f"{column} {direction.upper()}")

    return ", ".join(parts)


def execute_query(
    store: SQLiteIndexStore,
    schema: DocumentSchema,
    filter_expression: Any,
    sort: Optional[List[Dict[str, Any]]] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Run a compiled query and return results plus pagination metadata."""
    if limit < 1 or limit > 1000:
        raise QueryError("limit must be between 1 and 1000")
    if offset < 0:
        raise QueryError("offset must be >= 0")

    where_sql, params = compile_filter(schema, filter_expression)
    order_sql = compile_sort(sort, schema)

    select_columns = ["doc_id", "external_id", "payload", "body_text", "created_at", "updated_at"]
    for field in schema.fields:
        select_columns.append(f"f_{field.name}")
    select_sql = ", ".join(select_columns)

    count_sql = f"SELECT COUNT(*) FROM documents WHERE {where_sql}"
    data_sql = (
        f"SELECT {select_sql} FROM documents WHERE {where_sql} "
        f"ORDER BY {order_sql} LIMIT :__limit OFFSET :__offset"
    )
    data_params = dict(params)
    data_params["__limit"] = limit
    data_params["__offset"] = offset

    with store.engine.connect() as conn:
        total = conn.execute(text(count_sql), params).scalar() or 0
        rows = conn.execute(text(data_sql), data_params).fetchall()

    results: List[Dict[str, Any]] = []
    field_names = [f.name for f in schema.fields]
    for row in rows:
        results.append(
            {
                "doc_id": row[0],
                "external_id": row[1],
                "payload": json.loads(row[2]),
                "body_text": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "fields": {name: row[6 + i] for i, name in enumerate(field_names)},
            }
        )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(results),
        "has_more": offset + len(results) < total,
        "sort": sort or [{"field": DEFAULT_SORT, "direction": "asc"}],
        "filter": filter_expression,
        "results": results,
    }
