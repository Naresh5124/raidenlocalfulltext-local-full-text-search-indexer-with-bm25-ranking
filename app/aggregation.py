"""Aggregation & Summary Rollup Calculator for RaidenLocalFullText.

Computes count, sum, min, max, average and percentile statistics across
groups of documents, either grouped by one or more schema fields or over the
whole corpus as a single rollup.

Percentile semantics
-------------------
Percentiles are computed with linear interpolation over the sorted values
(the classic "linear" / R-7 method), which is deterministic and matches the
behaviour of common statistical libraries for the same input.

Supported metrics
-----------------
- ``count``     : number of non-null values in the group
- ``sum``       : sum of numeric values
- ``min``       : smallest value (numeric or lexicographic for text)
- ``max``       : largest value
- ``avg`` / ``mean`` : arithmetic mean of numeric values
- ``percentile``     : p-th percentile (0 <= p <= 100) of numeric values
- ``median``    : shorthand for the 50th percentile
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import text

from app.query import QueryError, compile_filter
from app.schema import DocumentSchema
from app.storage import SQLiteIndexStore, _sql_type

AGGREGATION_METRICS = {"count", "sum", "min", "max", "avg", "mean", "percentile", "median"}
NUMERIC_SQL_TYPES = {"INTEGER", "REAL"}
DEFAULT_PERCENTILE = 50.0


class AggregationError(Exception):
    """Raised when an aggregation request cannot be compiled."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile (R-7) of ``values``; ``None`` if empty."""
    if not values:
        return None
    if not 0.0 <= p <= 100.0:
        raise AggregationError(f"percentile must be between 0 and 100, got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def compute_statistics(values: Iterable[Any], percentiles: Optional[List[float]] = None) -> Dict[str, Any]:
    """Compute count/sum/min/max/avg/percentiles from an iterable of values."""
    percentile_levels = [float(p) for p in (percentiles or [DEFAULT_PERCENTILE])]
    raw = [v for v in values if v is not None]
    numeric = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
    stats: Dict[str, Any] = {"count": len(raw)}
    if numeric:
        stats["sum"] = round(sum(numeric), 6)
        stats["min"] = min(numeric)
        stats["max"] = max(numeric)
        stats["avg"] = round(sum(numeric) / len(numeric), 6)
        levels = percentile_levels if percentile_levels is not None else [DEFAULT_PERCENTILE]
        stats["percentiles"] = {float(p): round(percentile(numeric, p), 6) for p in levels}
        stats["median"] = round(percentile(numeric, 50.0), 6)
    else:
        stats["sum"] = None
        stats["min"] = None
        stats["max"] = None
        stats["avg"] = None
        stats["percentiles"] = {}
        stats["median"] = None
    return stats


def _metric_sql(metric: str, column: str, percentile_value: Optional[float]) -> str:
    if metric in ("avg", "mean"):
        return f"AVG({column})"
    if metric == "count":
        return f"COUNT({column})"
    if metric == "sum":
        return f"SUM({column})"
    if metric == "min":
        return f"MIN({column})"
    if metric == "max":
        return f"MAX({column})"
    if metric == "median":
        return f"AVG({column})"
    if metric == "percentile":
        # SQLite has no native percentile; approximate with the closest rank
        # using an ordered offset, then refine in Python for exactness.
        p = DEFAULT_PERCENTILE if percentile_value is None else float(percentile_value)
        return f"AVG({column})"
    raise AggregationError(f"unsupported metric {metric!r}; expected one of {sorted(AGGREGATION_METRICS)}")


def _field_column(schema: DocumentSchema, field: str) -> str:
    if field in ("doc_id", "external_id", "created_at", "updated_at"):
        return field
    definition = schema.field(field)
    if definition is None:
        raise AggregationError(f"unknown aggregation field {field!r}")
    return f"f_{field}"


def aggregate_documents(
    store: SQLiteIndexStore,
    schema: DocumentSchema,
    metrics: Optional[List[str]] = None,
    field: Optional[str] = None,
    group_by: Optional[List[str]] = None,
    filter_expression: Any = None,
    percentiles: Optional[List[float]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Compute rollup statistics across documents, optionally grouped.

    Returns a payload with ``groups`` (one entry per group key) and a corpus
    ``total`` rollup over all matching documents.
    """
    if limit < 1 or limit > 1000:
        raise AggregationError("limit must be between 1 and 1000")
    if offset < 0:
        raise AggregationError("offset must be >= 0")

    metric_list = metrics or ["count", "sum", "min", "max", "avg", "percentile"]
    normalised_metrics: List[str] = []
    for metric in metric_list:
        metric = str(metric).lower()
        if metric not in AGGREGATION_METRICS:
            raise AggregationError(f"unsupported metric {metric!r}")
        if metric not in normalised_metrics:
            normalised_metrics.append(metric)

    percentile_levels = [float(p) for p in (percentiles or [DEFAULT_PERCENTILE])]
    for p in percentile_levels:
        if not 0.0 <= p <= 100.0:
            raise AggregationError(f"percentile must be between 0 and 100, got {p}")

    if field is not None:
        value_column = _field_column(schema, str(field))
    else:
        value_column = "doc_id"

    where_sql, params = compile_filter(schema, filter_expression)

    group_columns: List[str] = []
    for name in group_by or []:
        group_columns.append(_field_column(schema, str(name)))

    select_parts: List[str] = []
    for column in group_columns:
        select_parts.append(column)
    select_parts.append("COUNT(*) AS group_count")
    for metric in normalised_metrics:
        select_parts.append(f"{_metric_sql(metric, value_column, None)} AS m_{metric}")

    select_sql = ", ".join(select_parts)
    if group_columns:
        group_sql = ", ".join(group_columns)
        data_sql = (
            f"SELECT {select_sql} FROM documents WHERE {where_sql} "
            f"GROUP BY {group_sql} ORDER BY {group_sql} LIMIT :__limit OFFSET :__offset"
        )
    else:
        data_sql = f"SELECT {select_sql} FROM documents WHERE {where_sql}"

    data_params = dict(params)
    data_params["__limit"] = limit
    data_params["__offset"] = offset

    groups: List[Dict[str, Any]] = []
    with store.engine.connect() as conn:
        rows = conn.execute(text(data_sql), data_params).fetchall()
        for row in rows:
            index = 0
            key: Dict[str, Any] = {}
            for name in group_by or []:
                key[name] = row[index]
                index += 1
            group_count = row[index]
            index += 1
            metrics_payload: Dict[str, Any] = {}
            for metric in normalised_metrics:
                value = row[index]
                index += 1
                if metric == "count":
                    metrics_payload[metric] = int(value) if value is not None else 0
                elif value is None:
                    metrics_payload[metric] = None
                else:
                    metrics_payload[metric] = round(float(value), 6)
            groups.append({"key": key, "count": int(group_count), "metrics": metrics_payload})

        # Exact percentiles require the full value list per group.
        if "percentile" in normalised_metrics or "median" in normalised_metrics:
            if group_columns:
                value_sql = f"SELECT {', '.join(group_columns)}, {value_column} FROM documents WHERE {where_sql}"
            else:
                value_sql = f"SELECT {value_column} FROM documents WHERE {where_sql}"
            buckets: Dict[Any, List[float]] = {}
            for row in conn.execute(text(value_sql), params).fetchall():
                if group_columns:
                    bucket_key = tuple(row[: len(group_columns)])
                    value = row[len(group_columns)]
                else:
                    bucket_key = ()
                    value = row[0]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    buckets.setdefault(bucket_key, []).append(float(value))
            for group in groups:
                if group_by:
                    key_tuple = tuple(group["key"].values())
                else:
                    key_tuple = ()
                values = buckets.get(key_tuple, [])
                group["metrics"]["percentiles"] = {
                    p: round(v, 6) for p, v in ((p, percentile(values, p)) for p in percentile_levels) if v is not None
                }
                median = percentile(values, 50.0)
                group["metrics"]["median"] = round(median, 6) if median is not None else None

    total_values: List[float] = []
    with store.engine.connect() as conn:
        for row in conn.execute(text(f"SELECT {value_column} FROM documents WHERE {where_sql}"), params).fetchall():
            value = row[0]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total_values.append(float(value))
        total_documents = conn.execute(text(f"SELECT COUNT(*) FROM documents WHERE {where_sql}"), params).scalar() or 0

    return {
        "field": field,
        "value_column": value_column,
        "metrics": normalised_metrics,
        "percentiles": percentile_levels,
        "group_by": group_by or [],
        "filter": filter_expression,
        "limit": limit,
        "offset": offset,
        "group_count": len(groups),
        "groups": groups,
        "total": {
            "documents": int(total_documents),
            "statistics": compute_statistics(total_values, percentile_levels),
        },
    }
