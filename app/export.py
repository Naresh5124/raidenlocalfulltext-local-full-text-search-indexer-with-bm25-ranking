"""Data Export & Serialization Formatter for RaidenLocalFullText.

Turns query results, aggregation rollups, snapshots of the whole index and
quarantine records into deterministic JSON, CSV and Markdown reports.

Design notes
------------
- Every exporter is *pure*: it receives already-materialised payloads and
  returns a string, so exports can be streamed, written to disk or returned
  straight from an HTTP endpoint.
- Output is deterministic: keys are sorted in JSON, columns are ordered
  (stable metadata columns first, then schema fields in declaration order),
  and numbers are rendered with a fixed precision.
- Row flattening is shared by CSV and Markdown so the two formats always agree
  on which columns exist and how values are rendered.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import text

from app.aggregation import AggregationError, aggregate_documents
from app.query import QueryError, execute_query
from app.schema import DocumentSchema
from app.storage import SQLiteIndexStore

EXPORT_FORMATS = {"json", "csv", "markdown", "md"}
MAX_EXPORT_ROWS = 10000
MAX_CELL_CHARS = 2000
STABLE_COLUMNS = ("doc_id", "external_id", "created_at", "updated_at")


class ExportError(Exception):
    """Raised when an export request cannot be satisfied."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_format(fmt: str) -> str:
    fmt = str(fmt).strip().lower()
    if fmt in {"md", "markdown"}:
        return "markdown"
    if fmt not in EXPORT_FORMATS:
        raise ExportError(
            f"unsupported export format {fmt!r}; expected one of {sorted(EXPORT_FORMATS)}"
        )
    return fmt


# --------------------------------------------------------------------- rows
def _render_value(value: Any) -> str:
    """Deterministic, human-readable rendering for CSV/Markdown cells."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        rounded = round(value, 6)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _flatten_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a query result row into a single-level dict.

    Columns are ordered: stable metadata, then schema fields in declaration
    order, then payload extras, then ``body_text`` and any ``score``.
    """
    flat: Dict[str, Any] = {
        "doc_id": record.get("doc_id"),
        "external_id": record.get("external_id"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }
    fields = record.get("fields")
    if isinstance(fields, dict):
        for name, value in fields.items():
            flat[f"field_{name}"] = value
    payload = record.get("payload")
    if isinstance(payload, dict):
        for name, value in payload.items():
            flat.setdefault(f"payload_{name}", value)
    body = record.get("body_text")
    if body:
        flat["body_text"] = body[:MAX_CELL_CHARS]
    if "score" in record:
        flat["score"] = record["score"]
    return flat


def _column_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


# ------------------------------------------------------------------- JSON
def render_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


# ------------------------------------------------------------------- CSV
def render_csv(rows: Iterable[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    materialised = [dict(row) for row in rows]
    header = columns or _column_order(materialised)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in materialised:
        writer.writerow([_render_value(row.get(column)) for column in header])
    return buffer.getvalue()


# --------------------------------------------------------------- Markdown
def _escape_cell(value: Any) -> str:
    rendered = _render_value(value).replace("|", "\\|")
    rendered = rendered.replace("\n", " ")
    if len(rendered) > MAX_CELL_CHARS:
        rendered = rendered[: MAX_CELL_CHARS - 3] + "..."
    return rendered or ""


def render_markdown_report(sections: Sequence[Dict[str, Any]]) -> str:
    """Render a report made of titled sections, each with optional tables.

    ``sections`` is a list of dicts with ``title``, optional ``lines`` and
    optional ``rows``/``columns`` table data.
    """
    lines: List[str] = []
    for section in sections:
        title = str(section.get("title", "")).strip()
        if title:
            lines.append(f"## {title}")
            lines.append("")
        if section.get("lines"):
            for line in section["lines"]:
                lines.append(f"- {line}")
            lines.append("")
        rows = section.get("rows") or []
        columns = section.get("columns") or _column_order(rows)
        if rows:
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in rows:
                lines.append("| " + " | ".join(_escape_cell(row.get(column)) for column in columns) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- exports
def build_query_snapshot(
    store: SQLiteIndexStore,
    schema: DocumentSchema,
    filter_expression: Any = None,
    sort: Optional[List[Dict[str, Any]]] = None,
    limit: int = 100,
    offset: int = 0,
    include_stats: bool = True,
) -> Dict[str, Any]:
    """Materialise a query result as a snapshot payload ready for export."""
    if limit < 1 or limit > MAX_EXPORT_ROWS:
        raise ExportError(f"limit must be between 1 and {MAX_EXPORT_ROWS}")
    if offset < 0:
        raise ExportError("offset must be >= 0")
    result = execute_query(store, schema, filter_expression, sort=sort, limit=limit, offset=offset)
    rows = [_flatten_row(record) for record in result["results"]]
    snapshot: Dict[str, Any] = {
        "kind": "query_snapshot",
        "generated_at": _now(),
        "schema_name": schema.name,
        "schema_version": schema.version,
        "filter": filter_expression,
        "sort": result["sort"],
        "limit": limit,
        "offset": offset,
        "total": result["total"],
        "returned": result["returned"],
        "has_more": result["has_more"],
        "columns": _column_order(rows),
        "rows": rows,
    }
    if include_stats:
        numeric_fields: Dict[str, Dict[str, Any]] = {}
        for field in schema.fields:
            if field.type in ("integer", "number"):
                numeric_fields[field.name] = aggregate_documents(
                    store, schema, metrics=["count", "sum", "min", "max", "avg"], field=field.name
                )["total"]["statistics"]
        snapshot["field_statistics"] = numeric_fields
    return snapshot


def build_index_snapshot(store: SQLiteIndexStore, schema: Optional[DocumentSchema] = None) -> Dict[str, Any]:
    """Snapshot the whole index: stats, ingestion runs and quarantine records."""
    stats = store.stats()
    persisted_schema: Optional[Any] = None
    if schema is not None:
        persisted_schema = schema.to_dict()
    elif stats.get("schema"):
        try:
            persisted_schema = json.loads(stats["schema"])
        except json.JSONDecodeError:  # pragma: no cover - defensive
            persisted_schema = stats["schema"]
    with store.engine.connect() as conn:
        runs = [
            {
                "run_id": r[0],
                "source": r[1],
                "format": r[2],
                "received": r[3],
                "accepted": r[4],
                "rejected": r[5],
                "created_at": r[6],
            }
            for r in conn.execute(
                text(
                    "SELECT run_id, source, format, received, accepted, rejected, created_at "
                    "FROM ingestion_runs ORDER BY run_id"
                )
            ).fetchall()
        ]
        quarantine = store.list_quarantine(limit=MAX_EXPORT_ROWS, offset=0)
    return {
        "kind": "index_snapshot",
        "generated_at": _now(),
        "db_path": store.db_path,
        "schema": persisted_schema,
        "stats": stats,
        "ingestion_runs": runs,
        "quarantine": quarantine,
    }


def export_documents(
    store: SQLiteIndexStore,
    schema: DocumentSchema,
    fmt: str = "json",
    filter_expression: Any = None,
    sort: Optional[List[Dict[str, Any]]] = None,
    limit: int = 100,
    offset: int = 0,
    include_stats: bool = True,
) -> Dict[str, Any]:
    """Export query results in the requested format.

    Returns ``{"format": ..., "snapshot": ..., "content": ...}`` where ``content``
    is the fully rendered report string.
    """
    format_name = _normalise_format(fmt)
    snapshot = build_query_snapshot(
        store,
        schema,
        filter_expression=filter_expression,
        sort=sort,
        limit=limit,
        offset=offset,
        include_stats=include_stats,
    )
    if format_name == "json":
        content = render_json(snapshot)
    elif format_name == "csv":
        content = render_csv(snapshot["rows"], snapshot["columns"])
    else:
        sections = [
            {
                "title": "Summary",
                "lines": [
                    f"Schema: {snapshot['schema_name']} (v{snapshot['schema_version']})",
                    f"Generated: {snapshot['generated_at']}",
                    f"Matching documents: {snapshot['total']}",
                    f"Returned rows: {snapshot['returned']} (limit {snapshot['limit']}, offset {snapshot['offset']})",
                    f"Filter: {json.dumps(snapshot['filter'], sort_keys=True, default=str)}",
                ],
            },
            {
                "title": "Results",
                "columns": snapshot["columns"],
                "rows": snapshot["rows"],
            },
        ]
        if snapshot.get("field_statistics"):
            stat_rows = []
            for name, stats in snapshot["field_statistics"].items():
                stat_rows.append(
                    {
                        "field": name,
                        "count": stats.get("count"),
                        "sum": stats.get("sum"),
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "avg": stats.get("avg"),
                    }
                )
            sections.append({"title": "Numeric field statistics", "rows": stat_rows})
        content = render_markdown_report(sections)
    return {
        "format": format_name,
        "row_count": snapshot["returned"],
        "content": content,
        "snapshot": snapshot,
    }


def export_snapshot(store: SQLiteIndexStore, schema: Optional[DocumentSchema], fmt: str = "json") -> Dict[str, Any]:
    """Export a whole-index snapshot in JSON, CSV (documents) or Markdown."""
    format_name = _normalise_format(fmt)
    snapshot = build_index_snapshot(store, schema)
    documents = store.list_documents(limit=MAX_EXPORT_ROWS, offset=0)
    rows = [_flatten_row(document) for document in documents]
    columns = _column_order(rows)
    if format_name == "json":
        content = render_json(snapshot)
    elif format_name == "csv":
        content = render_csv(rows, columns)
    else:
        sections = [
            {
                "title": "Index snapshot",
                "lines": [
                    f"Database: {snapshot['db_path']}",
                    f"Generated: {snapshot['generated_at']}",
                    f"Documents: {snapshot['stats']['documents']}",
                    f"Distinct terms: {snapshot['stats']['terms']}",
                    f"Postings: {snapshot['stats']['postings']}",
                    f"Total tokens: {snapshot['stats']['total_tokens']}",
                    f"Average document length: {snapshot['stats']['avg_doc_length']}",
                ],
            },
            {"title": "Ingestion runs", "rows": snapshot["ingestion_runs"]},
            {"title": "Quarantined rows", "rows": snapshot["quarantine"]},
            {"title": "Documents", "columns": columns, "rows": rows},
        ]
        content = render_markdown_report(sections)
    return {
        "format": format_name,
        "row_count": len(rows),
        "content": content,
        "snapshot": snapshot,
    }


def export_aggregation(
    store: SQLiteIndexStore,
    schema: DocumentSchema,
    fmt: str = "json",
    field: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    filter_expression: Any = None,
    percentiles: Optional[List[float]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Export an aggregation rollup in the requested format."""
    format_name = _normalise_format(fmt)
    rollup = aggregate_documents(
        store,
        schema,
        metrics=metrics,
        field=field,
        group_by=group_by,
        filter_expression=filter_expression,
        percentiles=percentiles,
        limit=limit,
        offset=offset,
    )
    flat_rows: List[Dict[str, Any]] = []
    for group in rollup["groups"]:
        row: Dict[str, Any] = {}
        for key, value in group["key"].items():
            row[f"group_{key}"] = value
        row["count"] = group["count"]
        for metric, value in group["metrics"].items():
            if metric == "percentiles":
                for level, level_value in value.items():
                    row[f"percentile_{level}"] = level_value
            else:
                row[metric] = value
        flat_rows.append(row)
    if format_name == "json":
        content = render_json(rollup)
    elif format_name == "csv":
        content = render_csv(flat_rows)
    else:
        sections = [
            {
                "title": "Aggregation report",
                "lines": [
                    f"Field: {rollup['field'] or 'doc_id'}",
                    f"Metrics: {', '.join(rollup['metrics'])}",
                    f"Group by: {', '.join(rollup['group_by']) or '(none)'}",
                    f"Groups: {rollup['group_count']}",
                    f"Total matching documents: {rollup['total']['documents']}",
                ],
            },
            {"title": "Groups", "rows": flat_rows},
        ]
        content = render_markdown_report(sections)
    return {
        "format": format_name,
        "row_count": len(flat_rows),
        "content": content,
        "rollup": rollup,
    }


def write_export(content: str, file_path: str) -> Dict[str, Any]:
    """Write rendered export content to disk and return a small receipt."""
    path = Path(file_path)
    if str(path.parent) and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return {"written": str(path), "bytes": len(content.encode("utf-8"))}
