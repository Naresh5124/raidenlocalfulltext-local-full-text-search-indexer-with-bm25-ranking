"""Data Export & Serialization Formatter HTTP endpoints.

Expose query-result snapshots, index snapshots and aggregation rollups as
JSON, CSV or Markdown reports. Text formats are returned with the matching
content type so browsers and CLI tools can download them directly.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Response

from app.aggregation import AggregationError
from app.export import (
    ExportError,
    export_aggregation,
    export_documents,
    export_snapshot,
    write_export,
)
from app.query import QueryError
from app.state import AppState

router = APIRouter()

CONTENT_TYPES = {
    "json": "application/json",
    "csv": "text/csv",
    "markdown": "text/markdown",
}


def _state() -> AppState:
    from app.main import state

    return state


def _text_response(format_name: str, content: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=CONTENT_TYPES[format_name],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _require_schema(state: AppState):
    if state.schema is None:
        raise HTTPException(
            status_code=400,
            detail={"kind": "schema_missing", "message": "configure a schema before exporting documents"},
        )
    return state.schema


def _handle_export_error(exc: Exception):
    if isinstance(exc, ExportError):
        raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
    if isinstance(exc, QueryError):
        raise HTTPException(status_code=400, detail={"kind": "query_error", "message": str(exc)})
    if isinstance(exc, AggregationError):
        raise HTTPException(status_code=400, detail={"kind": "aggregation_error", "message": str(exc)})
    raise exc


@router.get("/export/documents")
def export_documents_endpoint(
    format: str = Query("json", description="json | csv | markdown"),
    filter: Optional[str] = Query(None, description="filter JSON expression"),
    sort: Optional[str] = Query(None, description="comma separated sort fields"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    include_stats: bool = Query(True),
    file: Optional[str] = Query(None, description="optional path to write the report to"),
):
    state = _state()
    schema = _require_schema(state)
    filter_expression: Optional[Dict[str, Any]] = None
    if filter:
        try:
            filter_expression = __import__("json").loads(filter) if isinstance(filter, str) else filter
        except Exception as exc:  # json.JSONDecodeError
            raise HTTPException(status_code=400, detail={"kind": "query_error", "message": f"invalid filter JSON: {exc}"})
    sort_spec = None
    if sort:
        sort_spec = [{"field": part.strip(), "direction": "desc"} for part in sort.split(",") if part.strip()]
    try:
        payload = export_documents(
            state.store,
            schema,
            fmt=format,
            filter_expression=filter_expression,
            sort=sort_spec,
            limit=limit,
            offset=offset,
            include_stats=include_stats,
        )
    except (ExportError, QueryError, AggregationError) as exc:
        _handle_export_error(exc)
    if file:
        try:
            receipt = write_export(payload["content"], file)
        except OSError as exc:
            raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
        return {"format": payload["format"], "row_count": payload["row_count"], **receipt}
    if payload["format"] == "json":
        return payload["snapshot"]
    return _text_response(payload["format"], payload["content"], f"raiden_export_{payload['format']}.{payload['format']}")


@router.post("/export/documents")
def export_documents_post(request: Dict[str, Any] = Body(...)):
    state = _state()
    schema = _require_schema(state)
    fmt = request.get("format", "json")
    try:
        payload = export_documents(
            state.store,
            schema,
            fmt=fmt,
            filter_expression=request.get("filter"),
            sort=request.get("sort"),
            limit=int(request.get("limit", 100)),
            offset=int(request.get("offset", 0)),
            include_stats=bool(request.get("include_stats", True)),
        )
    except (ExportError, QueryError, AggregationError) as exc:
        _handle_export_error(exc)
    file = request.get("file")
    if file:
        try:
            receipt = write_export(payload["content"], file)
        except OSError as exc:
            raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
        return {"format": payload["format"], "row_count": payload["row_count"], **receipt}
    if payload["format"] == "json":
        return payload["snapshot"]
    return _text_response(payload["format"], payload["content"], f"raiden_export.{payload['format']}")


@router.get("/export/snapshot")
def export_snapshot_endpoint(
    format: str = Query("json", description="json | csv | markdown"),
    file: Optional[str] = Query(None),
):
    state = _state()
    try:
        payload = export_snapshot(state.store, state.schema, fmt=format)
    except ExportError as exc:
        raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
    if file:
        try:
            receipt = write_export(payload["content"], file)
        except OSError as exc:
            raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
        return {"format": payload["format"], "row_count": payload["row_count"], **receipt}
    if payload["format"] == "json":
        return payload["snapshot"]
    return _text_response(payload["format"], payload["content"], f"raiden_snapshot.{payload['format']}")


@router.get("/export/aggregate")
def export_aggregation_endpoint(
    format: str = Query("json", description="json | csv | markdown"),
    field: Optional[str] = Query(None),
    metrics: Optional[str] = Query(None, description="comma separated metrics"),
    group_by: Optional[str] = Query(None, description="comma separated grouping fields"),
    percentiles: Optional[str] = Query(None, description="comma separated percentile levels"),
    filter: Optional[str] = Query(None, description="optional filter JSON expression"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    file: Optional[str] = Query(None),
):
    state = _state()
    schema = _require_schema(state)
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
    group_list = [g.strip() for g in group_by.split(",") if g.strip()] if group_by else None
    percentile_levels = [float(p.strip()) for p in percentiles.split(",") if p.strip()] if percentiles else None
    filter_expression: Optional[Dict[str, Any]] = None
    if filter:
        try:
            filter_expression = __import__("json").loads(filter) if isinstance(filter, str) else filter
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"kind": "query_error", "message": f"invalid filter JSON: {exc}"})
    try:
        payload = export_aggregation(
            state.store,
            schema,
            fmt=format,
            field=field,
            metrics=metric_list,
            group_by=group_list,
            filter_expression=filter_expression,
            percentiles=percentile_levels,
            limit=limit,
            offset=offset,
        )
    except (ExportError, QueryError, AggregationError) as exc:
        _handle_export_error(exc)
    if file:
        try:
            receipt = write_export(payload["content"], file)
        except OSError as exc:
            raise HTTPException(status_code=400, detail={"kind": "export_error", "message": str(exc)})
        return {"format": payload["format"], "row_count": payload["row_count"], **receipt}
    if payload["format"] == "json":
        return payload["rollup"]
    return _text_response(payload["format"], payload["content"], f"raiden_aggregate.{payload['format']}")


@router.get("/export/formats")
def export_formats():
    return {
        "formats": ["json", "csv", "markdown"],
        "sources": ["query_results", "index_snapshot", "aggregation_rollup"],
        "max_rows": 10000,
    }
