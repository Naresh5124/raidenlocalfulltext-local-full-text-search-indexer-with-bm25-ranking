"""Ingestion, quarantine and query HTTP endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.ingestion import IngestionError, IngestionPipeline
from app.query import QueryError, execute_query
from app.schema import FieldValidationError, ValidationError
from app.storage import DocumentNotFound, IndexConflictError
from app.state import AppState

router = APIRouter()


class QueryRequest(BaseModel):
    """Body schema for the compound query endpoint."""

    filter: Optional[Dict[str, Any]] = None
    sort: Optional[List[Dict[str, Any]]] = None
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class StreamRequest(BaseModel):
    """Body schema for the streaming ingestion endpoint.

    Accepts either ``lines`` (an array of raw lines) or a bare JSON array of
    lines posted directly as the request body.
    """

    lines: Optional[List[str]] = None
    format: Optional[str] = None
    source: Optional[str] = None
    batch_size: Optional[int] = Field(None, ge=1, le=1000)
    max_quarantine_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)

    def resolved_lines(self) -> List[str]:
        return self.lines or []


def _state() -> AppState:
    from app.main import state

    return state


def _pipeline(state: AppState, batch_size: Optional[int], max_quarantine_ratio: Optional[float]) -> IngestionPipeline:
    return IngestionPipeline(
        state.store,
        state.schema,
        batch_size=batch_size if batch_size is not None else 100,
        max_quarantine_ratio=max_quarantine_ratio if max_quarantine_ratio is not None else 0.5,
    )


@router.post("/schema", status_code=201)
def set_schema(payload: Dict[str, Any] = Body(...)):
    try:
        schema = _state().set_schema(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"kind": "schema_error", "message": str(exc)})
    return schema.to_dict()


@router.get("/schema")
def get_schema():
    schema = _state().schema
    if schema is None:
        raise HTTPException(status_code=404, detail={"kind": "schema_missing", "message": "no schema has been configured"})
    return schema.to_dict()


@router.post("/documents", status_code=201)
def add_document(document: Dict[str, Any] = Body(...)):
    try:
        created = _state().store.add_document(document)
    except FieldValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"kind": "schema_error", "message": str(exc), "errors": exc.errors},
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"kind": "schema_error", "message": str(exc), "errors": []},
        )
    except IndexConflictError as exc:
        raise HTTPException(status_code=409, detail={"kind": "conflict", "message": str(exc)})
    return created


@router.get("/documents/{external_id}")
def get_document(external_id: str):
    try:
        return _state().store.get_document(external_id)
    except DocumentNotFound:
        raise HTTPException(status_code=404, detail={"kind": "not_found", "message": f"document {external_id!r} not found"})


@router.put("/documents/{external_id}")
def update_document(external_id: str, document: Dict[str, Any] = Body(...)):
    try:
        return _state().store.update_document(external_id, document)
    except DocumentNotFound:
        raise HTTPException(status_code=404, detail={"kind": "not_found", "message": f"document {external_id!r} not found"})
    except FieldValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"kind": "schema_error", "message": str(exc), "errors": exc.errors},
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"kind": "schema_error", "message": str(exc), "errors": []},
        )
    except IndexConflictError as exc:
        raise HTTPException(status_code=409, detail={"kind": "conflict", "message": str(exc)})


@router.delete("/documents/{external_id}", status_code=204)
def delete_document(external_id: str):
    if not _state().store.delete_document(external_id):
        raise HTTPException(status_code=404, detail={"kind": "not_found", "message": f"document {external_id!r} not found"})
    return None


@router.get("/documents")
def list_documents(limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return {"results": _state().store.list_documents(limit=limit, offset=offset), "limit": limit, "offset": offset}


@router.post("/ingest")
def ingest(
    payload: Any = Body(...),
    format: Optional[str] = Query(None, description="json | jsonl | csv (auto-detected when omitted)"),
    source: str = Query("inline"),
    batch_size: int = Query(100, ge=1, le=1000),
    max_quarantine_ratio: float = Query(0.5, ge=0.0, le=1.0),
):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before ingesting documents"})
    try:
        pipeline = _pipeline(state, batch_size, max_quarantine_ratio)
        return pipeline.ingest(payload, fmt=format, source=source)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail={"kind": "ingestion_error", "message": str(exc)})


@router.post("/ingest/stream")
def ingest_stream(
    request: Any = Body(...),
    format: Optional[str] = Query(None, description="jsonl | csv"),
    source: Optional[str] = Query(None),
    batch_size: Optional[int] = Query(None, ge=1, le=1000),
    max_quarantine_ratio: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before ingesting documents"})

    lines: List[str] = []
    inline_batch_size: Optional[int] = None
    inline_ratio: Optional[float] = None
    inline_format: Optional[str] = None
    inline_source: Optional[str] = None

    if isinstance(request, list):
        lines = [str(line) for line in request]
    elif isinstance(request, dict):
        try:
            parsed = StreamRequest(**request)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=422, detail={"kind": "invalid_request", "message": str(exc)})
        lines = parsed.resolved_lines()
        inline_batch_size = parsed.batch_size
        inline_ratio = parsed.max_quarantine_ratio
        inline_format = parsed.format
        inline_source = parsed.source
    else:
        raise HTTPException(status_code=422, detail={"kind": "invalid_request", "message": "stream body must be an array of lines or an object with 'lines'"})

    try:
        pipeline = _pipeline(
            state,
            batch_size if batch_size is not None else inline_batch_size,
            max_quarantine_ratio if max_quarantine_ratio is not None else inline_ratio,
        )
        return pipeline.ingest_stream(
            lines,
            fmt=format or inline_format or "jsonl",
            source=source or inline_source or "stream",
        )
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail={"kind": "ingestion_error", "message": str(exc)})


@router.get("/quarantine")
def list_quarantine(limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    return {
        "total": _state().store.quarantine_count(),
        "limit": limit,
        "offset": offset,
        "results": _state().store.list_quarantine(limit=limit, offset=offset),
    }


@router.delete("/quarantine")
def clear_quarantine():
    return {"removed": _state().store.clear_quarantine()}


@router.post("/query")
def query(request: QueryRequest):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before querying documents"})
    try:
        return execute_query(
            state.store,
            state.schema,
            request.filter,
            sort=request.sort,
            limit=request.limit,
            offset=request.offset,
        )
    except QueryError as exc:
        raise HTTPException(status_code=400, detail={"kind": "query_error", "message": str(exc)})


@router.get("/search")
def search(
    q: str = Query(..., description="full-text query string"),
    sort: Optional[str] = Query(None, description="comma separated sort fields, e.g. created_at,doc_id"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before querying documents"})
    sort_spec = None
    if sort:
        sort_spec = [{"field": part.strip(), "direction": "desc"} for part in sort.split(",") if part.strip()]
    try:
        return execute_query(state.store, state.schema, {"term": q}, sort=sort_spec, limit=limit, offset=offset)
    except QueryError as exc:
        raise HTTPException(status_code=400, detail={"kind": "query_error", "message": str(exc)})


@router.get("/stats")
def stats():
    return _state().store.stats()