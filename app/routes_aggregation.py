"""Aggregation, summary rollup and CLI-facing HTTP endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.aggregation import AggregationError, aggregate_documents, compute_statistics, percentile
from app.cli import run_command, build_parser
from app.query import QueryError
from app.schema import ValidationError
from app.state import AppState

router = APIRouter()


class AggregateRequest(BaseModel):
    metrics: Optional[List[str]] = None
    field: Optional[str] = None
    group_by: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None
    percentiles: Optional[List[float]] = Field(None, description="percentile levels between 0 and 100")
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class StatisticsRequest(BaseModel):
    values: List[Any]
    percentiles: Optional[List[float]] = None


def _state() -> AppState:
    from app.main import state

    return state


@router.post("/aggregate")
def aggregate(request: AggregateRequest):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before aggregating"})
    try:
        return aggregate_documents(
            state.store,
            state.schema,
            metrics=request.metrics,
            field=request.field,
            group_by=request.group_by,
            filter_expression=request.filter,
            percentiles=request.percentiles,
            limit=request.limit,
            offset=request.offset,
        )
    except AggregationError as exc:
        raise HTTPException(status_code=400, detail={"kind": "aggregation_error", "message": str(exc)})
    except QueryError as exc:
        raise HTTPException(status_code=400, detail={"kind": "query_error", "message": str(exc)})


@router.get("/aggregate")
def aggregate_get(
    field: Optional[str] = Query(None),
    metrics: Optional[str] = Query(None, description="comma separated metrics"),
    group_by: Optional[str] = Query(None, description="comma separated grouping fields"),
    percentiles: Optional[str] = Query(None, description="comma separated percentile levels"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before aggregating"})
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None
    group_list = [g.strip() for g in group_by.split(",") if g.strip()] if group_by else None
    percentile_levels = [float(p.strip()) for p in percentiles.split(",") if p.strip()] if percentiles else None
    try:
        return aggregate_documents(
            state.store,
            state.schema,
            metrics=metric_list,
            field=field,
            group_by=group_list,
            percentiles=percentile_levels,
            limit=limit,
            offset=offset,
        )
    except AggregationError as exc:
        raise HTTPException(status_code=400, detail={"kind": "aggregation_error", "message": str(exc)})


@router.post("/statistics")
def statistics(request: StatisticsRequest):
    try:
        return compute_statistics(request.values, request.percentiles)
    except AggregationError as exc:
        raise HTTPException(status_code=400, detail={"kind": "aggregation_error", "message": str(exc)})


@router.get("/percentile")
def percentile_endpoint(
    field: str = Query(..., description="numeric field to sample"),
    p: float = Query(50.0, ge=0.0, le=100.0),
    filter: Optional[str] = Query(None, description="optional filter JSON expression"),
):
    state = _state()
    if state.schema is None:
        raise HTTPException(status_code=400, detail={"kind": "schema_missing", "message": "configure a schema before aggregating"})
    try:
        result = aggregate_documents(
            state.store,
            state.schema,
            metrics=["percentile"],
            field=field,
            filter_expression=json.loads(filter) if filter else None,
            percentiles=[p],
        )
    except AggregationError as exc:
        raise HTTPException(status_code=400, detail={"kind": "aggregation_error", "message": str(exc)})
    except QueryError as exc:
        raise HTTPException(status_code=400, detail={"kind": "query_error", "message": str(exc)})
    values: List[float] = []
    with state.store.engine.connect() as conn:
        where_sql, params = compile_filter(state.schema, json.loads(filter) if filter else None)
        for row in conn.execute(text(f"SELECT f_{field} FROM documents WHERE {where_sql}"), params).fetchall():
            if row[0] is not None:
                values.append(float(row[0]))
    return {"field": field, "percentile": p, "count": len(values), "value": percentile(values, p)}
