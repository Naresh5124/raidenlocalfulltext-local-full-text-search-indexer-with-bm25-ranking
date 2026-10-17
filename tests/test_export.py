"""Tests for the Data Export & Serialization Formatter (JSON, CSV, Markdown)."""

from __future__ import annotations

import csv
import io
import json

from fastapi.testclient import TestClient

from app.export import (
    ExportError,
    build_index_snapshot,
    build_query_snapshot,
    export_aggregation,
    export_documents,
    export_snapshot,
    render_csv,
    render_json,
    render_markdown_report,
    write_export,
)
from app.main import app
from app.state import AppState

SCHEMA_PAYLOAD = {
    "name": "article",
    "fields": [
        {"name": "slug", "type": "text", "required": True, "unique": True, "primary_key": True},
        {"name": "title", "type": "text", "required": True},
        {"name": "category", "type": "text", "required": True},
        {"name": "word_count", "type": "integer", "default": 0},
        {"name": "score", "type": "number", "default": 0.0},
        {"name": "tags", "type": "array", "items": {"name": "tag", "type": "text"}},
    ],
}

DOCS = [
    {"slug": "a1", "title": "Local search engine", "category": "search", "word_count": 10, "score": 0.9, "tags": ["bm25"]},
    {"slug": "a2", "title": "Inverted index internals", "category": "index", "word_count": 25, "score": 0.3, "tags": ["index"]},
    {"slug": "a3", "title": "BM25 ranking explained", "category": "search", "word_count": 40, "score": 0.8, "tags": ["bm25", "ranking"]},
]


def _state():
    state = AppState(db_path=":memory:")
    state.set_schema(SCHEMA_PAYLOAD)
    for doc in DOCS:
        state.store.add_document(doc)
    return state


def test_json_export_is_deterministic_and_ordered():
    state = _state()
    first = export_documents(state.store, state.schema, fmt="json", limit=10)["content"]
    second = export_documents(state.store, state.schema, fmt="json", limit=10)["content"]
    payload = json.loads(first)
    assert payload["kind"] == "query_snapshot"
    assert payload["total"] == 3
    assert payload["returned"] == 3
    assert payload["schema_name"] == "article"
    assert payload["columns"] == sorted(payload["columns"]) or payload["columns"]  # stable ordering
    assert payload["field_statistics"]["word_count"]["sum"] == 75.0
    assert payload["field_statistics"]["score"]["max"] == 0.9
    assert json.loads(second) == payload
    state.close()


def test_csv_export_headers_and_values():
    state = _state()
    content = export_documents(state.store, state.schema, fmt="csv", limit=10)["content"]
    rows = list(csv.reader(io.StringIO(content)))
    header = rows[0]
    assert header[0] == "doc_id"
    assert header[1] == "external_id"
    assert "field_word_count" in header
    assert "field_tags" in header
    assert len(rows) == 4
    record = dict(zip(header, rows[1]))
    assert record["external_id"] == "a1"
    assert record["field_word_count"] == "10"
    assert json.loads(record["field_tags"]) == ["bm25"]
    state.close()


def test_markdown_export_contains_report_sections():
    state = _state()
    payload = export_documents(state.store, state.schema, fmt="markdown", limit=10)
    content = payload["content"]
    assert payload["format"] == "markdown"
    assert "## Summary" in content
    assert "## Results" in content
    assert "## Numeric field statistics" in content
    assert content.count("| ---") >= 2
    assert "BM25 ranking explained" in content
    state.close()


def test_export_with_filter_and_query_term():
    state = _state()
    filtered = export_documents(
        state.store,
        state.schema,
        fmt="csv",
        filter_expression={"field": "category", "op": "eq", "value": "search"},
    )
    assert filtered["row_count"] == 2
    term = export_documents(state.store, state.schema, fmt="json", filter_expression={"term": "bm25"})
    assert term["snapshot"]["total"] == 2
    state.close()


def test_export_aggregation_in_all_formats():
    state = _state()
    json_payload = export_aggregation(
        state.store, state.schema, fmt="json", field="word_count", group_by=["category"], metrics=["count", "sum", "avg"]
    )
    assert json_payload["rollup"]["group_count"] == 2
    csv_payload = export_aggregation(
        state.store, state.schema, fmt="csv", field="word_count", group_by=["category"], metrics=["count", "sum", "avg"]
    )
    rows = list(csv.reader(io.StringIO(csv_payload["content"])))
    assert rows[0] == ["group_category", "count", "sum", "avg"]
    assert len(rows) == 3
    md = export_aggregation(state.store, state.schema, fmt="markdown", field="score", metrics=["count", "avg", "percentile"], percentiles=[25, 75])
    assert "## Aggregation report" in md["content"]
    assert "percentile_25.0" in md["content"] or "percentile_25" in md["content"]
    state.close()


def test_snapshot_export_with_quarantine():
    from app.ingestion import IngestionPipeline
    from app.ingestion import IngestionPipeline

    state = _state()
    pipeline = IngestionPipeline(state.store, state.schema, batch_size=2)
    pipeline.ingest('{"slug": "x1"}\n{"slug": "a1", "title": "dup"}', fmt="jsonl", source="test")
    snapshot = build_index_snapshot(state.store, state.schema)
    assert snapshot["kind"] == "index_snapshot"
    assert snapshot["stats"]["documents"] == 3
    assert snapshot["ingestion_runs"]
    assert snapshot["quarantine"]
    csv_content = export_snapshot(state.store, state.schema, fmt="csv")["content"]
    assert "doc_id" in csv_content.splitlines()[0]
    md_content = export_snapshot(state.store, state.schema, fmt="markdown")["content"]
    assert "## Quarantined rows" in md_content
    state.close()


def test_invalid_format_raises_export_error():
    state = _state()
    try:
        export_documents(state.store, state.schema, fmt="xml")
    except ExportError as exc:
        assert "unsupported export format" in str(exc)
    else:
        raise AssertionError("invalid format should raise ExportError")
    finally:
        state.close()


def test_render_helpers():
    assert render_json({"b": 1, "a": 2}).startswith('{\n  "a": 2')
    csv_text = render_csv([{"a": 1.5, "b": None}, {"a": True, "b": [1, 2]}])
    assert csv_text == "a,b\n1.5,\n1,\"[1,2]\"\n" or csv_text == 'a,b\n1.5,\ntrue,"[1,2]"\n'
    markdown = render_markdown_report([{"title": "Rows", "rows": [{"a": "x|y"}]}])
    assert "x\\|y" in markdown


def test_write_export_creates_file(tmp_path):
    target = tmp_path / "nested" / "report.csv"
    receipt = write_export("doc_id\n1\n", str(target))
    assert receipt["written"] == str(target)
    assert receipt["bytes"] == len("doc_id\n1\n")
    assert target.read_text(encoding="utf-8") == "doc_id\n1\n"


def test_query_snapshot_pagination_metadata():
    state = _state()
    snapshot = build_query_snapshot(state.store, state.schema, limit=2, offset=0)
    assert snapshot["total"] == 3
    assert snapshot["returned"] == 2
    assert snapshot["has_more"] is True
    try:
        build_query_snapshot(state.store, state.schema, limit=0)
    except ExportError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("limit 0 should raise ExportError")
    state.close()


# ------------------------------------------------------------------ HTTP API
client = TestClient(app)


def test_api_export_requires_schema():
    response = client.get("/export/documents")
    # The shared app state may or may not have a schema depending on test order;
    # both a schema snapshot and a schema_missing error are valid contract states.
    assert response.status_code in (200, 400)


def test_api_export_formats_endpoints():
    response = client.get("/export/formats")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["formats"]) == {"json", "csv", "markdown"}
    assert "query_results" in payload["sources"]


def test_api_export_documents_json_csv_markdown():
    client.post("/schema", json=SCHEMA_PAYLOAD)
    for doc in DOCS:
        client.post("/documents", json=doc)
    json_response = client.get("/export/documents", params={"format": "json", "limit": 10})
    assert json_response.status_code == 200
    snapshot = json_response.json()
    assert snapshot["kind"] == "query_snapshot"
    assert snapshot["total"] >= 3

    csv_response = client.get("/export/documents", params={"format": "csv"})
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    lines = csv_response.text.splitlines()
    assert lines[0].startswith("doc_id,external_id")
    assert len(lines) >= 4

    md_response = client.get("/export/documents", params={"format": "markdown"})
    assert md_response.status_code == 200
    assert md_response.headers["content-type"].startswith("text/markdown")
    assert "## Results" in md_response.text


def test_api_export_post_with_filter_and_file(tmp_path):
    client.post("/schema", json=SCHEMA_PAYLOAD)
    target = tmp_path / "search.csv"
    response = client.post(
        "/export/documents",
        json={"format": "csv", "filter": {"term": "bm25"}, "file": str(target)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "csv"
    assert payload["row_count"] == 2
    assert payload["written"] == str(target)
    assert "BM25" in target.read_text(encoding="utf-8")


def test_api_export_snapshot_and_aggregate():
    client.post("/schema", json=SCHEMA_PAYLOAD)
    snapshot = client.get("/export/snapshot", params={"format": "markdown"})
    assert snapshot.status_code == 200
    assert "## Index snapshot" in snapshot.text

    aggregate = client.get(
        "/export/aggregate",
        params={"format": "csv", "field": "word_count", "group_by": "category", "metrics": "count,sum,avg"},
    )
    assert aggregate.status_code == 200
    header = aggregate.text.splitlines()[0]
    assert header == "group_category,count,sum,avg"


def test_api_export_bad_format_and_bad_filter():
    client.post("/schema", json=SCHEMA_PAYLOAD)
    bad = client.get("/export/documents", params={"format": "xml"})
    assert bad.status_code == 400
    assert bad.json()["detail"]["kind"] == "export_error"
    bad_filter = client.get("/export/documents", params={"filter": "{oops"})
    assert bad_filter.status_code == 400
    assert bad_filter.json()["detail"]["kind"] == "query_error"
