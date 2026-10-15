import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state import AppState

client = TestClient(app)

SCHEMA_PAYLOAD = {
    "name": "article",
    "fields": [
        {"name": "slug", "type": "text", "required": True, "unique": True, "primary_key": True},
        {"name": "title", "type": "text", "required": True, "min_length": 3},
        {"name": "body", "type": "text", "required": True},
        {"name": "tags", "type": "array", "items": {"name": "item", "type": "text"}},
        {"name": "word_count", "type": "integer", "default": 0, "minimum": 0},
        {"name": "score", "type": "number", "default": 0.5},
        {"name": "published", "type": "boolean", "default": False},
    ],
}


@pytest.fixture()
def fresh_state():
    state = AppState(db_path=":memory:")
    import app.main as main_module

    original = main_module.state
    main_module.state = state
    state.set_schema(SCHEMA_PAYLOAD)
    yield state
    state.close()
    main_module.state = original


def test_set_and_get_schema(fresh_state):
    response = client.get("/schema")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "article"
    assert payload["primary_key"] == "slug"
    assert payload["unique_fields"] == ["slug"]


def test_schema_endpoint_rejects_invalid_definition(fresh_state):
    response = client.post("/schema", json={"fields": [{"name": "x", "type": "blob"}]})
    assert response.status_code == 422
    assert response.json()["detail"]["kind"] == "schema_error"


def test_ingest_json_batch_and_query(fresh_state):
    payload = [
        {"slug": "a1", "title": "Local search", "body": "quick brown fox", "tags": ["search"], "word_count": 3},
        {"slug": "a2", "title": "Inverted index", "body": "terms to documents", "tags": ["index"], "word_count": 4},
    ]
    response = client.post("/ingest?format=json", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["accepted"] == 2 and result["rejected"] == 0

    query = client.post("/query", json={"filter": {"field": "word_count", "op": "gte", "value": 4}, "sort": [{"field": "word_count", "direction": "desc"}]})
    assert query.status_code == 200
    body = query.json()
    assert body["total"] == 1
    assert body["results"][0]["external_id"] == "a2"


def test_ingest_jsonl_quarantine_and_list(fresh_state):
    lines = [
        '{"slug": "b1", "title": "Good row", "body": "content"}',
        "definitely not json",
        '{"slug": "b2", "body": "no title"}',
    ]
    response = client.post("/ingest?format=jsonl", json="\n".join(lines))
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 1 and body["rejected"] == 2 and body["backpressure"] is True

    quarantine = client.get("/quarantine")
    assert quarantine.status_code == 200
    assert quarantine.json()["total"] == 2

    cleared = client.delete("/quarantine")
    assert cleared.json()["removed"] == 2


def test_ingest_csv(fresh_state):
    csv_payload = "slug,title,body,word_count\nc1,CSV row,hello,5\nc2,Another row,world,2\n"
    response = client.post("/ingest", json=csv_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "csv" and body["accepted"] == 2


def test_stream_endpoint(fresh_state):
    lines = [
        '{"slug": "s1", "title": "Streamed", "body": "one"}',
        '{"slug": "s2", "title": "Streamed two", "body": "two"}',
    ]
    response = client.post("/ingest/stream?format=jsonl", json=lines)
    assert response.status_code == 200
    assert response.json()["accepted"] == 2


def test_ingest_requires_schema():
    state = AppState(db_path=":memory:")
    import app.main as main_module

    original = main_module.state
    main_module.state = state
    try:
        response = client.post("/ingest", json=[{"slug": "x"}])
        assert response.status_code == 400
        assert response.json()["detail"]["kind"] == "schema_missing"
        bad = client.post("/query", json={"filter": {"field": "slug", "value": "x"}})
        assert bad.status_code == 400
    finally:
        state.close()
        main_module.state = original


def test_document_crud_endpoints(fresh_state):
    created = client.post("/documents", json={"slug": "d1", "title": "Doc one", "body": "body"})
    assert created.status_code == 201
    fetched = client.get("/documents/d1")
    assert fetched.status_code == 200
    updated = client.put("/documents/d1", json={"title": "Updated title"})
    assert updated.status_code == 200 and updated.json()["payload"]["title"] == "Updated title"
    assert client.get("/documents/missing").status_code == 404
    assert client.delete("/documents/d1").status_code == 204
    assert client.delete("/documents/d1").status_code == 404


def test_document_conflict_and_validation(fresh_state):
    client.post("/documents", json={"slug": "d1", "title": "Doc", "body": "b"})
    conflict = client.post("/documents", json={"slug": "d1", "title": "Dup", "body": "b2"})
    assert conflict.status_code == 409
    invalid = client.post("/documents", json={"slug": "d2", "title": "ab", "body": "x"})
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["kind"] == "schema_error"


def test_search_and_stats_and_list(fresh_state):
    client.post(
        "/ingest",
        json=[
            {"slug": "x1", "title": "Searchable title", "body": "bm25 ranking works", "tags": ["search"]},
            {"slug": "x2", "title": "Other", "body": "nothing relevant", "tags": ["misc"]},
        ],
    )
    search = client.get("/search?q=bm25&limit=10")
    assert search.status_code == 200
    assert search.json()["total"] == 1
    assert search.json()["results"][0]["external_id"] == "x1"

    listing = client.get("/documents?limit=1&offset=0")
    assert listing.status_code == 200 and len(listing.json()["results"]) == 1

    stats = client.get("/stats")
    assert stats.status_code == 200 and stats.json()["documents"] == 2


def test_query_bad_filter_returns_400(fresh_state):
    response = client.post("/query", json={"filter": {"field": "unknown", "op": "eq", "value": 1}})
    assert response.status_code == 400
    assert response.json()["detail"]["kind"] == "query_error"
