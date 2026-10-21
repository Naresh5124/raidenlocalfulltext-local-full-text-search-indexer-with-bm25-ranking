import pytest

from app.ingestion import IngestionError, IngestionPipeline
from app.schema import DocumentSchema, FieldDefinition
from app.storage import SQLiteIndexStore

SCHEMA = DocumentSchema(
    name="article",
    fields=[
        FieldDefinition(name="slug", type="text", required=True, unique=True, primary_key=True),
        FieldDefinition(name="title", type="text", required=True, min_length=3),
        FieldDefinition(name="body", type="text", required=True),
        FieldDefinition(name="tags", type="array", items=FieldDefinition(name="item", type="text")),
        FieldDefinition(name="word_count", type="integer", default=0, minimum=0),
    ],
)


@pytest.fixture()
def pipeline():
    store = SQLiteIndexStore(db_path=":memory:", schema=SCHEMA)
    yield IngestionPipeline(store, SCHEMA, batch_size=2)
    store.close()


def test_json_array_batch_ingestion(pipeline):
    payload = [
        {"slug": "a1", "title": "First post", "body": "alpha beta", "tags": ["search"]},
        {"slug": "a2", "title": "Second post", "body": "gamma delta", "tags": ["index"]},
        {"slug": "a3", "title": "Third post", "body": "epsilon"},
    ]
    result = pipeline.ingest(payload, fmt="json", source="batch")
    assert result["received"] == 3
    assert result["accepted"] == 3
    assert result["rejected"] == 0
    assert result["format"] == "json"
    assert result["batch_size"] == 2
    assert {d["external_id"] for d in result["documents"]} == {"a1", "a2", "a3"}
    assert pipeline.store.stats()["documents"] == 3


def test_json_object_single_document(pipeline):
    result = pipeline.ingest({"slug": "solo", "title": "Solo doc", "body": "only one"}, fmt="json")
    assert result["accepted"] == 1
    assert pipeline.store.get_document("solo")["payload"]["word_count"] == 0


def test_jsonl_ingestion_with_malformed_rows_quarantined(pipeline):
    payload = "\n".join(
        [
            '{"slug": "j1", "title": "Valid one", "body": "text here"}',
            "this is not json at all",
            '{"slug": "j2", "body": "missing title"}',
            '{"slug": "j1", "title": "Duplicate", "body": "conflict"}',
        ]
    )
    result = pipeline.ingest(payload, fmt="jsonl", source="stream-file")
    assert result["received"] == 4
    assert result["accepted"] == 1
    assert result["rejected"] == 3
    assert result["backpressure"] is True

    quarantined = pipeline.store.list_quarantine()
    reasons = {q["reason"] for q in quarantined}
    assert {"parse_error", "validation_error", "conflict"} <= reasons
    assert any("missing_field" in {e["code"] for e in q["errors"]} for q in quarantined if q["reason"] == "validation_error")


def test_csv_ingestion(pipeline):
    payload = "slug,title,body,word_count\na1,Hello world,first row,3\na2,Second row,another row,4\n"
    result = pipeline.ingest(payload, fmt="csv", source="csv-file")
    assert result["format"] == "csv"
    assert result["accepted"] == 2
    assert result["rejected"] == 0
    doc = pipeline.store.get_document("a1")
    assert doc["payload"]["word_count"] == 3


def test_csv_malformed_row_quarantined(pipeline):
    payload = "slug,title,body\na1,Good row,body text\nbadrow,ab,short title\n"
    result = pipeline.ingest(payload, fmt="csv")
    assert result["accepted"] == 1
    assert result["rejected"] == 1
    assert result["quarantine"][0]["line_number"] == 3


def test_format_auto_detection(pipeline):
    assert pipeline.ingest([{"slug": "d1", "title": "auto", "body": "x"}])["format"] == "json"
    assert pipeline.ingest('{"slug": "d2", "title": "auto", "body": "y"}')["format"] == "json"
    assert pipeline.ingest('{"slug": "d3", "title": "line", "body": "z"}\n')["format"] == "jsonl"


def test_unsupported_format_and_bad_payload(pipeline):
    with pytest.raises(IngestionError):
        pipeline.ingest("slug,title\na,b", fmt="xml")
    with pytest.raises(IngestionError):
        pipeline.ingest("{not valid json", fmt="json")


def test_streaming_ingestion_jsonl(pipeline):
    lines = [
        '{"slug": "s1", "title": "Stream one", "body": "b1"}',
        '{"slug": "s2", "title": "Stream two", "body": "b2"}',
        "garbage line",
        '{"slug": "s3", "title": "Stream three", "body": "b3"}',
    ]
    result = pipeline.ingest_stream(lines, fmt="jsonl", source="live-stream")
    assert result["received"] == 4
    assert result["accepted"] == 3
    assert result["rejected"] == 1
    assert pipeline.store.stats()["documents"] == 3


def test_streaming_ingestion_csv(pipeline):
    lines = ["slug,title,body", "c1,Stream csv,row one", "c2,Another,Row two"]
    result = pipeline.ingest_stream(lines, fmt="csv", source="live-csv")
    assert result["accepted"] == 2
    assert pipeline.store.get_document("c2")["payload"]["title"] == "Another"


def test_backpressure_flag_not_triggered_when_clean(pipeline):
    payload = "\n".join(
        f'{{"slug": "b{i}", "title": "batch {i}", "body": "content"}}' for i in range(10)
    )
    result = pipeline.ingest(payload, fmt="jsonl", batch_size_override=None) if False else pipeline.ingest(payload, fmt="jsonl")
    assert result["backpressure"] is False
    assert result["accepted"] == 10


def test_quarantine_persisted_and_cleared(pipeline):
    pipeline.ingest("{bad json\n{\"slug\": \"q1\", \"title\": \"Valid title\", \"body\": \"b\"}", fmt="jsonl")
    assert pipeline.store.quarantine_count() == 1
    removed = pipeline.store.clear_quarantine()
    assert removed == 1
    assert pipeline.store.quarantine_count() == 0
