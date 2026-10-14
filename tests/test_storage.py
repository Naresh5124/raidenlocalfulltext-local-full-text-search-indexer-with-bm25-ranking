import json
import sqlite3

import pytest

from app.schema import DocumentSchema, FieldDefinition
from app.storage import SQLiteIndexStore, DocumentNotFound, IndexConflictError, tokenize


@pytest.fixture()
def store():
    schema = DocumentSchema(
        name="article",
        fields=[
            FieldDefinition(name="slug", type="text", required=True, unique=True, primary_key=True),
            FieldDefinition(name="title", type="text", required=True),
            FieldDefinition(name="body", type="text", required=True),
            FieldDefinition(name="tags", type="array", items=FieldDefinition(name="item", type="text")),
            FieldDefinition(name="word_count", type="integer", default=0),
        ],
    )
    s = SQLiteIndexStore(db_path=":memory:", schema=schema)
    yield s
    s.close()


def indexed_text(doc: dict) -> str:
    return " ".join([doc["slug"], doc["title"], doc["body"], " ".join(doc["tags"]), "0"])


DOCS = [
    {"slug": "a1", "title": "Local search", "body": "the quick brown fox jumps over the lazy dog", "tags": ["search", "bm25"]},
    {"slug": "a2", "title": "Inverted index", "body": "inverted index maps terms to documents and term frequencies", "tags": ["index"]},
    {"slug": "a3", "title": "Ranking", "body": "bm25 ranking uses term frequency and inverse document frequency", "tags": ["search", "ranking"]},
]


def test_indexed_text_includes_all_fields():
    assert tokenize(indexed_text(DOCS[0]))[:2] == ["a1", "local"]


def test_tables_created(store):
    tables = set(store.index_tables())
    for name in ("documents", "terms", "postings", "doc_lengths", "unique_values", "metadata"):
        assert name in tables


def test_add_and_get_document(store):
    created = store.add_document(DOCS[0])
    assert created["external_id"] == "a1"
    assert created["payload"]["title"] == "Local search"
    fetched = store.get_document("a1")
    assert fetched["doc_id"] == created["doc_id"]
    with pytest.raises(DocumentNotFound):
        store.get_document("missing")


def test_inverted_index_populated(store):
    for doc in DOCS:
        store.add_document(doc)
    hits = store.search_terms("the")
    assert len(hits) == 1
    assert hits[0]["term_freq"] == 2
    assert hits[0]["external_id"] == "a1"
    hits = store.search_terms("search")
    assert {h["external_id"] for h in hits} == {"a1", "a3"}
    stats = store.stats()
    assert stats["documents"] == 3
    assert stats["terms"] > 0
    assert stats["postings"] == sum(len(set(tokenize(indexed_text(d)))) for d in DOCS)
    assert stats["terms"] == len({t for d in DOCS for t in tokenize(indexed_text(d))})


def test_metadata_and_schema_persisted(store):
    store.add_document(DOCS[0])
    stats = store.stats()
    schema_payload = json.loads(stats["schema"])
    assert schema_payload["primary_key"] == "slug"
    assert schema_payload["unique_fields"] == ["slug"]


def test_unique_constraint_conflict(store):
    store.add_document(DOCS[0])
    with pytest.raises(IndexConflictError) as exc:
        store.add_document({"slug": "a1", "title": "dup", "body": "again"})
    assert exc.value.field == "slug"


def test_update_document(store):
    store.add_document(DOCS[0])
    updated = store.update_document("a1", {"title": "Local search engine", "body": "quick fox"})
    assert updated["payload"]["title"] == "Local search engine"
    assert store.search_terms("engine")
    assert not store.search_terms("lazy")


def test_delete_document_removes_postings(store):
    for doc in DOCS:
        store.add_document(doc)
    assert store.delete_document("a2") is True
    assert store.delete_document("a2") is False
    assert not store.search_terms("inverted")
    stats = store.stats()
    assert stats["documents"] == 2
    remaining = [d for d in DOCS if d["slug"] != "a2"]
    expected_terms = {t for d in remaining for t in tokenize(indexed_text(d))}
    assert stats["terms"] == len(expected_terms)
    assert stats["postings"] == sum(len(set(tokenize(indexed_text(d)))) for d in remaining)


def test_secondary_indexes_exist(store):
    store.add_document(DOCS[0])
    names = store.index_tables()
    assert "ux_f_slug" in names
    assert "ix_f_title" in names


def test_bm25_stats(store):
    for doc in DOCS:
        store.add_document(doc)
    stats = store.bm25_stats()
    assert stats["document_count"] == 3
    assert stats["avg_doc_length"] > 0


def test_tokenizer_is_deterministic():
    assert tokenize("The Quick-Brown Fox!") == ["the", "quick", "brown", "fox"]
    assert tokenize("don't") == ["don't"]
    assert tokenize("") == []


def test_persistence_to_sqlite_file(tmp_path):
    db = tmp_path / "index.db"
    schema = DocumentSchema(
        fields=[
            FieldDefinition(name="id", type="text", primary_key=True),
            FieldDefinition(name="body", type="text", required=True),
        ]
    )
    store = SQLiteIndexStore(db_path=str(db), schema=schema)
    store.add_document({"id": "x", "body": "alpha beta gamma"})
    store.close()

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT external_id, payload FROM documents").fetchall()
        assert rows and rows[0][0] == "x"
    finally:
        conn.close()

    reopened = SQLiteIndexStore(db_path=str(db), schema=schema)
    assert reopened.get_document("x")["payload"]["body"] == "alpha beta gamma"
    assert reopened.stats()["documents"] == 1
    reopened.close()
