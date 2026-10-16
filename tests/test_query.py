import pytest

from app.query import QueryError, compile_filter, compile_sort, execute_query
from app.schema import DocumentSchema, FieldDefinition
from app.storage import SQLiteIndexStore

SCHEMA = DocumentSchema(
    name="article",
    fields=[
        FieldDefinition(name="slug", type="text", required=True, unique=True, primary_key=True),
        FieldDefinition(name="title", type="text", required=True),
        FieldDefinition(name="body", type="text", required=True),
        FieldDefinition(name="tags", type="array", items=FieldDefinition(name="item", type="text")),
        FieldDefinition(name="word_count", type="integer", default=0),
        FieldDefinition(name="score", type="number", default=0.5),
        FieldDefinition(name="published", type="boolean", default=False),
        FieldDefinition(name="created_at", type="datetime", nullable=True),
    ],
)

DOCS = [
    {"slug": "a1", "title": "Local search", "body": "the quick brown fox", "tags": ["search", "bm25"], "word_count": 4, "score": 0.9, "published": True, "created_at": "2024-01-01T00:00:00Z"},
    {"slug": "a2", "title": "Inverted index", "body": "index maps terms to documents", "tags": ["index"], "word_count": 6, "score": 0.2, "published": False, "created_at": "2024-02-01T00:00:00Z"},
    {"slug": "a3", "title": "Ranking engine", "body": "bm25 ranking uses term frequency", "tags": ["search", "ranking"], "word_count": 6, "score": 0.7, "published": True, "created_at": "2024-03-01T00:00:00Z"},
    {"slug": "a4", "title": "Notes", "body": "short note", "tags": [], "word_count": 2, "score": 0.1, "published": False, "created_at": None},
]


@pytest.fixture()
def store():
    s = SQLiteIndexStore(db_path=":memory:", schema=SCHEMA)
    for doc in DOCS:
        s.add_document(doc)
    yield s
    s.close()


def external_ids(result):
    return {r["external_id"] for r in result["results"]}


def test_equality_and_shorthand(store):
    result = execute_query(store, SCHEMA, {"field": "slug", "value": "a1"})
    assert result["total"] == 1
    assert external_ids(result) == {"a1"}

    result = execute_query(store, SCHEMA, {"field": "published", "op": "eq", "value": True})
    assert external_ids(result) == {"a1", "a3"}

    result = execute_query(store, SCHEMA, {"field": "published", "op": "ne", "value": True})
    assert external_ids(result) == {"a2", "a4"}


def test_range_predicates(store):
    result = execute_query(store, SCHEMA, {"field": "word_count", "op": "gte", "value": 6})
    assert external_ids(result) == {"a2", "a3"}
    result = execute_query(store, SCHEMA, {"field": "score", "op": "lt", "value": 0.5})
    assert external_ids(result) == {"a2", "a4"}
    result = execute_query(store, SCHEMA, {"field": "created_at", "op": "gte", "value": "2024-02-01T00:00:00+00:00"})
    assert external_ids(result) == {"a2", "a3"}


def test_compound_and_or_not(store):
    expr = {
        "and": [
            {"field": "published", "op": "eq", "value": True},
            {"field": "word_count", "op": "gte", "value": 4},
        ]
    }
    assert external_ids(execute_query(store, SCHEMA, expr)) == {"a1", "a3"}

    expr = {"or": [{"field": "slug", "op": "in", "value": ["a1", "a4"]}, {"field": "score", "op": "gt", "value": 0.8}]}
    assert external_ids(execute_query(store, SCHEMA, expr)) == {"a1", "a4"}

    expr = {"not": {"field": "published", "op": "eq", "value": True}}
    assert external_ids(execute_query(store, SCHEMA, expr)) == {"a2", "a4"}

    nested = {"and": [{"or": [{"field": "slug", "value": "a1"}, {"field": "slug", "value": "a2"}]}, {"not": {"field": "slug", "value": "a1"}}]}
    assert external_ids(execute_query(store, SCHEMA, nested)) == {"a2"}


def test_full_text_term_filter(store):
    result = execute_query(store, SCHEMA, {"term": "search"})
    assert external_ids(result) == {"a1", "a3"}
    result = execute_query(store, SCHEMA, {"term": "quick brown", "mode": "all"})
    assert external_ids(result) == {"a1"}
    result = execute_query(store, SCHEMA, {"term": "quick brown", "mode": "any"})
    assert external_ids(result) == {"a1"}
    result = execute_query(store, SCHEMA, {"term": "nonexistent"})
    assert result["total"] == 0


def test_contains_and_exists_and_missing(store):
    assert external_ids(execute_query(store, SCHEMA, {"field": "title", "op": "contains", "value": "search"})) == {"a1"}
    assert external_ids(execute_query(store, SCHEMA, {"field": "created_at", "op": "exists"})) == {"a1", "a2", "a3"}
    assert external_ids(execute_query(store, SCHEMA, {"field": "created_at", "op": "missing"})) == {"a4"}


def test_sorting(store):
    result = execute_query(store, SCHEMA, None, sort=[{"field": "created_at", "direction": "desc"}])
    assert [r["external_id"] for r in result["results"]][0] == "a3"
    result = execute_query(store, SCHEMA, None, sort=[{"field": "score", "direction": "desc"}, {"field": "doc_id", "direction": "asc"}])
    assert [r["external_id"] for r in result["results"]][:2] == ["a1", "a3"]


def test_pagination(store):
    result = execute_query(store, SCHEMA, None, sort=[{"field": "doc_id", "direction": "asc"}], limit=2, offset=0)
    assert result["total"] == 4
    assert result["returned"] == 2
    assert result["has_more"] is True
    assert [r["external_id"] for r in result["results"]] == ["a1", "a2"]

    result = execute_query(store, SCHEMA, None, sort=[{"field": "doc_id", "direction": "asc"}], limit=2, offset=3)
    assert result["returned"] == 1
    assert result["has_more"] is False
    assert result["results"][0]["external_id"] == "a4"


def test_invalid_queries_raise(store):
    with pytest.raises(QueryError):
        execute_query(store, SCHEMA, {"field": "bogus", "op": "eq", "value": 1})
    with pytest.raises(QueryError):
        execute_query(store, SCHEMA, {"field": "slug", "op": "bogus", "value": 1})
    with pytest.raises(QueryError):
        execute_query(store, SCHEMA, {"and": []})
    with pytest.raises(QueryError):
        execute_query(store, SCHEMA, {"field": "word_count", "op": "gt"})
    with pytest.raises(QueryError):
        execute_query(store, SCHEMA, None, limit=0)
    with pytest.raises(QueryError):
        compile_sort([{"field": "title", "direction": "sideways"}])
    with pytest.raises(QueryError):
        compile_sort([{"field": "not_sortable"}])


def test_compile_filter_returns_parameterised_sql(store):
    sql, params = compile_filter(SCHEMA, {"field": "word_count", "op": "between_placeholder"})
    assert sql == "1 = 1" and params == {}


def test_query_over_empty_store(store):
    result = execute_query(store, SCHEMA, {"term": "nothing"})
    assert result["total"] == 0 and result["results"] == [] and result["has_more"] is False
