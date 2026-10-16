import pytest

from app.aggregation import (
    AggregationError,
    aggregate_documents,
    compute_statistics,
    percentile,
)
from app.schema import DocumentSchema, FieldDefinition
from app.storage import SQLiteIndexStore

SCHEMA = DocumentSchema(
    name="article",
    fields=[
        FieldDefinition(name="slug", type="text", required=True, unique=True, primary_key=True),
        FieldDefinition(name="title", type="text", required=True),
        FieldDefinition(name="category", type="text", required=True),
        FieldDefinition(name="word_count", type="integer", default=0),
        FieldDefinition(name="score", type="number", default=0.0),
        FieldDefinition(name="published", type="boolean", default=False),
    ],
)

DOCS = [
    {"slug": "a1", "title": "Alpha", "category": "search", "word_count": 10, "score": 0.9, "published": True},
    {"slug": "a2", "title": "Beta", "category": "search", "word_count": 20, "score": 0.4, "published": False},
    {"slug": "a3", "title": "Gamma", "category": "index", "word_count": 30, "score": 0.7, "published": True},
    {"slug": "a4", "title": "Delta", "category": "index", "word_count": 40, "score": 0.1, "published": False},
    {"slug": "a5", "title": "Epsilon", "category": "index", "word_count": 50, "score": 0.5, "published": True},
]


@pytest.fixture()
def store():
    s = SQLiteIndexStore(db_path=":memory:", schema=SCHEMA)
    for doc in DOCS:
        s.add_document(doc)
    yield s
    s.close()


def test_percentile_interpolation():
    assert percentile([1, 2, 3, 4], 0) == 1.0
    assert percentile([1, 2, 3, 4], 100) == 4.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30], 25) == 15.0
    assert percentile([], 50) is None
    with pytest.raises(AggregationError):
        percentile([1, 2], 150)


def test_compute_statistics():
    stats = compute_statistics([1, 2, 3, 4, 5], percentiles=[25, 75])
    assert stats["count"] == 5
    assert stats["sum"] == 15
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["avg"] == 3.0
    assert stats["median"] == 3.0
    assert stats["percentiles"][25.0] == 2.0
    assert stats["percentiles"][75.0] == 4.0


def test_compute_statistics_ignores_nulls_and_booleans():
    stats = compute_statistics([None, True, 2, 4, "x"])
    assert stats["count"] == 4
    assert stats["sum"] == 6.0
    assert stats["avg"] == 3.0


def test_rollup_without_grouping(store):
    result = aggregate_documents(store, SCHEMA, field="word_count", metrics=["count", "sum", "min", "max", "avg", "percentile"])
    assert result["group_count"] == 1
    metrics = result["groups"][0]["metrics"]
    assert metrics["count"] == 5
    assert metrics["sum"] == 150.0
    assert metrics["min"] == 10.0
    assert metrics["max"] == 50.0
    assert metrics["avg"] == 30.0
    assert metrics["percentiles"][50.0] == 30.0
    assert metrics["median"] == 30.0
    assert result["total"]["documents"] == 5
    assert result["total"]["statistics"]["sum"] == 150.0


def test_grouped_aggregation(store):
    result = aggregate_documents(
        store,
        SCHEMA,
        field="score",
        group_by=["category"],
        metrics=["count", "sum", "min", "max", "avg", "percentile"],
        percentiles=[25, 75],
    )
    keys = {tuple(g["key"].values()) for g in result["groups"]}
    assert keys == {("search",), ("index",)}
    by_key = {tuple(g["key"].values()): g for g in result["groups"]}
    search = by_key[("search",)]
    assert search["count"] == 2
    assert search["metrics"]["sum"] == 1.3
    assert search["metrics"]["min"] == 0.4
    assert search["metrics"]["max"] == 0.9
    assert search["metrics"]["avg"] == 0.65
    assert search["metrics"]["percentiles"][75.0] == 0.775
    index = by_key[("index",)]
    assert index["count"] == 3
    assert index["metrics"]["percentiles"][25.0] == 0.3
    assert index["metrics"]["median"] == 0.5


def test_aggregation_with_filter(store):
    result = aggregate_documents(
        store,
        SCHEMA,
        field="word_count",
        group_by=["published"],
        filter_expression={"field": "word_count", "op": "gte", "value": 20},
    )
    by_key = {tuple(g["key"].values()): g for g in result["groups"]}
    assert by_key[(1,)]["count"] == 2  # a3 (30), a5 (50)
    assert by_key[(1,)]["metrics"]["sum"] == 80.0
    assert by_key[(0,)]["count"] == 2  # a2 (20), a4 (40)
    assert by_key[(0,)]["metrics"]["avg"] == 30.0


def test_multiple_group_fields(store):
    result = aggregate_documents(store, SCHEMA, field="word_count", group_by=["category", "published"], metrics=["count", "avg"])
    assert result["group_count"] == 4
    for group in result["groups"]:
        assert group["key"]["category"] in {"search", "index"}
        assert group["count"] >= 1


def test_invalid_metric_and_field(store):
    with pytest.raises(AggregationError):
        aggregate_documents(store, SCHEMA, field="word_count", metrics=["stddev"])
    with pytest.raises(AggregationError):
        aggregate_documents(store, SCHEMA, field="nonexistent")
    with pytest.raises(AggregationError):
        aggregate_documents(store, SCHEMA, field="word_count", percentiles=[150])
    with pytest.raises(AggregationError):
        aggregate_documents(store, SCHEMA, field="word_count", limit=0)
