import json
import subprocess
import sys

import pytest

from app.cli import build_parser, main, run_command
from app.state import AppState

SCHEMA_PAYLOAD = {
    "name": "article",
    "fields": [
        {"name": "slug", "type": "text", "required": True, "unique": True, "primary_key": True},
        {"name": "title", "type": "text", "required": True},
        {"name": "category", "type": "text", "required": True},
        {"name": "word_count", "type": "integer", "default": 0},
        {"name": "score", "type": "number", "default": 0.0},
    ],
}

DOCS = [
    {"slug": "a1", "title": "Local search engine", "category": "search", "word_count": 10, "score": 0.9},
    {"slug": "a2", "title": "Inverted index internals", "category": "index", "word_count": 25, "score": 0.3},
    {"slug": "a3", "title": "BM25 ranking explained", "category": "search", "word_count": 40, "score": 0.8},
]


def parse(argv):
    return build_parser().parse_args(argv)


@pytest.fixture()
def state():
    s = AppState(db_path=":memory:")
    s.set_schema(SCHEMA_PAYLOAD)
    yield s
    s.close()


def test_cli_stats_reports_documents(state):
    stats = run_command(parse(["stats"]), state)
    assert stats["documents"] == 0
    assert stats["terms"] == 0


def test_cli_add_search_and_query(state):
    for doc in DOCS:
        created = run_command(parse(["add", "--json", json.dumps(doc)]), state)
        assert created["external_id"] == doc["slug"]
    search = run_command(parse(["search", "--query", "bm25", "--limit", "5"]), state)
    assert search["total"] == 1
    assert search["results"][0]["external_id"] == "a3"
    query = run_command(parse(["query", "--filter", json.dumps({"field": "word_count", "op": "gte", "value": 20})]), state)
    assert query["total"] == 2
    sorted_query = run_command(parse(["query", "--filter", json.dumps({"field": "category", "op": "exists"}), "--sort", "created_at"]), state)
    assert sorted_query["total"] == 3
    assert len(sorted_query["results"]) == 3
    assert sorted_query["sort"] == [{"field": "created_at", "direction": "desc"}]


def test_cli_aggregate(state):
    for doc in DOCS:
        run_command(parse(["add", "--json", json.dumps(doc)]), state)
    result = run_command(
        parse(
            [
                "aggregate",
                "--field",
                "word_count",
                "--metrics",
                "count,sum,min,max,avg,percentile",
                "--group-by",
                "category",
                "--percentiles",
                "25,75",
            ]
        ),
        state,
    )
    assert result["group_count"] == 2
    by_key = {tuple(g["key"].values()): g for g in result["groups"]}
    search = by_key[("search",)]
    assert search["count"] == 2
    assert search["metrics"]["sum"] == 50.0
    assert search["metrics"]["avg"] == 25.0
    assert search["metrics"]["percentiles"][25.0] == 17.5
    assert search["metrics"]["percentiles"][75.0] == 32.5
    assert search["metrics"]["median"] == 25.0
    assert result["total"]["statistics"]["count"] == 3
    assert result["total"]["statistics"]["min"] == 10.0
    assert result["total"]["statistics"]["max"] == 40.0


def test_cli_aggregate_with_filter(state):
    for doc in DOCS:
        run_command(parse(["add", "--json", json.dumps(doc)]), state)
    result = run_command(
        parse(
            [
                "aggregate",
                "--field",
                "score",
                "--metrics",
                "count,sum,avg",
                "--filter",
                json.dumps({"field": "category", "op": "eq", "value": "index"}),
            ]
        ),
        state,
    )
    assert result["group_count"] == 1
    metrics = result["groups"][0]["metrics"]
    assert metrics["count"] == 1
    assert metrics["sum"] == 0.3
    assert result["total"]["documents"] == 1


def test_cli_ingest_and_export(state, tmp_path):
    file_path = tmp_path / "docs.jsonl"
    file_path.write_text("\n".join(json.dumps(doc) for doc in DOCS), encoding="utf-8")
    result = run_command(parse(["ingest", "--file", str(file_path), "--format", "jsonl", "--batch-size", "2"]), state)
    assert result["accepted"] == 3
    assert result["rejected"] == 0
    assert result["backpressure"] is False
    export_path = tmp_path / "dump.json"
    export = run_command(parse(["export", "--file", str(export_path)]), state)
    assert export["count"] == 3
    written = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(written["documents"]) == 3


def test_cli_requires_schema(state):
    empty = AppState(db_path=":memory:")
    try:
        run_command(parse(["aggregate", "--field", "word_count"]), empty)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("aggregate should require a schema")
    finally:
        empty.close()


def test_cli_main_returns_error_code_for_bad_payload(capsys):
    code = main(["--db", ":memory:", "query", "--filter", "{not json}"])
    assert code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err


def test_cli_module_runs_as_script(tmp_path):
    db_path = tmp_path / "cli.db"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA_PAYLOAD), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "app.cli", "--db", str(db_path), "init-db", "--schema", str(schema_path)],
        check=True,
        capture_output=True,
    )
    output = subprocess.run(
        [sys.executable, "-m", "app.cli", "--db", str(db_path), "stats"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.stdout)
    assert payload["documents"] == 0
