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


def test_cli_init_and_schema():
    state = AppState(db_path=":memory:")
    result = run_command(parse(["--db", ":memory:", "set-schema", "--json", json.dumps(SCHEMA_PAYLOAD)]), state)
    assert result["name"] == "article"
    stats = run_command(parse(["--db", ":memory:", "stats"]), state)
    assert stats["documents"] == 0
    state.close()


def test_cli_add_search_and_query():
    state = AppState(db_path=":memory:")
    run_command(parse(["set-schema", "--json", json.dumps(SCHEMA_PAYLOAD)]), state)
    for doc in DOCS:
        created = run_command(parse(["add", "--json", json.dumps(doc)]), state)
        assert created["external_id"] == doc["slug"]
    search = run_command(parse(["search", "--query", "bm25", "--limit", "5"]), state)
    assert search["total"] == 1
    assert search["results"][0]["external_id"] == "a3"
    query = run_command(parse(["query", "--filter", json.dumps({"field": "word_count", "op": "gte", "value": 20})]), state)
    assert query["total"] == 2
    state.close()


def test_cli_aggregate():
    state = AppState(db_path=":memory:")
    run_command(parse(["set-schema", "--json", json.dumps(SCHEMA_PAYLOAD)]), state)
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
    assert result["total"]["statistics"]["count"] == 3
    state.close()


def test_cli_ingest_and_export(tmp_path):
    state = AppState(db_path=":memory:")
    run_command(parse(["set-schema", "--json", json.dumps(SCHEMA_PAYLOAD)]), state)
    payload = "\n".join(json.dumps(doc) for doc in DOCS)
    result = run_command(parse(["ingest", "--format", "jsonl", "--batch-size", "2"]), state) if False else None
    # ingest reads from stdin; use the pipeline directly through run_command with a file
    file_path = tmp_path / "docs.jsonl"
    file_path.write_text(payload, encoding="utf-8")
    result = run_command(parse(["ingest", "--file", str(file_path), "--format", "jsonl", "--batch-size", "2"]), state)
    assert result["accepted"] == 3
    assert result["rejected"] == 0
    export_path = tmp_path / "dump.json"
    export = run_command(parse(["export", "--file", str(export_path)]), state)
    assert export["count"] == 3
    written = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(written["documents"]) == 3
    state.close()


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
