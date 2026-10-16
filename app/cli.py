"""Command-line interface for RaidenLocalFullText.

Provides clean, composable commands that reuse the library layer (schema,
storage, ingestion, query and aggregation modules) so the same operations are
available programmatically and from a shell:

    python -m app.cli init-db --db raiden_local_fulltext.db
    python -m app.cli set-schema --schema schema.json
    python -m app.cli add --json '{"slug": "a1", "title": "Hello", "body": "world"}'
    python -m app.cli ingest --file docs.jsonl --format jsonl
    python -m app.cli search --query "bm25 ranking" --limit 10
    python -m app.cli query --filter '{"field": "word_count", "op": "gte", "value": 5}'
    python -m app.cli aggregate --field word_count --metrics count,sum,min,max,avg,percentile
    python -m app.cli aggregate --field score --group-by tags --percentiles 25,75
    python -m app.cli stats
    python -m app.cli export --file dump.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from app.aggregation import AggregationError, aggregate_documents
from app.ingestion import IngestionError, IngestionPipeline
from app.query import QueryError, execute_query
from app.schema import ValidationError
from app.storage import DocumentNotFound, IndexConflictError
from app.state import AppState

VALID_ACTIONS = {
    "init-db",
    "set-schema",
    "add",
    "ingest",
    "search",
    "query",
    "aggregate",
    "stats",
    "export",
}


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_json_argument(value: Optional[str], file_path: Optional[str]) -> Any:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if value is not None:
        return json.loads(value)
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return json.loads(data)
    raise ValueError("no payload provided: use --json, --file or stdin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raiden-cli",
        description="RaidenLocalFullText local full-text search CLI",
    )
    parser.add_argument("--db", default=None, help="SQLite database path (default: raiden_local_fulltext.db)")
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init-db", help="create the SQLite tables and show the schema status")
    init_parser.add_argument("--schema", default=None, help="optional schema JSON file to install")

    schema_parser = subparsers.add_parser("set-schema", help="install a document schema")
    schema_parser.add_argument("--schema", default=None, help="schema JSON file")
    schema_parser.add_argument("--json", default=None, help="inline schema JSON")

    add_parser = subparsers.add_parser("add", help="add a single document")
    add_parser.add_argument("--json", default=None, help="inline document JSON")
    add_parser.add_argument("--file", default=None, help="document JSON file")

    ingest_parser = subparsers.add_parser("ingest", help="ingest documents from a file or stdin")
    ingest_parser.add_argument("--file", default=None, help="input file (json, jsonl or csv)")
    ingest_parser.add_argument("--format", default=None, choices=["json", "jsonl", "csv"])
    ingest_parser.add_argument("--source", default="cli")
    ingest_parser.add_argument("--batch-size", type=int, default=100)
    ingest_parser.add_argument("--max-quarantine-ratio", type=float, default=0.5)

    search_parser = subparsers.add_parser("search", help="full-text search with BM25 ranking")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=50)
    search_parser.add_argument("--offset", type=int, default=0)
    search_parser.add_argument("--sort", default=None, help="comma separated sort fields")

    query_parser = subparsers.add_parser("query", help="run a structured filter query")
    query_parser.add_argument("--filter", default=None, help="filter JSON expression")
    query_parser.add_argument("--sort", default=None, help="comma separated sort fields, e.g. created_at,doc_id")
    query_parser.add_argument("--limit", type=int, default=50)
    query_parser.add_argument("--offset", type=int, default=0)

    agg_parser = subparsers.add_parser("aggregate", help="compute rollup statistics across groups")
    agg_parser.add_argument("--field", default=None, help="numeric field to aggregate")
    agg_parser.add_argument("--metrics", default="count,sum,min,max,avg,percentile")
    agg_parser.add_argument("--group-by", default=None, help="comma separated grouping fields")
    agg_parser.add_argument("--percentiles", default="50", help="comma separated percentile levels")
    agg_parser.add_argument("--filter", default=None, help="optional filter JSON expression")
    agg_parser.add_argument("--limit", type=int, default=100)
    agg_parser.add_argument("--offset", type=int, default=0)

    stats_parser = subparsers.add_parser("stats", help="show index statistics")

    export_parser = subparsers.add_parser("export", help="export indexed documents as JSON")
    export_parser.add_argument("--file", default=None, help="output file (defaults to stdout)")
    export_parser.add_argument("--limit", type=int, default=1000)

    return parser


def run_command(args: argparse.Namespace, state: Optional[AppState] = None) -> Dict[str, Any]:
    """Execute one CLI action and return the JSON-serialisable result payload."""
    if args.action not in VALID_ACTIONS:
        raise ValueError(f"unknown action {args.action!r}")

    owned_state = False
    if state is None:
        state = AppState(db_path=args.db)
        owned_state = True
    try:
        if args.action == "init-db":
            payload: Dict[str, Any] = {"db_path": state.db_path, "tables": state.store.index_tables()}
            if getattr(args, "schema", None):
                with open(args.schema, "r", encoding="utf-8") as handle:
                    payload["schema"] = state.set_schema(json.load(handle)).to_dict()
            else:
                payload["schema"] = state.schema.to_dict() if state.schema else None
            return payload

        if args.action == "set-schema":
            schema_payload = _load_json_argument(getattr(args, "json", None), getattr(args, "schema", None))
            return state.set_schema(schema_payload).to_dict()

        if args.action == "add":
            document = _load_json_argument(getattr(args, "json", None), getattr(args, "file", None))
            if state.schema is None:
                raise ValueError("configure a schema before adding documents")
            return state.store.add_document(document)

        if args.action == "ingest":
            if state.schema is None:
                raise ValueError("configure a schema before ingesting documents")
            if args.file:
                with open(args.file, "r", encoding="utf-8") as handle:
                    payload = handle.read()
            else:
                payload = sys.stdin.read()
            pipeline = IngestionPipeline(
                state.store,
                state.schema,
                batch_size=args.batch_size,
                max_quarantine_ratio=args.max_quarantine_ratio,
            )
            result = pipeline.ingest(payload, fmt=args.format, source=args.source)
            result.pop("documents", None)
            result.pop("quarantine", None)
            return result

        if args.action == "search":
            if state.schema is None:
                raise ValueError("configure a schema before searching")
            sort_spec = None
            if args.sort:
                sort_spec = [{"field": part.strip(), "direction": "desc"} for part in args.sort.split(",") if part.strip()]
            return execute_query(state.store, state.schema, {"term": args.query}, sort=sort_spec, limit=args.limit, offset=args.offset)

        if args.action == "query":
            if state.schema is None:
                raise ValueError("configure a schema before querying")
            filter_expression = _load_json_argument(args.filter, None) if args.filter else None
            sort_spec = None
            if args.sort:
                sort_spec = [{"field": part.strip(), "direction": "desc"} for part in args.sort.split(",") if part.strip()]
            return execute_query(state.store, state.schema, filter_expression, sort=sort_spec, limit=args.limit, offset=args.offset)

        if args.action == "aggregate":
            if state.schema is None:
                raise ValueError("configure a schema before aggregating")
            metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
            group_by = [g.strip() for g in args.group_by.split(",") if g.strip()] if args.group_by else []
            percentiles = [float(p.strip()) for p in args.percentiles.split(",") if p.strip()]
            filter_expression = json.loads(args.filter) if args.filter else None
            return aggregate_documents(
                state.store,
                state.schema,
                metrics=metrics,
                field=args.field,
                group_by=group_by,
                filter_expression=filter_expression,
                percentiles=percentiles,
                limit=args.limit,
                offset=args.offset,
            )

        if args.action == "stats":
            return state.store.stats()

        if args.action == "export":
            documents = state.store.list_documents(limit=args.limit, offset=0)
            payload = {"db_path": state.db_path, "count": len(documents), "documents": documents}
            if args.file:
                with open(args.file, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True, default=str)
                return {"written": args.file, "count": len(documents)}
            return payload
    finally:
        if owned_state:
            state.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_command(args)
    except (ValidationError, IngestionError, QueryError, AggregationError, IndexConflictError, DocumentNotFound, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
