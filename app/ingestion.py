"""Batch & Streaming ingestion pipeline for RaidenLocalFullText.

Accepts JSON, JSONL or CSV payloads, validates every row against the active
schema, indexes valid rows in bounded batches (backpressure) and quarantines
malformed rows instead of failing the whole run.

Backpressure model
------------------
Rows are consumed in chunks of ``batch_size``. A new chunk is only pulled from
the input stream once the previous chunk has been written to SQLite, so memory
use stays bounded by ``batch_size`` regardless of payload size. When the
quarantine ratio of a completed batch exceeds ``max_quarantine_ratio`` the
pipeline signals backpressure and stops pulling further batches, returning a
``backpressure`` flag so callers can throttle, repair and retry.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from app.schema import DocumentSchema, FieldValidationError
from app.storage import IndexConflictError, SQLiteIndexStore

SUPPORTED_FORMATS = {"json", "jsonl", "csv"}


class IngestionError(Exception):
    """Raised for payload-level ingestion problems (bad format, no header)."""


def _detect_format(payload: Any, fmt: Optional[str]) -> str:
    if fmt is not None:
        fmt = fmt.lower()
        if fmt not in SUPPORTED_FORMATS:
            raise IngestionError(f"unsupported format {fmt!r}; expected one of {sorted(SUPPORTED_FORMATS)}")
        return fmt

    if isinstance(payload, (list, dict)):
        return "json"
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise IngestionError("payload must be a string, list or object")

    stripped = payload.strip()
    if not stripped:
        return "json"
    first = stripped[0]
    if first == "[":
        return "json"
    if first == "{":
        try:
            json.loads(stripped)
            return "json"
        except json.JSONDecodeError:
            return "jsonl"
    if "\n" in stripped:
        return "csv"
    if "," in stripped:
        return "csv"
    return "jsonl"


def _csv_rows(text_payload: str) -> Iterator[Tuple[int, Any, List[Dict[str, Any]]]]:
    reader = csv.DictReader(io.StringIO(text_payload))
    if reader.fieldnames is None:
        raise IngestionError("CSV payload must contain a header row")
    header = [name for name in reader.fieldnames if name is not None]
    for line_number, raw_row in enumerate(reader, start=2):
        values = list(raw_row.values())
        if len(values) != len(header):
            yield line_number, dict(zip(header, values)), [
                {"field": None, "code": "csv_column_count_mismatch", "message": "row has more columns than the header"}
            ]
            continue
        yield line_number, {name: value for name, value in zip(header, values)}, []


def _read_rows(payload: Any, fmt: str) -> Iterator[Tuple[int, Any, List[Dict[str, Any]]]]:
    """Yield ``(line_number, row, parse_errors)`` tuples for the payload."""
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")

    if fmt == "json":
        if isinstance(payload, (list, dict)):
            records = payload if isinstance(payload, list) else [payload]
        else:
            try:
                parsed = json.loads(payload or "[]")
            except json.JSONDecodeError as exc:
                raise IngestionError(f"invalid JSON payload: {exc}") from exc
            records = parsed if isinstance(parsed, list) else [parsed]
        for index, record in enumerate(records, start=1):
            if isinstance(record, dict):
                yield index, record, []
            else:
                yield index, record, [
                    {"field": None, "code": "invalid_document", "message": "record must be a JSON object"}
                ]
        return

    if fmt == "jsonl":
        for line_number, line in enumerate((payload or "").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, line, [{"field": None, "code": "invalid_json", "message": f"invalid JSON line: {exc}"}]
                continue
            if isinstance(record, dict):
                yield line_number, record, []
            else:
                yield line_number, record, [
                    {"field": None, "code": "invalid_document", "message": "record must be a JSON object"}
                ]
        return

    return _csv_rows(payload or "")


class IngestionPipeline:
    """Schema-aware ingestion with batching, backpressure and quarantine."""

    def __init__(
        self,
        store: SQLiteIndexStore,
        schema: DocumentSchema,
        batch_size: int = 100,
        max_quarantine_ratio: float = 0.5,
    ):
        if batch_size < 1:
            raise IngestionError("batch_size must be >= 1")
        if not 0.0 <= max_quarantine_ratio <= 1.0:
            raise IngestionError("max_quarantine_ratio must be between 0.0 and 1.0")
        self.store = store
        self.schema = schema
        self.batch_size = batch_size
        self.max_quarantine_ratio = max_quarantine_ratio

    def ingest(self, payload: Any, fmt: Optional[str] = None, source: str = "inline") -> Dict[str, Any]:
        detected = _detect_format(payload, fmt)
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        batch: List[Tuple[int, Any]] = []
        received = 0
        batch_rejected = 0
        backpressure_triggered = False
        stopped_at: Optional[int] = None

        for line_number, row, parse_errors in _read_rows(payload, detected):
            if backpressure_triggered:
                stopped_at = line_number
                break
            received += 1
            if parse_errors:
                batch_rejected += 1
                rejected.append(self.store.add_quarantined_row(source, line_number, row, parse_errors, "parse_error"))
                continue
            try:
                validated = self.schema.validate(row)
            except FieldValidationError as exc:
                batch_rejected += 1
                rejected.append(self.store.add_quarantined_row(source, line_number, row, exc.errors, "validation_error"))
                continue
            batch.append((line_number, validated))

            if len(batch) >= self.batch_size:
                self._flush_batch(batch, source, accepted, rejected)
                batch = []
                if batch_rejected / self.batch_size > self.max_quarantine_ratio:
                    backpressure_triggered = True

        if batch and not backpressure_triggered:
            self._flush_batch(batch, source, accepted, rejected)
            if batch_rejected / len(batch) > self.max_quarantine_ratio:
                backpressure_triggered = True

        run = self.store.record_ingestion_run(source, detected, received, len(accepted), len(rejected))
        return {
            "run_id": run["run_id"],
            "source": source,
            "format": detected,
            "batch_size": self.batch_size,
            "received": received,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "backpressure": backpressure_triggered,
            "stopped_at_line": stopped_at,
            "documents": accepted,
            "quarantine": rejected,
        }

    def ingest_stream(self, lines: Iterable[str], fmt: str = "jsonl", source: str = "stream") -> Dict[str, Any]:
        """Streaming entry point: consume an iterable of lines in bounded batches."""
        fmt = fmt.lower()
        if fmt not in {"jsonl", "csv"}:
            raise IngestionError("streaming ingestion supports jsonl or csv line iterables")

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        batch: List[Tuple[int, Any]] = []
        received = 0
        batch_rejected = 0
        backpressure_triggered = False
        stopped_at: Optional[int] = None

        if fmt == "csv":
            row_iter: Iterable[Tuple[int, Any, List[Dict[str, Any]]]] = _csv_rows("\n".join(lines))
        else:
            row_iter = _read_rows("\n".join(lines), "jsonl")

        row_iter = list(row_iter)

        for line_number, row, parse_errors in row_iter:
            if backpressure_triggered:
                stopped_at = line_number
                break
            received += 1
            if parse_errors:
                batch_rejected += 1
                rejected.append(self.store.add_quarantined_row(source, line_number, row, parse_errors, "parse_error"))
                continue
            try:
                validated = self.schema.validate(row)
            except FieldValidationError as exc:
                batch_rejected += 1
                rejected.append(self.store.add_quarantined_row(source, line_number, row, exc.errors, "validation_error"))
                continue
            batch.append((line_number, validated))
            if len(batch) >= self.batch_size:
                self._flush_batch(batch, source, accepted, rejected)
                batch = []
                if batch_rejected / self.batch_size > self.max_quarantine_ratio:
                    backpressure_triggered = True

        if batch and not backpressure_triggered:
            self._flush_batch(batch, source, accepted, rejected)
            if batch_rejected / len(batch) > self.max_quarantine_ratio:
                backpressure_triggered = True

        run = self.store.record_ingestion_run(source, fmt, received, len(accepted), len(rejected))
        return {
            "run_id": run["run_id"],
            "source": source,
            "format": fmt,
            "batch_size": self.batch_size,
            "received": received,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "backpressure": backpressure_triggered,
            "stopped_at_line": stopped_at,
            "documents": accepted,
            "quarantine": rejected,
        }

    def _flush_batch(
        self,
        batch: List[Tuple[int, Any]],
        source: str,
        accepted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
    ) -> None:
        for line_number, validated in batch:
            try:
                created = self.store.add_document(validated)
            except IndexConflictError as exc:
                rejected.append(
                    self.store.add_quarantined_row(
                        source,
                        line_number,
                        validated,
                        [{"field": exc.field, "code": "unique_conflict", "message": str(exc)}],
                        "conflict",
                    )
                )
                continue
            accepted.append(
                {
                    "doc_id": created["doc_id"],
                    "external_id": created["external_id"],
                    "payload": created["payload"],
                }
            )
