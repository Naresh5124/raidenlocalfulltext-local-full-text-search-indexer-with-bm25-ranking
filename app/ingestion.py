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


def _normalize_payload(payload: Any) -> str:
    """Return a plain text payload, unwrapping JSON-encoded strings."""
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return payload
            if isinstance(decoded, str):
                return decoded
        return payload
    return str(payload)


def _looks_like_csv(text_payload: str) -> bool:
    """True when the payload reads as a CSV table (header + data rows)."""
    lines = [line for line in text_payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header = lines[0]
    if "{" in header or "[" in header:
        return False
    if "," not in header:
        return False
    try:
        header_columns = len(next(csv.reader([header])))
    except Exception:  # pragma: no cover - defensive
        return False
    if header_columns < 2:
        return False
    for line in lines[1:]:
        if "{" in line or "[" in line:
            return False
        if "," not in line:
            return False
    return True


def _is_jsonl(text_payload: str) -> bool:
    """True when every non-empty line parses as a JSON object.

    A payload made of a single JSON object on a single line is plain JSON, so
    at least two object lines are required for JSONL detection.
    """
    lines = [line.strip() for line in text_payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        if not line.startswith("{"):
            return False
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
    return True


def _detect_format(payload: Any, fmt: Optional[str]) -> str:
    if fmt is not None:
        fmt = fmt.lower()
        if fmt not in SUPPORTED_FORMATS:
            raise IngestionError(f"unsupported format {fmt!r}; expected one of {sorted(SUPPORTED_FORMATS)}")
        return fmt

    if isinstance(payload, (list, dict)):
        return "json"

    text = _normalize_payload(payload)
    stripped = text.strip()
    if not stripped:
        return "json"

    # Line-delimited JSON is detected first: each non-empty line is its own
    # JSON object. Only payloads that parse as one JSON value count as JSON.
    if _is_jsonl(stripped):
        return "jsonl"
    try:
        json.loads(stripped)
        return "json"
    except json.JSONDecodeError:
        pass
    if _looks_like_csv(stripped):
        return "csv"
    return "jsonl"


def _coerce_csv_value(value: Any) -> Any:
    """Best-effort conversion of a raw CSV cell into a JSON-ish python value."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return text


def _csv_rows(text_payload: str) -> Iterator[Tuple[int, Any, List[Dict[str, Any]]]]:
    """Yield ``(line_number, row, parse_errors)`` tuples for a CSV body."""
    body = _normalize_payload(text_payload)
    if not body:
        return
    if not body.endswith("\n"):
        body += "\n"
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames is None:
        raise IngestionError("CSV payload must contain a header row")
    header = [name for name in reader.fieldnames if name is not None]
    line_number = 1
    for raw_row in reader:
        line_number += 1
        values = list(raw_row.values())
        if len(values) != len(header):
            yield (
                line_number,
                dict(zip(header, values)),
                [
                    {
                        "field": None,
                        "code": "csv_column_count_mismatch",
                        "message": "row has more columns than the header",
                    }
                ],
            )
            continue
        yield (
            line_number,
            {name: _coerce_csv_value(value) for name, value in zip(header, values)},
            [],
        )


def _read_rows(payload: Any, fmt: str) -> Iterator[Tuple[int, Any, List[Dict[str, Any]]]]:
    """Yield ``(line_number, row, parse_errors)`` tuples for the payload."""
    if fmt == "json":
        if isinstance(payload, (list, dict)):
            records: List[Any] = payload if isinstance(payload, list) else [payload]
        else:
            text = _normalize_payload(payload).strip()
            try:
                parsed = json.loads(text) if text else []
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
        for line_number, line in enumerate(_normalize_payload(payload).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, line, [
                    {"field": None, "code": "invalid_json", "message": f"invalid JSON line: {exc}"}
                ]
                continue
            if isinstance(record, dict):
                yield line_number, record, []
            else:
                yield line_number, record, [
                    {"field": None, "code": "invalid_document", "message": "record must be a JSON object"}
                ]
        return

    if fmt == "csv":
        for item in _csv_rows(_normalize_payload(payload)):
            yield item
        return

    raise IngestionError(f"unsupported format {fmt!r}")


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
        rows = list(_read_rows(payload, detected))
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        batch: List[Tuple[int, Any]] = []
        received = 0
        batch_rejected = 0
        backpressure_triggered = False
        stopped_at: Optional[int] = None

        for line_number, row, parse_errors in rows:
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

        joined = "\n".join(str(line) for line in lines)
        if fmt == "csv":
            row_iter: Iterable[Tuple[int, Any, List[Dict[str, Any]]]] = list(_csv_rows(joined))
        else:
            row_iter = list(_read_rows(joined, "jsonl"))

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
