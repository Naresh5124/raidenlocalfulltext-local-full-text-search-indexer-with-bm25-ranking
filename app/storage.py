"""SQLite storage & indexing engine for RaidenLocalFullText.

Persists primary records, a secondary inverted index (terms/postings),
unique-value indexes and metadata tables in a local SQLite database.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.schema import DocumentSchema

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
DEFAULT_DB_PATH = "raiden_local_fulltext.db"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v is not None)
    return str(value)


def tokenize(value: str) -> List[str]:
    """Deterministic tokenisation: lower-case alphanumerics, apostrophes kept."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(value or "")]


def _sql_type(field_type: str) -> str:
    return {
        "text": "TEXT",
        "integer": "INTEGER",
        "number": "REAL",
        "boolean": "INTEGER",
        "array": "TEXT",
        "object": "TEXT",
        "datetime": "TEXT",
    }[field_type]


def _encode(field, value: Any) -> Any:
    if value is None:
        return None
    if field.type == "boolean":
        return 1 if value else 0
    if field.type in ("array", "object"):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


class IndexConflictError(Exception):
    """Raised when a ``unique`` schema field collides with an existing record."""

    def __init__(self, field: str, value: Any):
        super().__init__(f"unique constraint violated for field {field!r}: {value!r} already exists")
        self.field = field
        self.value = value


class DocumentNotFound(Exception):
    pass


class SQLiteIndexStore:
    """Local SQLite store for documents, inverted index and metadata."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, schema: Optional[DocumentSchema] = None):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.schema: Optional[DocumentSchema] = None
        self._create_base_tables()
        if schema is not None:
            self.set_schema(schema)

    # ---------------------------------------------------------------- tables
    def _create_base_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        external_id TEXT NOT NULL UNIQUE,
                        payload     TEXT NOT NULL,
                        body_text   TEXT NOT NULL DEFAULT '',
                        created_at  TEXT NOT NULL,
                        updated_at  TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS terms (
                        term     TEXT PRIMARY KEY,
                        doc_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS postings (
                        term     TEXT NOT NULL REFERENCES terms(term) ON DELETE CASCADE,
                        doc_id   INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        term_freq INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (term, doc_id)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS doc_lengths (
                        doc_id  INTEGER PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
                        length  INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS unique_values (
                        field_name TEXT NOT NULL,
                        value      TEXT NOT NULL,
                        doc_id     INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                        PRIMARY KEY (field_name, value)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_postings_doc ON postings(doc_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_unique_values_field ON unique_values(field_name)"))

    def set_schema(self, schema: DocumentSchema) -> None:
        """Materialise the schema as columns on ``documents`` plus secondary indexes."""
        with self.engine.begin() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)"))}
            for field in schema.fields:
                column = f"f_{field.name}"
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE documents ADD COLUMN {column} {_sql_type(field.type)}"))
                if field.unique:
                    conn.execute(
                        text(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{column} ON documents({column})")
                    )
                elif field.type in ("text", "integer", "number", "datetime"):
                    conn.execute(
                        text(f"CREATE INDEX IF NOT EXISTS ix_{column} ON documents({column})")
                    )
            conn.execute(
                text(
                    "INSERT INTO metadata (key, value, updated_at) VALUES (:k, :v, :t) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at"
                ),
                {"k": "schema", "v": json.dumps(schema.to_dict(), sort_keys=True), "t": _now()},
            )
        self.schema = schema

    # ---------------------------------------------------------------- writes
    def add_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        if self.schema is None:
            raise RuntimeError("no schema configured: call set_schema before indexing documents")
        validated = self.schema.validate(document)
        pk = self.schema.primary_key_field.name
        external_id = str(validated[pk])
        now = _now()
        columns = {"external_id": external_id, "payload": json.dumps(validated, sort_keys=True), "created_at": now, "updated_at": now}
        for field in self.schema.fields:
            columns[f"f_{field.name}"] = _encode(field, validated.get(field.name))
        body_text = self._document_text(validated)
        columns["body_text"] = body_text

        col_names = ", ".join(columns)
        placeholders = ", ".join(f":{name}" for name in columns)
        with self.engine.begin() as conn:
            try:
                result = conn.execute(
                    text(f"INSERT INTO documents ({col_names}) VALUES ({placeholders})"),
                    columns,
                )
            except Exception as exc:  # sqlite3.IntegrityError
                if "UNIQUE" in str(exc).upper():
                    raise IndexConflictError(pk, external_id) from exc
                raise
            doc_id = result.lastrowid
            self._index_document(conn, doc_id, body_text, validated)
        return self.get_document(doc_id)

    def update_document(self, external_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
        if self.schema is None:
            raise RuntimeError("no schema configured: call set_schema before indexing documents")
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT doc_id FROM documents WHERE external_id = :e"), {"e": str(external_id)}).fetchone()
        if row is None:
            raise DocumentNotFound(f"document {external_id!r} not found")
        doc_id = row[0]
        merged = dict(self.get_document(doc_id)["payload"])
        merged.update(document)
        validated = self.schema.validate(merged)
        body_text = self._document_text(validated)
        columns = {"payload": json.dumps(validated, sort_keys=True), "updated_at": _now(), "body_text": body_text}
        for field in self.schema.fields:
            columns[f"f_{field.name}"] = _encode(field, validated.get(field.name))
        sets = ", ".join(f"{name} = :{name}" for name in columns)
        with self.engine.begin() as conn:
            try:
                conn.execute(text(f"UPDATE documents SET {sets} WHERE doc_id = :doc_id"), {**columns, "doc_id": doc_id})
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise IndexConflictError(self.schema.primary_key_field.name, external_id) from exc
                raise
            conn.execute(text("DELETE FROM postings WHERE doc_id = :d"), {"d": doc_id})
            conn.execute(text("DELETE FROM unique_values WHERE doc_id = :d"), {"d": doc_id})
            self._index_document(conn, doc_id, body_text, validated)
        return self.get_document(doc_id)

    def delete_document(self, external_id: str) -> bool:
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT doc_id FROM documents WHERE external_id = :e"), {"e": str(external_id)}).fetchone()
            if row is None:
                return False
            conn.execute(text("DELETE FROM documents WHERE doc_id = :d"), {"d": row[0]})
            self._refresh_term_counts(conn)
            return True

    def _index_document(self, conn, doc_id: int, body_text: str, validated: Dict[str, Any]) -> None:
        tokens = tokenize(body_text)
        if tokens:
            conn.execute(
                text("INSERT INTO doc_lengths (doc_id, length) VALUES (:d, :l) ON CONFLICT(doc_id) DO UPDATE SET length = excluded.length"),
                {"d": doc_id, "l": len(tokens)},
            )
        freqs: Dict[str, int] = {}
        for token in tokens:
            freqs[token] = freqs.get(token, 0) + 1
        for term, freq in freqs.items():
            conn.execute(
                text("INSERT INTO terms (term, doc_count) VALUES (:t, 1) ON CONFLICT(term) DO UPDATE SET doc_count = terms.doc_count + 1"),
                {"t": term},
            )
            conn.execute(
                text("INSERT INTO postings (term, doc_id, term_freq) VALUES (:t, :d, :f) ON CONFLICT(term, doc_id) DO UPDATE SET term_freq = excluded.term_freq"),
                {"t": term, "d": doc_id, "f": freq},
            )
        for field in self.schema.unique_field_names():
            value = validated.get(field)
            if value is not None:
                conn.execute(
                    text("INSERT INTO unique_values (field_name, value, doc_id) VALUES (:f, :v, :d) ON CONFLICT(field_name, value) DO UPDATE SET doc_id = excluded.doc_id"),
                    {"f": field, "v": str(value), "d": doc_id},
                )

    def _refresh_term_counts(self, conn) -> None:
        conn.execute(text("DELETE FROM terms WHERE doc_count <= 0"))
        conn.execute(
            text(
                "UPDATE terms SET doc_count = (SELECT COUNT(*) FROM postings WHERE postings.term = terms.term)"
            )
        )
        conn.execute(text("DELETE FROM terms WHERE doc_count = 0"))

    def _document_text(self, validated: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field in self.schema.fields:
            if field.type in ("text", "array", "object", "number", "integer", "datetime", "boolean"):
                parts.append(normalize_text(validated.get(field.name)))
        return " ".join(parts)

    # ---------------------------------------------------------------- reads
    def get_document(self, doc_id_or_external: Any) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT doc_id, external_id, payload, body_text, created_at, updated_at FROM documents WHERE doc_id = :d OR external_id = :e"),
                {"d": doc_id_or_external, "e": str(doc_id_or_external)},
            ).fetchone()
        if row is None:
            raise DocumentNotFound(f"document {doc_id_or_external!r} not found")
        return {
            "doc_id": row[0],
            "external_id": row[1],
            "payload": json.loads(row[2]),
            "body_text": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def list_documents(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT doc_id, external_id, payload, body_text, created_at, updated_at "
                    "FROM documents ORDER BY doc_id LIMIT :limit OFFSET :offset"
                ),
                {"limit": limit, "offset": offset},
            ).fetchall()
        return [
            {
                "doc_id": r[0],
                "external_id": r[1],
                "payload": json.loads(r[2]),
                "body_text": r[3],
                "created_at": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    def find_by_unique(self, field: str, value: Any) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT doc_id FROM unique_values WHERE field_name = :f AND value = :v"),
                {"f": field, "v": str(value)},
            ).fetchone()
        if row is None:
            return None
        return self.get_document(row[0])

    def search_terms(self, term: str, limit: int = 100) -> List[Dict[str, Any]]:
        term = term.lower().strip()
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT p.doc_id, p.term_freq, d.external_id, d.body_text "
                    "FROM postings p JOIN documents d ON d.doc_id = p.doc_id "
                    "WHERE p.term = :t ORDER BY p.term_freq DESC, p.doc_id ASC LIMIT :limit"
                ),
                {"t": term, "limit": limit},
            ).fetchall()
        return [
            {"doc_id": r[0], "term_freq": r[1], "external_id": r[2], "snippet": r[3][:200]}
            for r in rows
        ]

    # ---------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            term_count = conn.execute(text("SELECT COUNT(*) FROM terms")).scalar()
            posting_count = conn.execute(text("SELECT COUNT(*) FROM postings")).scalar()
            total_length = conn.execute(text("SELECT COALESCE(SUM(length), 0) FROM doc_lengths")).scalar()
            metadata = {r[0]: r[1] for r in conn.execute(text("SELECT key, value FROM metadata ORDER BY key"))}
        avg = (total_length / doc_count) if doc_count else 0.0
        return {
            "documents": doc_count,
            "terms": term_count,
            "postings": posting_count,
            "total_tokens": total_length,
            "avg_doc_length": round(avg, 4),
            "db_path": self.db_path,
            "schema": metadata.get("schema"),
            "metadata": metadata,
        }

    def index_tables(self) -> List[str]:
        with self.engine.connect() as conn:
            names = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY name"))]
        return names

    def bm25_stats(self) -> Dict[str, float]:
        with self.engine.connect() as conn:
            doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            avg = conn.execute(text("SELECT AVG(length) FROM doc_lengths")).scalar() or 0.0
        return {"document_count": float(doc_count), "avg_doc_length": float(avg)}

    def close(self) -> None:
        self.engine.dispose()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
