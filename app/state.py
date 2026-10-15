"""Shared application state: SQLite store + active schema."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from app.schema import DocumentSchema, parse_schema
from app.storage import DEFAULT_DB_PATH, SQLiteIndexStore


class AppState:
    """Owns the SQLite index store and the currently active document schema."""

    def __init__(self, db_path: Optional[str] = None, schema: Optional[DocumentSchema] = None):
        self.db_path = db_path or os.environ.get("RAIDEN_DB_PATH", DEFAULT_DB_PATH)
        self.store = SQLiteIndexStore(db_path=self.db_path, schema=schema)
        self.schema: Optional[DocumentSchema] = schema

    def set_schema(self, payload: Dict[str, Any]) -> DocumentSchema:
        schema = parse_schema(payload)
        self.store.set_schema(schema)
        self.schema = schema
        return schema

    def close(self) -> None:
        self.store.close()
