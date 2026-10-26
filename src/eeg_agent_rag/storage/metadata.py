"""SQLite document state store. This MVP is single-tenant and single-process oriented."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from eeg_agent_rag.schemas.models import DocumentInfo


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    doi TEXT NOT NULL DEFAULT '',
                    year INTEGER NOT NULL DEFAULT 0,
                    license TEXT NOT NULL DEFAULT 'unknown',
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, document_id: str) -> DocumentInfo | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,)
            ).fetchone()
        return DocumentInfo(**dict(row)) if row else None

    def upsert(self, item: DocumentInfo) -> None:
        with self._connect() as con:
            con.execute("""
                INSERT INTO documents
                  (document_id, title, source, doi, year, license, status, chunk_count, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                  title=excluded.title, source=excluded.source,
                  doi=excluded.doi, year=excluded.year, license=excluded.license,
                  status=excluded.status, chunk_count=excluded.chunk_count,
                  error=excluded.error, updated_at=CURRENT_TIMESTAMP
            """, (
                item.document_id, item.title, item.source, item.doi, item.year,
                item.license, item.status, item.chunk_count, item.error,
            ))

    def list(self, limit: int = 100) -> list[DocumentInfo]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [DocumentInfo(**dict(row)) for row in rows]

    def delete(self, document_id: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
