"""SQLite-хранилище: документы, фрагменты с эмбеддингами, найденные противоречия."""
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,            -- pdf | sql | text
    source_ref  TEXT NOT NULL,            -- имя файла или id записи в SQL
    content_hash TEXT NOT NULL,
    doc_number  TEXT,
    doc_date    TEXT,                     -- ISO yyyy-mm-dd
    pages       INTEGER DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued | waiting | processing | ready | error
    error       TEXT,
    created_at  REAL NOT NULL,
    UNIQUE (source, source_ref)
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    page       INTEGER,
    text       TEXT NOT NULL,
    normative  INTEGER NOT NULL DEFAULT 0,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS contradictions (
    id          INTEGER PRIMARY KEY,
    chunk_a     INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    chunk_b     INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    topic       TEXT,
    explanation TEXT,
    quote_a     TEXT,
    quote_b     TEXT,
    similarity  REAL,
    status      TEXT NOT NULL DEFAULT 'new',  -- new | confirmed | dismissed
    created_at  REAL NOT NULL,
    UNIQUE (chunk_a, chunk_b)
);
CREATE TABLE IF NOT EXISTS checked_pairs (
    chunk_a INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    chunk_b INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_a, chunk_b)
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            # Документы, обработка которых прервалась при перезапуске, вернём в очередь.
            self._conn.execute(
                "UPDATE documents SET status='queued' WHERE status='processing'"
            )
            self._conn.commit()
        # Увеличивается при любом изменении фрагментов: по нему индексы понимают, что пора перестроиться.
        self.version = 0

    # --- служебное ---
    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _x(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.lastrowid

    # --- документы ---
    def find_document(self, source: str, source_ref: str):
        rows = self._q(
            "SELECT * FROM documents WHERE source=? AND source_ref=?", (source, source_ref)
        )
        return dict(rows[0]) if rows else None

    def find_by_hash(self, content_hash: str):
        rows = self._q("SELECT * FROM documents WHERE content_hash=?", (content_hash,))
        return dict(rows[0]) if rows else None

    def add_document(self, title: str, source: str, source_ref: str, content_hash: str) -> int:
        return self._x(
            "INSERT INTO documents (title, source, source_ref, content_hash, created_at)"
            " VALUES (?,?,?,?,?)",
            (title, source, source_ref, content_hash, time.time()),
        )

    def update_document(self, doc_id: int, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE documents SET {cols} WHERE id=?", (*fields.values(), doc_id))

    def get_document(self, doc_id: int):
        rows = self._q("SELECT * FROM documents WHERE id=?", (doc_id,))
        return dict(rows[0]) if rows else None

    def list_documents(self) -> list[dict]:
        rows = self._q(
            "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) AS chunks"
            " FROM documents d ORDER BY d.created_at DESC"
        )
        return [dict(r) for r in rows]

    def documents_with_status(self, status: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM documents WHERE status=?", (status,))]

    def delete_document(self, doc_id: int):
        self._x("DELETE FROM documents WHERE id=?", (doc_id,))
        self.version += 1

    # --- фрагменты ---
    def replace_chunks(self, doc_id: int, chunks: list[dict], embeddings: np.ndarray):
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.executemany(
                "INSERT INTO chunks (doc_id, ord, page, text, normative, embedding)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        doc_id,
                        i,
                        c.get("page"),
                        c["text"],
                        int(c.get("normative", False)),
                        embeddings[i].astype(np.float32).tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
            self._conn.commit()
            self.version += 1

    def all_chunks(self) -> list[dict]:
        """Все фрагменты готовых документов вместе с эмбеддингами."""
        rows = self._q(
            "SELECT c.id, c.doc_id, c.page, c.text, c.normative, c.embedding"
            " FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.status='ready'"
            " ORDER BY c.id"
        )
        return [dict(r) for r in rows]

    def chunks_by_ids(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._q(
            "SELECT c.id, c.doc_id, c.page, c.text, d.title, d.doc_number, d.doc_date"
            f" FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id IN ({marks})",
            ids,
        )
        return {r["id"]: dict(r) for r in rows}

    # --- противоречия ---
    def is_checked(self, a: int, b: int) -> bool:
        a, b = min(a, b), max(a, b)
        return bool(self._q("SELECT 1 FROM checked_pairs WHERE chunk_a=? AND chunk_b=?", (a, b)))

    def mark_checked(self, a: int, b: int):
        a, b = min(a, b), max(a, b)
        self._x("INSERT OR IGNORE INTO checked_pairs (chunk_a, chunk_b) VALUES (?,?)", (a, b))

    def add_contradiction(self, a: int, b: int, similarity: float, verdict: dict):
        a_, b_ = min(a, b), max(a, b)
        quote_a, quote_b = verdict.get("quote_a"), verdict.get("quote_b")
        if a_ != a:  # цитаты должны соответствовать фрагментам после упорядочивания
            quote_a, quote_b = quote_b, quote_a
        self._x(
            "INSERT OR REPLACE INTO contradictions"
            " (chunk_a, chunk_b, topic, explanation, quote_a, quote_b, similarity, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (a_, b_, verdict.get("topic"), verdict.get("explanation"), quote_a, quote_b,
             similarity, time.time()),
        )

    def list_contradictions(self, status: str | None = None, chunk_ids: list[int] | None = None):
        sql = "SELECT * FROM contradictions"
        args: list = []
        where = []
        if status:
            where.append("status=?")
            args.append(status)
        if chunk_ids:
            marks = ",".join("?" * len(chunk_ids))
            where.append(f"(chunk_a IN ({marks}) OR chunk_b IN ({marks}))")
            args += list(chunk_ids) * 2
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC"
        rows = [dict(r) for r in self._q(sql, args)]
        chunks = self.chunks_by_ids(sorted({r[k] for r in rows for k in ("chunk_a", "chunk_b")}))
        for r in rows:
            r["a"] = chunks.get(r["chunk_a"])
            r["b"] = chunks.get(r["chunk_b"])
        return rows

    def set_contradiction_status(self, cid: int, status: str):
        self._x("UPDATE contradictions SET status=? WHERE id=?", (status, cid))
