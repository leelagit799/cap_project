"""FAISS vector store for indexed discharge cases — doc Table 14.

Embeddings come from ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim). The
index uses inner product over L2-normalised vectors, which is cosine
similarity, and a SQLite sidecar holds chunk text and metadata so answers can
cite their source document and be filtered per patient.

The embedding model is loaded lazily: importing this module must not pull
PyTorch into a process that only needs the metadata.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hospital_ai.core.config import get_settings
from hospital_ai.core.logging import get_logger

_log = get_logger(__name__, component="faiss-store")

EMBEDDING_DIM = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    row_id      INTEGER PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    case_id     TEXT,
    patient_id  TEXT,
    doc_type    TEXT,
    section     TEXT,
    source_uri  TEXT,
    text        TEXT NOT NULL,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_patient ON chunks(patient_id);
CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id);
"""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    patient_id: str | None = None
    case_id: str | None = None
    doc_type: str | None = None
    section: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_embedding_model():
    """Load all-MiniLM-L6-v2 once per process."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                name = get_settings().llm.embedding_model
                _log.info("loading embedding model", extra={"model": name})
                _MODEL = SentenceTransformer(name)
    return _MODEL


def embed(texts: list[str]):
    import numpy as np

    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    vectors = get_embedding_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.astype("float32")


class FaissStore:
    """Persistent FAISS index plus its SQLite metadata sidecar."""

    def __init__(self, directory: Path | None = None) -> None:
        settings = get_settings()
        self.directory = directory or settings.vector_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.faiss"
        self.db_path = self.directory / "chunks.sqlite"
        self._index = None
        self._lock = threading.Lock()
        self._init_db()

    # --- persistence ---------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    @property
    def index(self):
        import faiss

        if self._index is None:
            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
            else:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBEDDING_DIM))
        return self._index

    def _persist(self) -> None:
        import faiss

        faiss.write_index(self.index, str(self.index_path))

    # --- writes --------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> int:
        """Insert chunks, replacing any that were indexed before."""
        if not chunks:
            return 0

        import numpy as np

        with self._lock, sqlite3.connect(self.db_path) as conn:
            fresh: list[Chunk] = []
            row_ids: list[int] = []

            for chunk in chunks:
                existing = conn.execute(
                    "SELECT row_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
                ).fetchone()
                if existing:
                    # Re-indexing a corrected case must replace, not duplicate.
                    self.index.remove_ids(np.array([existing[0]], dtype="int64"))
                    conn.execute("DELETE FROM chunks WHERE row_id = ?", (existing[0],))

                cursor = conn.execute(
                    "INSERT INTO chunks (chunk_id, case_id, patient_id, doc_type, section,"
                    " source_uri, text, metadata) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        chunk.chunk_id, chunk.case_id, chunk.patient_id, chunk.doc_type,
                        chunk.section, chunk.source_uri, chunk.text,
                        json.dumps(chunk.metadata or {}, ensure_ascii=False),
                    ),
                )
                row_ids.append(int(cursor.lastrowid))
                fresh.append(chunk)

            vectors = embed([c.text for c in fresh])
            self.index.add_with_ids(vectors, np.array(row_ids, dtype="int64"))
            conn.commit()

        self._persist()
        _log.info("chunks indexed", extra={"count": len(chunks), "total": self.size})
        return len(chunks)

    def clear(self) -> None:
        import faiss

        with self._lock:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBEDDING_DIM))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM chunks")
                conn.commit()
            self._persist()

    # --- reads ---------------------------------------------------------------

    @property
    def size(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def indexed_patients(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT patient_id FROM chunks WHERE patient_id IS NOT NULL"
                " ORDER BY patient_id"
            ).fetchall()
        return [row[0] for row in rows]

    def _rows(self, row_ids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(row_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT row_id, chunk_id, case_id, patient_id, doc_type, section,"
                f" source_uri, text, metadata FROM chunks WHERE row_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {
            row[0]: Chunk(
                chunk_id=row[1], case_id=row[2], patient_id=row[3], doc_type=row[4],
                section=row[5], source_uri=row[6], text=row[7],
                metadata=json.loads(row[8] or "{}"),
            )
            for row in rows
        }

    def search(
        self, query: str, top_k: int = 5, patient_id: str | None = None
    ) -> list[SearchHit]:
        if self.size == 0:
            return []

        import numpy as np

        # Over-fetch when filtering by patient, since the filter is applied
        # after the vector search rather than inside the index.
        fetch = top_k * 8 if patient_id else top_k
        vector = embed([query])
        scores, ids = self.index.search(vector, min(fetch, max(self.size, 1)))

        row_ids = [int(i) for i in ids[0] if i != -1]
        chunks = self._rows(row_ids)

        hits: list[SearchHit] = []
        for score, row_id in zip(scores[0], ids[0]):
            chunk = chunks.get(int(row_id))
            if chunk is None:
                continue
            if patient_id and chunk.patient_id != patient_id:
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score)))
            if len(hits) >= top_k:
                break
        return hits


_STORE: FaissStore | None = None


def get_store() -> FaissStore:
    global _STORE
    if _STORE is None:
        _STORE = FaissStore()
    return _STORE


def reset_store() -> None:
    global _STORE
    _STORE = None
