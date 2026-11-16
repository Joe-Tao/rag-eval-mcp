"""One index over all chunks: a BM25 index plus a matrix of chunk embeddings.

Searches can be limited to one document (one contract). CUAD questions are
per-contract ("find the governing-law clause in THIS contract"), and the MCP
tool will expose the same filter.

BM25 statistics (idf, average length) are computed over the whole corpus, not
per contract. That gives more stable idf values than a 60-chunk contract would.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .bm25 import BM25, tokenize
from .chunking import Chunk
from .embedding import Embedder


class Index:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder, vectors: np.ndarray | None = None):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.bm25 = BM25([tokenize(c.text) for c in self.chunks])
        if vectors is None:
            vectors = embedder.encode([c.text for c in self.chunks])
        if len(vectors) != len(self.chunks):
            raise ValueError("vectors and chunks have different lengths")
        self.vectors = vectors

        rows: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            rows.setdefault(c.doc_id, []).append(i)
        self._rows_by_doc = {d: np.array(r, dtype=np.int64) for d, r in rows.items()}
        self._all_rows = np.arange(len(self.chunks), dtype=np.int64)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._rows_by_doc)

    def rows(self, doc_id: str | None = None) -> np.ndarray:
        """Row numbers to search: every chunk, or only the chunks of one document."""
        if doc_id is None:
            return self._all_rows
        if doc_id not in self._rows_by_doc:
            raise KeyError(f"unknown doc_id: {doc_id}")
        return self._rows_by_doc[doc_id]

    def bm25_scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        return self.bm25.scores(tokenize(query))[rows]

    def vector_scores(self, query: str, rows: np.ndarray) -> np.ndarray:
        q = self.embedder.encode([query])[0]
        return self.vectors[rows] @ q  # cosine, since all vectors are unit length

    # --- embedding cache -------------------------------------------------
    # Embedding ~6k CUAD chunks on a laptop CPU takes minutes. The cache key is
    # the model name plus a hash of every chunk, so changing the chunk size or
    # the model can never silently reuse stale vectors.

    def cache_key(self) -> str:
        return cache_key(self.chunks, self.embedder.name)

    @classmethod
    def build_cached(cls, chunks: Sequence[Chunk], embedder: Embedder, cache_dir: str | Path) -> "Index":
        path = Path(cache_dir) / f"{cache_key(chunks, embedder.name)}.npy"
        if path.exists():
            return cls(chunks, embedder, vectors=np.load(path))
        index = cls(chunks, embedder)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, index.vectors)
        return index


def cache_key(chunks: Sequence[Chunk], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for c in chunks:
        h.update(json.dumps([c.chunk_id, c.start, c.end, c.text]).encode())
    return h.hexdigest()[:16]
