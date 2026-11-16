"""Text -> unit-length vectors.

Every embedder returns L2-normalised float32 rows, so cosine similarity is a
plain dot product (one matrix-vector multiply instead of a Python loop).
"""

from __future__ import annotations

import zlib
from typing import Protocol, Sequence

import numpy as np

from .bm25 import tokenize

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array with unit-length rows."""
        ...


def _normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Local sentence-transformers model. No API key; downloads weights once.

    all-MiniLM-L6-v2 truncates input at 256 word pieces, roughly 1000 chars of
    English, which is one reason the default chunk size is 1000.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 500,
        )
        return vecs.astype(np.float32)


class HashingEmbedder:
    """Bag-of-words hashed into a fixed number of buckets.

    For tests and CI only: deterministic, instant, no download. It has no
    notion of meaning, so its eval numbers say nothing about dense retrieval.
    """

    def __init__(self, dim: int = 256):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                # crc32, not hash(): Python's str hash changes between runs.
                out[row, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return _normalise(out)
