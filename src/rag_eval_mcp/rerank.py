"""Second-stage reranking with a cross-encoder.

A bi-encoder (the embedder) encodes query and chunk separately, so it is fast
enough to score every chunk. A cross-encoder reads (query, chunk) together,
which is more accurate but costs one model call per pair, so we only run it on
the top candidates from the first stage.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Higher = more relevant. One score per text."""
        ...


class CrossEncoderReranker:
    """Loaded once and reused. (The old rag_pipeline.py built a new CrossEncoder
    inside every query, reloading the weights each time.)"""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # heavy import, keep lazy

        self.name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, t) for t in texts]
        return np.asarray(self._model.predict(pairs, batch_size=self.batch_size), dtype=np.float32)
