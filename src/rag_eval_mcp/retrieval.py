"""The four retrieval methods the eval compares.

    bm25           keyword match only
    vector         embedding cosine only
    hybrid         BM25 and vector top-N lists merged with Reciprocal Rank Fusion
    hybrid_rerank  hybrid top-N, re-ordered by a cross-encoder
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .index import Index
from .rerank import Reranker

METHODS = ("bm25", "vector", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # meaning depends on method: BM25 score, cosine, RRF score or cross-encoder logit


def rank_rows(rows: np.ndarray, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k (row, score), best first. Stable sort: ties keep document order."""
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(rows[i]), float(scores[i])) for i in order]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], rrf_k: int = 60) -> list[tuple[int, float]]:
    """RRF (Cormack et al., 2009): score(d) = sum over lists of 1 / (rrf_k + rank), rank from 1.

    Uses only ranks, never raw scores, so BM25 scores (0..30ish) and cosine
    scores (-1..1) need no normalisation to be combined. rrf_k=60 is the
    paper's default; larger values flatten the gap between rank 1 and rank 10.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (rrf_k + rank)
    # Tie-break on item id so results are deterministic.
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(self, index: Index, reranker: Reranker | None = None, candidates: int = 30, rrf_k: int = 60):
        """candidates: how many results each first-stage list contributes to
        fusion, and how many hybrid results get reranked."""
        self.index = index
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, query: str, method: str = "hybrid", k: int = 5, doc_id: str | None = None) -> list[Hit]:
        if method == "bm25":
            ranked = self._bm25(query, k, doc_id)
        elif method == "vector":
            ranked = self._vector(query, k, doc_id)
        elif method == "hybrid":
            ranked = self._hybrid(query, doc_id)[:k]
        elif method == "hybrid_rerank":
            ranked = self._hybrid_rerank(query, k, doc_id)
        else:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        return [Hit(self.index.chunks[row], score) for row, score in ranked]

    def _bm25(self, query: str, k: int, doc_id: str | None) -> list[tuple[int, float]]:
        rows = self.index.rows(doc_id)
        return rank_rows(rows, self.index.bm25_scores(query, rows), k)

    def _vector(self, query: str, k: int, doc_id: str | None) -> list[tuple[int, float]]:
        rows = self.index.rows(doc_id)
        return rank_rows(rows, self.index.vector_scores(query, rows), k)

    def _hybrid(self, query: str, doc_id: str | None) -> list[tuple[int, float]]:
        n = self.candidates
        bm25_rows = [row for row, _ in self._bm25(query, n, doc_id)]
        vec_rows = [row for row, _ in self._vector(query, n, doc_id)]
        return reciprocal_rank_fusion([bm25_rows, vec_rows], rrf_k=self.rrf_k)

    def _hybrid_rerank(self, query: str, k: int, doc_id: str | None) -> list[tuple[int, float]]:
        if self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        cand_rows = [row for row, _ in self._hybrid(query, doc_id)[: self.candidates]]
        scores = self.reranker.score(query, [self.index.chunks[r].text for r in cand_rows])
        return rank_rows(np.array(cand_rows, dtype=np.int64), scores, k)
