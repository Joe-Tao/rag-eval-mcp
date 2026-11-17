"""Retrieval metrics. Each takes one query's ranked chunk ids and the set of relevant ids.

hit@k        1 if ANY relevant chunk is in the top k, else 0.
             "Would the LLM see the answer at all if we gave it k chunks?"
recall@k     share of the relevant chunks that are in the top k.
             Matters when the answer is spread over several chunks (e.g. "Parties").
reciprocal   1 / (position of the first relevant chunk), 0 if not in the top k.
rank         Averaged over queries this is MRR: rewards putting the answer FIRST.
"""

from __future__ import annotations

from math import comb
from typing import Sequence


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for position, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1.0 / position
    return 0.0


def random_hit_at_k(n_chunks: int, n_relevant: int, k: int) -> float:
    """Expected hit@k of picking k chunks at random: the floor every method must beat."""
    if n_chunks <= k:
        return 1.0 if n_relevant else 0.0
    return 1.0 - comb(n_chunks - n_relevant, k) / comb(n_chunks, k)
