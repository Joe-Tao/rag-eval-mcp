"""Run every eval question through every retrieval method and score the results.

The loop, in words:
    for each question:
        relevant = chunks of that contract that overlap a gold answer span
        for each method:
            ranked = top-10 chunk ids the method returns (searching that contract only)
            score ranked against relevant -> hit@k, recall@k, reciprocal rank, latency
Then average per method (the results table) and per (category, method).
"""

from __future__ import annotations

import random
import statistics
import time
from collections import defaultdict
from typing import Callable, Sequence

from .chunking import Chunk
from .cuad import EvalItem, relevant_chunk_ids
from .metrics import hit_at_k, random_hit_at_k, recall_at_k, reciprocal_rank
from .retrieval import Retriever

K_VALUES = (1, 3, 5, 10)
TOP_K = max(K_VALUES)


def sample_items(items: list[EvalItem], limit: int | None, seed: int = 0) -> list[EvalItem]:
    """A fixed random subset, so a small run is comparable between runs and machines."""
    if limit is None or limit >= len(items):
        return list(items)
    return random.Random(seed).sample(items, limit)


def evaluate(
    retriever: Retriever,
    items: Sequence[EvalItem],
    chunks_by_doc: dict[str, list[Chunk]],
    methods: Sequence[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """One row per (question, method)."""
    # Warm-up: the first call to a model is slow (lazy init). Don't count it.
    for method in methods:
        retriever.search("warm up", method, k=1, doc_id=items[0].doc_id)

    rows = []
    for n, item in enumerate(items, start=1):
        relevant = relevant_chunk_ids(chunks_by_doc[item.doc_id], item.spans)
        for method in methods:
            t0 = time.perf_counter()
            hits = retriever.search(item.query, method, k=TOP_K, doc_id=item.doc_id)
            latency_ms = (time.perf_counter() - t0) * 1000
            ranked = [h.chunk.chunk_id for h in hits]

            row = {"qid": item.qid, "doc_id": item.doc_id, "category": item.category,
                   "method": method, "n_relevant": len(relevant), "latency_ms": latency_ms,
                   "ranked": ranked}
            for k in K_VALUES:
                row[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
                row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            row["mrr@10"] = reciprocal_rank(ranked, relevant, TOP_K)
            rows.append(row)
        if on_progress:
            on_progress(n, len(items))
    return rows


METRICS = ("hit@1", "hit@5", "recall@5", "recall@10", "mrr@10")


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """method -> {metric: mean over questions, plus median latency}."""
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = {}
    for method, rs in by_method.items():
        out[method] = {m: statistics.mean(r[m] for r in rs) for m in METRICS}
        out[method]["latency_ms_p50"] = statistics.median(r["latency_ms"] for r in rs)
        out[method]["n"] = len(rs)
    return out


def by_category(rows: list[dict], metric: str = "hit@5") -> dict[str, dict[str, float]]:
    """category -> {method: mean metric}, plus 'n' = questions in that category."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["category"], r["method"])].append(r[metric])
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (cat, method), vals in groups.items():
        out[cat][method] = statistics.mean(vals)
        out[cat]["n"] = len(vals)
    return dict(out)


def random_baseline(items: Sequence[EvalItem], chunks_by_doc: dict[str, list[Chunk]], k: int = 5) -> float:
    return statistics.mean(
        random_hit_at_k(len(chunks_by_doc[it.doc_id]),
                        len(relevant_chunk_ids(chunks_by_doc[it.doc_id], it.spans)), k)
        for it in items
    )


def summary_markdown(summary: dict[str, dict[str, float]], baseline_hit5: float) -> str:
    header = "| method | " + " | ".join(METRICS) + " | p50 latency (ms) |"
    lines = [header, "|" + "---|" * (len(METRICS) + 2)]
    for method, s in summary.items():
        cells = " | ".join(f"{s[m]:.3f}" for m in METRICS)
        lines.append(f"| {method} | {cells} | {s['latency_ms_p50']:.0f} |")
    lines.append(f"| random (expected) | | {baseline_hit5:.3f} | | | | |")
    return "\n".join(lines)


def category_markdown(cats: dict[str, dict[str, float]], methods: Sequence[str], metric: str = "hit@5") -> str:
    lines = [f"| category ({metric}) | n | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 2)]
    for cat in sorted(cats, key=lambda c: -cats[c]["n"]):
        cells = " | ".join(f"{cats[cat].get(m, float('nan')):.2f}" for m in methods)
        lines.append(f"| {cat} | {cats[cat]['n']} | {cells} |")
    return "\n".join(lines)
