"""Compare the retrieval methods on the CUAD eval set and write the results.

    python scripts/run_eval.py --limit 100          # quick: 100 questions, a few minutes
    python scripts/run_eval.py                      # full: all 1244 questions
    python scripts/run_eval.py --limit 50 --fake-models --out results/smoke   # no models (CI)

Writes to --out (default results/):
    summary.md       the results table + per-category table + run settings
    summary.json     same numbers, machine-readable
    per_item.jsonl   one line per (question, method) with the ranked chunk ids, for digging into failures
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

from rag_eval_mcp.cuad import read_jsonl
from rag_eval_mcp.evaluate import (
    by_category, category_markdown, evaluate, random_baseline, sample_items, summarize, summary_markdown)
from rag_eval_mcp.factory import CHUNK_OVERLAP, CHUNK_SIZE, load_corpus, make_models, make_retriever
from rag_eval_mcp.retrieval import METHODS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", default="data/eval/cuad_test_eval.jsonl")
    p.add_argument("--cuad", default="data/cuad_test.json")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--limit", type=int, default=None, help="random subset of N questions (fixed seed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--candidates", type=int, default=30, help="first-stage list size for fusion and rerank")
    p.add_argument("--fake-models", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    corpus = load_corpus(args.cuad)
    chunks_by_doc = corpus.chunks_by_doc()
    items = sample_items(read_jsonl(args.eval_set), args.limit, args.seed)

    embedder, reranker = make_models(args.fake_models)
    methods = [m for m in args.methods if m != "hybrid_rerank" or reranker is not None]
    t0 = time.perf_counter()
    retriever = make_retriever(corpus, embedder, reranker, candidates=args.candidates)
    print(f"index ready: {len(corpus.chunks)} chunks ({time.perf_counter() - t0:.0f}s)")
    print(f"evaluating {len(items)} questions x {len(methods)} methods: {', '.join(methods)}")

    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            elapsed = time.perf_counter() - t0
            print(f"  {done}/{total}  ({elapsed:.0f}s, ~{elapsed / done * (total - done):.0f}s left)", flush=True)

    rows = evaluate(retriever, items, chunks_by_doc, methods, on_progress=progress)

    summary = summarize(rows)
    cats = by_category(rows, "hit@5")
    baseline = random_baseline(items, chunks_by_doc, k=5)
    settings = {
        "date": date.today().isoformat(),
        "questions": len(items), "limit": args.limit, "seed": args.seed,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "candidates": args.candidates,
        "embedder": embedder.name, "reranker": getattr(reranker, "name", None),
        "machine": f"{platform.machine()} {platform.system()}, Python {sys.version.split()[0]}",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = "\n\n".join([
        "# Retrieval eval: CUAD test split",
        "Search is restricted to the question's own contract. Top 10 chunks per query.",
        summary_markdown(summary, baseline),
        "## hit@5 by clause category",
        category_markdown(cats, methods),
        "## Settings",
        "\n".join(f"- {k}: {v}" for k, v in settings.items()),
    ]) + "\n"
    (out / "summary.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {"settings": settings, "summary": summary, "by_category_hit@5": cats, "random_hit@5": baseline},
        indent=2), encoding="utf-8")
    with (out / "per_item.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print(summary_markdown(summary, baseline))
    print(f"\nwrote {out}/summary.md, summary.json, per_item.jsonl")


if __name__ == "__main__":
    main()
