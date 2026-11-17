"""Try a query by hand and see what each method returns.

    python scripts/demo_search.py "Which state's law governs this agreement?" --doc c000
    python scripts/demo_search.py "termination for convenience" --methods bm25 vector --k 3
    python scripts/demo_search.py "governing law" --fake-models   # no model download
"""

from __future__ import annotations

import argparse
import time

from rag_eval_mcp.factory import load_corpus, make_models, make_retriever
from rag_eval_mcp.retrieval import METHODS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--doc", default=None, help="restrict to one contract, e.g. c000")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--cuad", default="data/cuad_test.json")
    p.add_argument("--fake-models", action="store_true", help="hashing embedder, no reranker")
    args = p.parse_args()

    corpus = load_corpus(args.cuad)
    embedder, reranker = make_models(args.fake_models)
    methods = [m for m in args.methods if m != "hybrid_rerank" or reranker is not None]

    t0 = time.perf_counter()
    retriever = make_retriever(corpus, embedder, reranker)
    print(f"index: {len(corpus.chunks)} chunks from {len(corpus.contracts)} contracts "
          f"({time.perf_counter() - t0:.1f}s)")

    for method in methods:
        t0 = time.perf_counter()
        hits = retriever.search(args.query, method, k=args.k, doc_id=args.doc)
        print(f"\n=== {method} ({(time.perf_counter() - t0) * 1000:.0f} ms) ===")
        for h in hits:
            snippet = " ".join(h.chunk.text.split())[:160]
            print(f"  {h.score:8.3f}  {h.chunk.chunk_id:<10} {snippet}")


if __name__ == "__main__":
    main()
