"""Build the retrieval eval set from the CUAD test split and print what it contains.

    python scripts/build_eval_set.py
    python scripts/build_eval_set.py --max-contracts 10 --out data/eval/cuad_small.jsonl   # CI-sized
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from rag_eval_mcp.chunking import chunk_text
from rag_eval_mcp.cuad import load_cuad, relevant_chunk_ids, write_jsonl
from rag_eval_mcp.metrics import random_hit_at_k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cuad", default="data/cuad_test.json")
    p.add_argument("--out", default="data/eval/cuad_test_eval.jsonl")
    p.add_argument("--max-contracts", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=1000, help="only used for the stats below")
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    contracts, items, stats = load_cuad(args.cuad, args.max_contracts)
    write_jsonl(items, args.out)

    chunks_by_doc = {c.doc_id: chunk_text(c.doc_id, c.text, args.chunk_size, args.overlap) for c in contracts}
    n_chunks = [len(v) for v in chunks_by_doc.values()]
    n_relevant = [len(relevant_chunk_ids(chunks_by_doc[it.doc_id], it.spans)) for it in items]
    span_lens = [e - s for it in items for s, e in it.spans]
    # Chance that a random pick of 5 chunks from the contract hits a relevant one,
    # averaged over items. A floor any real method should clear easily.
    random_r5 = statistics.mean(
        random_hit_at_k(len(chunks_by_doc[it.doc_id]), r, 5) for it, r in zip(items, n_relevant)
    )

    print(f"wrote {len(items)} items -> {args.out}")
    print(f"contracts: {len(contracts)}   questions: {stats['questions']}   "
          f"unanswerable (skipped): {stats['unanswerable']}   bad spans: {stats['bad_spans']}")
    print(f"categories: {len({it.category for it in items})}")
    print(f"chunks per contract @ {args.chunk_size}/{args.overlap}: "
          f"median {statistics.median(n_chunks)}, min {min(n_chunks)}, max {max(n_chunks)}, total {sum(n_chunks)}")
    print(f"gold spans: {len(span_lens)}, length chars median {statistics.median(span_lens)}, "
          f"longer than chunk: {sum(l > args.chunk_size for l in span_lens)}")
    print(f"relevant chunks per item: median {statistics.median(n_relevant)}, max {max(n_relevant)}")
    print(f"random-baseline recall@5: {random_r5:.3f}")
    print("items per category (top 5 / bottom 5):")
    counts = Counter(it.category for it in items).most_common()
    for cat, n in counts[:5] + [("...", "")] + counts[-5:]:
        print(f"  {cat:<40} {n}")


if __name__ == "__main__":
    main()
