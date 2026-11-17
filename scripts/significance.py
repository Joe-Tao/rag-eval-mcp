"""Confidence intervals for the eval results, from results/per_item.jsonl (no re-run needed).

    python scripts/significance.py                       # baseline = bm25
    python scripts/significance.py --baseline hybrid --metric mrr@10

Writes results/significance.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag_eval_mcp.stats import bootstrap_ci, paired_diff_ci

METRICS = ("hit@5", "mrr@10", "recall@10")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--baseline", default="bm25")
    p.add_argument("--resamples", type=int, default=5000)
    args = p.parse_args()

    rows = [json.loads(line) for line in (Path(args.results) / "per_item.jsonl").open()]
    # scores[method][metric] -> {qid: value}
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in METRICS:
            scores[r["method"]][m][r["qid"]] = r[m]
    methods = list(scores)
    if args.baseline not in scores:
        raise SystemExit(f"baseline {args.baseline!r} not in results: {methods}")
    qids = sorted(scores[args.baseline]["hit@5"])
    n = len(qids)

    out = [f"# Significance ({n} questions, {args.resamples} bootstrap resamples, 95% intervals)", ""]

    out += ["## Each method", "", "| method | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for method in methods:
        cells = []
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([scores[method][m][q] for q in qids], args.resamples)
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        out.append(f"| {method} | " + " | ".join(cells) + " |")

    out += ["", f"## Difference vs {args.baseline} (paired)", "",
            "Positive = better than the baseline. **Bold** = interval excludes 0, i.e. unlikely to be noise.", "",
            "| method | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for method in methods:
        if method == args.baseline:
            continue
        cells = []
        for m in METRICS:
            a = [scores[method][m][q] for q in qids]
            b = [scores[args.baseline][m][q] for q in qids]
            mean, lo, hi = paired_diff_ci(a, b, args.resamples)
            cell = f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells.append(f"**{cell}**" if lo > 0 or hi < 0 else cell)
        out.append(f"| {method} | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    (Path(args.results) / "significance.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
