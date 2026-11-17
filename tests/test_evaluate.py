import numpy as np
import pytest

from rag_eval_mcp.chunking import chunk_text
from rag_eval_mcp.cuad import EvalItem
from rag_eval_mcp.embedding import HashingEmbedder
from rag_eval_mcp.evaluate import by_category, evaluate, sample_items, summarize, summary_markdown
from rag_eval_mcp.index import Index
from rag_eval_mcp.retrieval import Retriever

DOC = ("Payment terms are net thirty days from invoice. " * 5
       + "This Agreement shall be governed by the laws of the State of New York. "
       + "Either party may terminate this agreement on ninety days written notice. " * 5)


def setup():
    chunks = chunk_text("c000", DOC, chunk_size=120, overlap=20)
    span_start = DOC.index("This Agreement shall be governed")
    span = (span_start, span_start + len("This Agreement shall be governed by the laws of the State of New York."))
    items = [
        EvalItem("q1", "c000", "Governing Law", "Governing Law. Which state's law governs the contract?", [span]),
        EvalItem("q2", "c000", "Payment", "payment net thirty days invoice", [(0, 40)]),
    ]
    return Retriever(Index(chunks, HashingEmbedder())), items, {"c000": chunks}


class FirstIsBest:
    name = "fake"

    def score(self, query, texts):
        return np.arange(len(texts), 0, -1, dtype=float)  # keep incoming order


def test_rows_and_summary():
    retriever, items, by_doc = setup()
    rows = evaluate(retriever, items, by_doc, ["bm25", "vector", "hybrid"])
    assert len(rows) == 6
    for r in rows:
        assert len(r["ranked"]) <= 10
        assert 0.0 <= r["mrr@10"] <= 1.0
        assert r["hit@1"] <= r["hit@5"] <= r["hit@10"]  # bigger k can only help
        assert r["recall@5"] <= r["recall@10"]
    bm25_q1 = next(r for r in rows if r["qid"] == "q1" and r["method"] == "bm25")
    assert bm25_q1["hit@1"] == 1.0  # the governing-law chunk is the obvious keyword match

    s = summarize(rows)
    assert set(s) == {"bm25", "vector", "hybrid"}
    assert s["bm25"]["n"] == 2
    table = summary_markdown(s, baseline_hit5=0.3)
    assert "| bm25 |" in table and "random" in table


def test_rerank_method_runs_when_reranker_given():
    retriever, items, by_doc = setup()
    retriever.reranker = FirstIsBest()
    rows = evaluate(retriever, items, by_doc, ["hybrid", "hybrid_rerank"])
    h = {r["qid"]: r["ranked"] for r in rows if r["method"] == "hybrid"}
    rr = {r["qid"]: r["ranked"] for r in rows if r["method"] == "hybrid_rerank"}
    assert h == rr  # a reranker that keeps the order changes nothing


def test_by_category():
    retriever, items, by_doc = setup()
    cats = by_category(evaluate(retriever, items, by_doc, ["bm25"]))
    assert set(cats) == {"Governing Law", "Payment"}
    assert cats["Payment"]["n"] == 1


def test_sample_is_fixed_by_seed():
    items = [EvalItem(str(i), "c", "x", "q", [(0, 1)]) for i in range(100)]
    assert sample_items(items, 10, seed=1) == sample_items(items, 10, seed=1)
    assert sample_items(items, 10, seed=1) != sample_items(items, 10, seed=2)
    assert len(sample_items(items, None)) == 100
