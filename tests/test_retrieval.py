import numpy as np
import pytest

from rag_eval_mcp.chunking import Chunk
from rag_eval_mcp.embedding import HashingEmbedder
from rag_eval_mcp.index import Index
from rag_eval_mcp.retrieval import Retriever, reciprocal_rank_fusion

TEXTS = {
    "a": ["This agreement is governed by the laws of New York.",
          "Either party may terminate on thirty days notice.",
          "Payment is due within 60 days of invoice."],
    "b": ["The laws of Delaware govern this contract.",
          "Licensee may not assign this license."],
}


def make_index():
    chunks = [Chunk(f"{d}::{i}", d, t, 0, len(t)) for d, ts in TEXTS.items() for i, t in enumerate(ts)]
    return Index(chunks, HashingEmbedder())


class KeywordReranker:
    """Fake reranker: prefers texts containing 'Delaware'."""
    name = "fake"

    def score(self, query, texts):
        return np.array([1.0 if "Delaware" in t else 0.0 for t in texts])


def test_rrf_by_hand():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], rrf_k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [i for i, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]])] == [1, 3, 2]


def test_rrf_rewards_agreement_between_lists():
    # 'x' is never first but appears in both lists; it should beat items found by one list only.
    fused = reciprocal_rank_fusion([["a", "x"], ["b", "x"]])
    assert fused[0][0] == "x"


@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
def test_each_method_finds_the_obvious_chunk(method):
    hits = Retriever(make_index()).search("governed by the laws of New York", method, k=2)
    assert hits[0].chunk.chunk_id == "a::0"
    assert len(hits) == 2


def test_doc_filter_only_returns_that_document():
    r = Retriever(make_index())
    for method in ("bm25", "vector", "hybrid"):
        hits = r.search("laws govern", method, k=10, doc_id="b")
        assert {h.chunk.doc_id for h in hits} == {"b"}
        assert len(hits) == 2


def test_rerank_reorders_candidates():
    r = Retriever(make_index(), reranker=KeywordReranker())
    assert r.search("which laws govern", "hybrid_rerank", k=1)[0].chunk.chunk_id == "b::0"


def test_errors():
    r = Retriever(make_index())
    with pytest.raises(ValueError):
        r.search("x", "hybrid_rerank")
    with pytest.raises(ValueError):
        r.search("x", "nope")
    with pytest.raises(KeyError):
        r.search("x", "bm25", doc_id="missing")


def test_embedding_cache_roundtrip(tmp_path):
    idx = make_index()
    first = Index.build_cached(idx.chunks, HashingEmbedder(), tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    second = Index.build_cached(idx.chunks, HashingEmbedder(), tmp_path)
    np.testing.assert_array_equal(first.vectors, second.vectors)
    # Different model name -> different cache file.
    Index.build_cached(idx.chunks, HashingEmbedder(dim=64), tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 2
