import pytest

from rag_eval_mcp.metrics import hit_at_k, random_hit_at_k, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e"]


def test_hit_at_k():
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0
    assert hit_at_k(RANKED, {"zzz"}, 5) == 0.0


def test_recall_at_k_counts_share_of_relevant_found():
    assert recall_at_k(RANKED, {"a", "d"}, 3) == 0.5
    assert recall_at_k(RANKED, {"a", "d"}, 5) == 1.0
    assert recall_at_k(RANKED, {"a", "zzz"}, 5) == 0.5
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_reciprocal_rank_uses_first_relevant():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"e"}, k=4) == 0.0


def test_random_baseline():
    assert random_hit_at_k(10, 1, 5) == pytest.approx(0.5)  # 5 of 10 slots
    assert random_hit_at_k(10, 0, 5) == 0.0
    assert random_hit_at_k(3, 1, 5) == 1.0  # fewer chunks than k: you get them all
