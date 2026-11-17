import pytest

from rag_eval_mcp.stats import bootstrap_ci, paired_diff_ci


def test_ci_contains_mean_and_shrinks_with_more_data():
    small = [1, 0] * 20
    big = [1, 0] * 500
    m1, lo1, hi1 = bootstrap_ci(small)
    m2, lo2, hi2 = bootstrap_ci(big)
    assert m1 == m2 == 0.5
    assert lo1 <= 0.5 <= hi1 and lo2 <= 0.5 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)


def test_constant_values_have_zero_width():
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)


def test_paired_diff_detects_consistent_gain():
    # a beats b on 30 of 200 questions and never loses: clearly real.
    b = [0] * 100 + [1] * 100
    a = [1] * 30 + [0] * 70 + [1] * 100
    mean, lo, hi = paired_diff_ci(a, b)
    assert mean == pytest.approx(0.15)
    assert lo > 0


def test_paired_diff_on_noise_includes_zero():
    # a wins 5, loses 5: no real difference.
    a = [1] * 5 + [0] * 5 + [1] * 90
    b = [0] * 5 + [1] * 5 + [1] * 90
    _, lo, hi = paired_diff_ci(a, b)
    assert lo < 0 < hi


def test_errors():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        paired_diff_ci([1, 0], [1])
