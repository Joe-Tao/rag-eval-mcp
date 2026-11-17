"""Is a difference between two methods real, or just which questions we happened to test?

Bootstrap: pretend the eval set is a sample from all possible questions. Redraw
the same number of questions WITH replacement, recompute the mean, repeat a few
thousand times; the middle 95% of those means is the confidence interval.

Paired: every method answered the SAME questions, so we resample questions once
and take the difference on each. That cancels out "this question is just hard
for everyone" and gives much tighter intervals than comparing two separate CIs.
If the 95% interval of (A - B) does not include 0, the difference is unlikely
to be noise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_resamples: int = 5000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """-> (mean, low, high)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(x.mean()), float(low), float(high)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_resamples: int = 5000, seed: int = 0,
                   level: float = 0.95) -> tuple[float, float, float]:
    """CI of mean(a - b), where a[i] and b[i] are two methods' scores on question i."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must score the same questions")
    return bootstrap_ci(a - b, n_resamples, seed, level)
