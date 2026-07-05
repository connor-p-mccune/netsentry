"""Bootstrap confidence intervals and the gap significance test."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.confidence import (
    Interval,
    bootstrap_ci,
    independent_diff,
    paired_diff,
    pr_auc,
    tpr_at_threshold,
)


def _separable(n: int = 600, sep: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    scores = np.clip(rng.normal(np.where(y == 1, 0.5 + sep / 2, 0.5 - sep / 2), 0.2), 0, 1)
    return y, scores


def test_ci_brackets_point_and_orders() -> None:
    y, s = _separable()
    ci = bootstrap_ci(y, s, pr_auc, n_boot=200, seed=1)
    assert isinstance(ci, Interval)
    assert ci.low <= ci.point <= ci.high
    assert ci.low >= 0.0 and ci.high <= 1.0


def test_perfect_separation_has_tight_high_ci() -> None:
    y = np.array([0, 0, 0, 1, 1, 1] * 50)
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 50)
    ci = bootstrap_ci(y, s, pr_auc, n_boot=200, seed=2)
    assert ci.point == 1.0
    assert ci.low > 0.95  # perfectly separable -> CI hugs 1.0


def test_tpr_at_threshold_metric() -> None:
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.8, 0.1])
    metric = tpr_at_threshold(0.5)
    assert metric(y, s) == 0.5  # one of two attacks is >= 0.5


def test_independent_diff_detects_a_real_gap() -> None:
    # A clearly-separable set should beat a barely-separable one (positive gap, low p).
    y_a, s_a = _separable(sep=0.2, seed=3)  # weak
    y_b, s_b = _separable(sep=2.0, seed=4)  # strong
    result = independent_diff(y_a, s_a, y_b, s_b, pr_auc, n_boot=300, seed=5)
    assert result.diff > 0
    assert result.p_value < 0.05  # gap is significant
    assert result.low <= result.diff <= result.high


def test_independent_diff_no_gap_is_not_significant() -> None:
    # Comparing a set against itself -> the gap distribution is symmetric about zero.
    y, s = _separable(sep=1.0, seed=6)
    result = independent_diff(y, s, y, s, pr_auc, n_boot=300, seed=8)
    assert result.low < 0 < result.high  # CI straddles zero
    assert 0.2 < result.p_value < 0.8  # no directional signal


def test_paired_diff_cancels_shared_noise() -> None:
    # Same rows scored by a strong and a weak model: paired delta is positive and
    # significant, and its CI is tighter than the independent comparison's.
    y, s_weak = _separable(sep=0.4, seed=9)
    rng = np.random.default_rng(10)
    s_strong = np.clip(s_weak + np.where(y == 1, 0.15, -0.15) + rng.normal(0, 0.02, len(y)), 0, 1)
    paired = paired_diff(y, s_weak, s_strong, pr_auc, n_boot=300, seed=11)
    independent = independent_diff(y, s_weak, y, s_strong, pr_auc, n_boot=300, seed=11)
    assert paired.diff > 0 and paired.p_value < 0.05
    assert (paired.high - paired.low) < (independent.high - independent.low)


def test_paired_diff_of_identical_models_is_exactly_zero() -> None:
    y, s = _separable(seed=12)
    result = paired_diff(y, s, s, pr_auc, n_boot=100, seed=13)
    assert result.diff == 0.0 and result.low == 0.0 and result.high == 0.0


def test_paired_diff_supports_per_side_thresholds() -> None:
    # Each side judged at its own operating threshold (the promotion gate's shape).
    y = np.array([1, 1, 1, 0, 0, 0] * 40)
    s_a = np.array([0.9, 0.6, 0.4, 0.3, 0.2, 0.1] * 40)  # catches 2/3 at 0.5
    s_b = s_a  # same scores, but the challenger's threshold is looser
    result = paired_diff(
        y, s_a, s_b, tpr_at_threshold(0.5), metric_b=tpr_at_threshold(0.35), n_boot=50, seed=14
    )
    assert result.diff > 0  # looser threshold detects strictly more on these scores
