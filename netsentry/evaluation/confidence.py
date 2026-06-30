"""Bootstrap confidence intervals and significance tests for the headline metrics.

A single PR-AUC number invites over-reading; the project's whole claim is the *gap*
between the honest temporal split and the optimistic stratified one, so that gap
deserves an interval and a p-value, not just a point estimate. These are
percentile-bootstrap intervals (resample the evaluation set with replacement,
recompute the metric) — assumption-light and the right tool when the sampling
distribution is unknown.

Two comparison shapes:
- **paired** (same test set, e.g. model vs baseline): resample row indices once and
  score both, so the difference cancels shared sampling noise.
- **independent** (different test sets, e.g. temporal vs stratified): resample each
  set on its own; the gap distribution is their difference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.evaluation.metrics import rates_at_threshold

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class Interval:
    """A point estimate with a percentile-bootstrap confidence interval."""

    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (PR-AUC)."""
    return float(average_precision_score(y_true, scores))


def tpr_at_threshold(threshold: float) -> Metric:
    """A metric that returns TPR at a *fixed* threshold (chosen on validation)."""

    def _metric(y_true: np.ndarray, scores: np.ndarray) -> float:
        return rates_at_threshold(y_true, scores, threshold)["tpr"]

    return _metric


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: Metric,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Interval:
    """Percentile-bootstrap CI for ``metric`` on one evaluation set."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = metric(y_true, scores)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:  # a degenerate resample can't score a ranking metric
            continue
        stats.append(metric(yt, scores[idx]))
    if not stats:
        return Interval(point, point, point)
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(point), float(low), float(high))


@dataclass
class DiffResult:
    """A difference (b - a) with a bootstrap CI and a one-sided p-value."""

    diff: float
    low: float
    high: float
    p_value: float  # P(diff <= 0) under the bootstrap — small => b > a is significant


def independent_diff(
    y_a: np.ndarray,
    s_a: np.ndarray,
    y_b: np.ndarray,
    s_b: np.ndarray,
    metric: Metric,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> DiffResult:
    """Bootstrap the gap ``metric(b) - metric(a)`` for two *independent* test sets."""
    y_a, s_a, y_b, s_b = map(np.asarray, (y_a, s_a, y_b, s_b))
    rng = np.random.default_rng(seed)
    na, nb = len(y_a), len(y_b)
    point = metric(y_b, s_b) - metric(y_a, s_a)
    diffs: list[float] = []
    for _ in range(n_boot):
        ia, ib = rng.integers(0, na, na), rng.integers(0, nb, nb)
        if len(np.unique(y_a[ia])) < 2 or len(np.unique(y_b[ib])) < 2:
            continue
        diffs.append(metric(y_b[ib], s_b[ib]) - metric(y_a[ia], s_a[ia]))
    if not diffs:
        return DiffResult(point, point, point, 1.0)
    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = float(np.mean(np.asarray(diffs) <= 0.0))
    return DiffResult(float(point), float(low), float(high), p_value)
