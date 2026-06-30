"""Calibration diagnostics: reliability curve, ECE/MCE, and the Brier score.

Gradient-boosted tree outputs are *not* probabilities (``ml.md`` §4): a LightGBM
score of 0.9 does not mean "attack 90% of the time". These metrics quantify the
gap between predicted confidence and observed frequency, so any threshold or
probability the system *reports* can be backed by evidence rather than asserted.

All functions operate on a binary attack/benign target and a 1-D attack score in
``[0, 1]``; they are pure and unit-tested on hand-computed cases.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions and compare mean confidence to observed frequency.

    Equal-width bins over ``[0, 1]``. Returns ``(mean_predicted, observed,
    weight)`` for the non-empty bins only, where ``weight`` is each bin's share
    of the samples (so the three arrays line up and ``weight`` sums to 1).
    """
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize on the interior edges yields bin indices 0..n_bins-1.
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    weight = np.zeros(n_bins)
    n = max(len(prob), 1)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count:
            mean_pred[b] = float(prob[mask].mean())
            observed[b] = float(y_true[mask].mean())
            weight[b] = count / n
    keep = weight > 0
    return mean_pred[keep], observed[keep], weight[keep]


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: sample-weighted mean ``|confidence - accuracy|`` across bins."""
    mean_pred, observed, weight = reliability_curve(y_true, prob, n_bins)
    if len(weight) == 0:
        return 0.0
    return float(np.sum(weight * np.abs(mean_pred - observed)))


def maximum_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """MCE: the worst (largest) per-bin calibration gap."""
    mean_pred, observed, _ = reliability_curve(y_true, prob, n_bins)
    if len(mean_pred) == 0:
        return 0.0
    return float(np.max(np.abs(mean_pred - observed)))


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Brier score (lower is better): mean squared error of the probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:  # brier_score_loss needs both classes present
        return float(np.mean((p - y) ** 2))
    return float(brier_score_loss(y, p))


def calibration_summary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Bundle the three headline calibration numbers for a single score vector."""
    return {
        "brier": brier_score(y_true, prob),
        "ece": expected_calibration_error(y_true, prob, n_bins),
        "mce": maximum_calibration_error(y_true, prob, n_bins),
    }
