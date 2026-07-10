"""Threshold refresh: trailing-window re-choice mechanics and budget compliance."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.monitoring.refresh import refresh_threshold, simulate_threshold_policies


def _batch(
    rng: np.random.Generator, benign_center: float, n: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """A labeled batch with overlapping classes, so the FPR threshold has to live
    inside the benign tail (a separable toy would park the cut at the lowest
    attack score and no drift could ever spend the budget)."""
    n_attack = n // 4
    y = np.concatenate([np.zeros(n - n_attack, dtype=int), np.ones(n_attack, dtype=int)])
    s = np.concatenate(
        [
            rng.normal(benign_center, 0.15, n - n_attack).clip(0, 1),
            rng.normal(0.75, 0.10, n_attack).clip(0, 1),
        ]
    )
    return y, s


def test_refresh_uses_only_the_trailing_window() -> None:
    # Two past batches with very different benign score scales; window=1 must
    # calibrate on the recent one only.
    rng = np.random.default_rng(0)
    old = _batch(rng, benign_center=0.1)
    recent = _batch(rng, benign_center=0.5)
    got = refresh_threshold([old[0], recent[0]], [old[1], recent[1]], 0.01, 1, fallback=0.0)
    assert got == threshold_at_fpr(recent[0], recent[1], 0.01)
    assert got != threshold_at_fpr(old[0], old[1], 0.01)


def test_refresh_falls_back_on_one_class_history() -> None:
    y = np.zeros(100, dtype=int)  # a quiet, all-benign stretch: nothing to calibrate
    s = np.linspace(0, 1, 100)
    assert refresh_threshold([y], [s], 0.01, 2, fallback=0.42) == 0.42
    assert refresh_threshold([], [], 0.01, 2, fallback=0.42) == 0.42


def test_refresh_restores_budget_compliance_under_score_drift() -> None:
    # Benign scores drift upward mid-stream: the frozen cut starts over-alerting;
    # the refreshed cut must return the realized FPR toward the budget.
    rng = np.random.default_rng(7)
    calm = [_batch(rng, 0.1) for _ in range(2)]
    drifted = [_batch(rng, 0.45) for _ in range(4)]
    batches = calm + drifted
    batch_y = [b[0] for b in batches]
    batch_s = [b[1] for b in batches]
    y_cal, s_cal = _batch(rng, 0.1, n=2000)
    initial = threshold_at_fpr(y_cal, s_cal, 0.01)

    static, refresh = simulate_threshold_policies(batch_y, batch_s, 0.01, initial, window=2)
    # The frozen threshold blows the budget once benign scores shift up...
    assert max(static.realized_fpr[2:]) > 0.05
    # ...while the refreshed cut, once the drifted batches enter its window, returns
    # to the neighbourhood of the 1% budget for the remainder of the stream.
    assert max(refresh.realized_fpr[-2:]) < 0.05
    assert np.mean(refresh.realized_fpr[-2:]) < np.mean(static.realized_fpr[-2:]) / 2


def test_static_and_refresh_agree_before_any_history_exists() -> None:
    rng = np.random.default_rng(3)
    batches = [_batch(rng, 0.1) for _ in range(3)]
    static, refresh = simulate_threshold_policies(
        [b[0] for b in batches], [b[1] for b in batches], 0.01, initial_threshold=0.7, window=2
    )
    # Batch 0 has no trailing evidence: both policies must run the initial cut.
    assert static.thresholds[0] == refresh.thresholds[0] == 0.7
    assert static.detection[0] == refresh.detection[0]
