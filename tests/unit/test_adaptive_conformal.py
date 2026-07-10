"""Adaptive conformal: the ACI update, quantile lookups, and coverage repair."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.adaptive_conformal import (
    AdaptiveAlpha,
    QuantileTable,
    rolling_class_coverage,
    run_adaptive_stream,
    run_static_stream,
)
from netsentry.evaluation.conformal import conformal_quantile


def test_quantile_table_matches_split_conformal_quantile() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    table = QuantileTable(scores)
    for alpha in (0.05, 0.1, 0.25, 0.5):
        assert table.threshold(alpha) == pytest.approx(conformal_quantile(scores, alpha))


def test_quantile_table_extreme_alphas() -> None:
    table = QuantileTable(np.array([0.1, 0.2, 0.3]))
    assert table.threshold(0.0) == float("inf")  # alpha <= 0: include everything
    assert table.threshold(-0.5) == float("inf")
    assert table.threshold(1.0) == float("-inf")  # alpha >= 1: the empty set
    assert QuantileTable(np.array([])).threshold(0.1) == float("inf")


def test_alpha_holds_steady_when_errors_match_target() -> None:
    # err rate exactly the target (1 miss per 10) leaves alpha where it started.
    tracker = AdaptiveAlpha(target=0.1, gamma=0.01)
    for step in range(200):
        tracker.update(step % 10 == 0)
    assert tracker.alpha == pytest.approx(0.1, abs=0.01)


def test_persistent_miscoverage_drives_alpha_negative() -> None:
    # Every prediction misses: alpha must fall without bound (sets forced wide).
    tracker = AdaptiveAlpha(target=0.1, gamma=0.05)
    values = [tracker.update(True) for _ in range(50)]
    assert values == sorted(values, reverse=True)  # strictly decreasing
    assert tracker.alpha < 0  # deliberately unclamped: ACI's wide-open excursion


def test_adaptive_restores_coverage_where_static_fails() -> None:
    # Calibration attacks score ~0.9 (nonconformity ~0.1); stream attacks drift to
    # ~0.5 — every one falls outside the static set, the textbook broken guarantee.
    rng = np.random.default_rng(7)
    cal_p = np.concatenate([rng.uniform(0.0, 0.2, 500), rng.uniform(0.85, 0.95, 500)])
    cal_y = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    stream_p = np.tile(np.array([0.05, 0.5]), 1000)  # benign stays; attacks drift
    stream_y = np.tile(np.array([0, 1]), 1000)

    static = run_static_stream(stream_p, stream_y, *_taus(cal_p, cal_y, alpha=0.1))
    adaptive, _, alpha_attack = run_adaptive_stream(
        stream_p, stream_y, cal_p, cal_y, target_alpha=0.1, gamma=0.02
    )
    assert static.coverage(stream_y, 1) < 0.05  # static: drifted attacks never covered
    # ACI drives alpha down until the sets include the drifted attacks again.
    late = slice(len(stream_y) // 2, None)
    late_attacks = stream_y[late] == 1
    assert float(np.mean(adaptive.in_attack[late][late_attacks])) > 0.8
    assert min(alpha_attack.history) < 0.1


def test_label_delay_defers_the_first_updates() -> None:
    cal_p = np.concatenate([np.linspace(0.0, 0.2, 50), np.linspace(0.8, 1.0, 50)])
    cal_y = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    stream_p = np.full(10, 0.5)
    stream_y = np.ones(10, dtype=int)
    _, _, no_delay = run_adaptive_stream(stream_p, stream_y, cal_p, cal_y, 0.1, 0.05)
    _, _, delayed = run_adaptive_stream(stream_p, stream_y, cal_p, cal_y, 0.1, 0.05, label_delay=4)
    assert len(no_delay.history) == 10
    assert len(delayed.history) == 6  # the last 4 labels never arrived in time


def test_rolling_coverage_windows_are_class_conditional() -> None:
    outcome = run_static_stream(np.array([0.05, 0.9, 0.05, 0.9]), np.array([0, 1, 0, 1]), 0.5, 0.5)
    xs, cov = rolling_class_coverage(outcome, np.array([0, 1, 0, 1]), 1, window=2)
    assert list(xs) == [2, 4]
    assert list(cov) == [1.0, 1.0]  # attacks at 0.9 are inside the attack set


def _taus(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float) -> tuple[float, float]:
    from netsentry.evaluation.conformal import class_conditional_thresholds

    return class_conditional_thresholds(cal_p, cal_y, alpha)
