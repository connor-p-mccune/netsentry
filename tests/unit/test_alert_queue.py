"""Capacity-constrained alert-queue simulation: ranking -> detection vs budget."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.alert_queue import simulate_queue

_KW = {"minutes_per_alert": 10.0, "analyst_minutes_per_day": 420.0}


def _separable(n_attack: int = 200, n_benign: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """A perfectly separable, graded ranking: attacks above benign, no ties."""
    scores = np.concatenate([np.linspace(0.55, 1.0, n_attack), np.linspace(0.0, 0.45, n_benign)])
    y = np.concatenate([np.ones(n_attack), np.zeros(n_benign)])
    return y, scores


def test_recall_is_monotone_in_budget() -> None:
    y, scores = _separable()
    points = simulate_queue(
        y, scores, base_rate=0.01, flows_per_day=100_000, budgets=[100, 500, 1000, 5000], **_KW
    )
    recalls = [p.recall for p in points]
    assert recalls == sorted(recalls)  # more budget never detects fewer attacks
    assert points[-1].recall > points[0].recall  # and a big budget detects more


def test_separable_model_beats_random_triage() -> None:
    y, scores = _separable()
    # base_rate 1% over 100k flows: full detection needs ~1000 alerts/day.
    points = simulate_queue(y, scores, base_rate=0.01, flows_per_day=100_000, budgets=[1000], **_KW)
    p = points[0]
    assert p.recall == 1.0  # separable -> the whole 1% attack budget is caught
    assert p.precision == 1.0  # no false positives at the separating threshold
    assert p.lift > 50  # random triage of 1000/100000 catches ~1%; the model catches all


def test_analyst_headcount_scales_with_budget() -> None:
    y, scores = _separable()
    points = simulate_queue(
        y, scores, base_rate=0.01, flows_per_day=100_000, budgets=[420, 840], **_KW
    )
    # 420 alerts * 10 min / 420 min-per-analyst = 10 analysts; double budget -> double.
    assert points[0].analysts == 10.0
    assert points[1].analysts == 20.0


def test_precision_beats_base_rate_for_a_useful_ranking() -> None:
    y, scores = _separable()
    points = simulate_queue(
        y, scores, base_rate=0.02, flows_per_day=50_000, budgets=[200, 1000], **_KW
    )
    for p in points:
        if p.alerts_per_day > 0:
            assert p.precision >= 0.02  # a real ranking is never worse than random precision
            assert p.lift >= 1.0
