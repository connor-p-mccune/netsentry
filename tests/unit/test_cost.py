"""Cost-sensitive threshold selection: rates, expected cost, and the Bayes optimum."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.cost import (
    bayes_threshold,
    cost_optimal_threshold,
    per_flow_cost_rates,
)


def test_bayes_threshold_is_cost_ratio() -> None:
    assert bayes_threshold(25.0, 500.0) == pytest.approx(0.05)
    assert bayes_threshold(5.0, 0.0) == 0.0  # guard against divide-by-zero


def test_per_flow_cost_rates_known_case() -> None:
    # prior 0.1: 0.1 attacks (20% missed @100, 80% alerted @5) + 0.9 benign (10% FP @5).
    cost = per_flow_cost_rates(tpr=0.8, fpr=0.1, prior=0.1, cost_per_alert=5.0, cost_per_miss=100.0)
    expected = 0.1 * (0.2 * 100.0 + 0.8 * 5.0) + 0.9 * 0.1 * 5.0
    assert cost == pytest.approx(expected)


def test_optimal_threshold_separates_and_costs_only_triage() -> None:
    # Separable scores: optimum has tpr=1, fpr=0, so per-flow cost = prior*cost_alert.
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
    thr, cost = cost_optimal_threshold(
        y, scores, prior=0.1, cost_per_alert=5.0, cost_per_miss=2000.0
    )
    assert 0.2 < thr <= 0.8
    assert cost == pytest.approx(0.1 * 5.0)


def test_rarer_attacks_raise_the_optimal_threshold() -> None:
    # A lower production base rate means the benign pool dominates the false-alarm
    # cost, so the cost-optimal single threshold rises (be more conservative).
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    common, _ = cost_optimal_threshold(y, p, prior=0.20, cost_per_alert=25.0, cost_per_miss=500.0)
    rare, _ = cost_optimal_threshold(y, p, prior=0.01, cost_per_alert=25.0, cost_per_miss=500.0)
    assert rare > common


def test_expensive_miss_pushes_threshold_down() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    cheap, _ = cost_optimal_threshold(y, p, prior=0.05, cost_per_alert=10.0, cost_per_miss=40.0)
    dear, _ = cost_optimal_threshold(y, p, prior=0.05, cost_per_alert=10.0, cost_per_miss=5000.0)
    assert dear < cheap
