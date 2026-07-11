"""Threshold transfer: quantile arithmetic, compliance bookkeeping, budget trials."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.evaluation.transfer import (
    compliance_share,
    label_budget_trials,
    quantile_threshold,
)


def test_quantile_threshold_hits_budget_on_benign_only_stream() -> None:
    # A dense uniform benign score grid: the (1 - budget) quantile realizes the
    # budget exactly (this is the assumption the policy rests on).
    scores = np.linspace(0.0, 1.0, 10001)
    y = np.zeros_like(scores, dtype=int)
    threshold = quantile_threshold(scores, target_fpr=0.1)
    assert threshold == pytest.approx(0.9)
    assert rates_at_threshold(y, scores, threshold)["fpr"] == pytest.approx(0.1, abs=1e-3)


def test_quantile_threshold_is_biased_up_by_attacks_in_the_stream() -> None:
    # Benign scores low, attacks high, 30% attack mix: the unlabeled quantile
    # lands inside the attack mass — stricter than the benign-only cut, so the
    # policy under-alerts exactly when the stream is hostile.
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.0, 0.5, size=7000)
    attacks = rng.uniform(0.9, 1.0, size=3000)
    mixed = np.concatenate([benign, attacks])
    budget = 0.01
    biased = quantile_threshold(mixed, budget)
    clean = quantile_threshold(benign, budget)
    assert biased > clean
    assert biased >= 0.9  # inside the attack mass: benign FPR is zero, TPR pays
    y = np.concatenate([np.zeros(7000, dtype=int), np.ones(3000, dtype=int)])
    rates = rates_at_threshold(y, mixed, biased)
    assert rates["fpr"] == 0.0
    assert rates["tpr"] < 1.0


def test_compliance_share_counts_both_sides_of_the_budget() -> None:
    budget, factor = 0.01, 2.0
    fprs = np.array([0.01, 0.019, 0.006, 0.03, 0.001])
    # Held: 0.01, 0.019, 0.006 (within [0.005, 0.02]); flooded: 0.03; starved: 0.001.
    assert compliance_share(fprs, budget, factor) == pytest.approx(3 / 5)


def test_label_budget_trials_full_sample_matches_oracle() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=2000)
    y = (scores > 0.8).astype(int)  # separable so the threshold is stable
    budget = 0.05
    fprs, tprs = label_budget_trials(y, scores, k=2000, budget=budget, n_resamples=3, seed=7)
    oracle = rates_at_threshold(y, scores, threshold_at_fpr(y, scores, budget))
    # k == n means every "sample" is the full set: all trials equal the oracle.
    assert np.allclose(fprs, oracle["fpr"])
    assert np.allclose(tprs, oracle["tpr"])


def test_label_budget_trials_are_seeded() -> None:
    rng = np.random.default_rng(2)
    scores = rng.uniform(size=500)
    y = (scores > 0.7).astype(int)
    first = label_budget_trials(y, scores, k=50, budget=0.1, n_resamples=5, seed=11)
    second = label_budget_trials(y, scores, k=50, budget=0.1, n_resamples=5, seed=11)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
