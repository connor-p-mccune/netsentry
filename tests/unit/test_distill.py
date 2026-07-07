"""Distillation fidelity math: matched-volume thresholds and agreement metrics."""

from __future__ import annotations

import numpy as np

from netsentry.explain.distill import fidelity_metrics, matched_volume_threshold


def test_matched_volume_threshold_hits_the_requested_fraction() -> None:
    scores = np.linspace(0.0, 1.0, 1001)
    threshold = matched_volume_threshold(scores, alert_fraction=0.1)
    assert np.isclose(np.mean(scores >= threshold), 0.1, atol=1e-3)


def test_zero_volume_means_alert_on_nothing() -> None:
    scores = np.linspace(0.0, 1.0, 100)
    assert matched_volume_threshold(scores, alert_fraction=0.0) == float("inf")
    assert matched_volume_threshold(np.array([]), alert_fraction=0.5) == float("inf")


def test_perfect_copy_has_perfect_fidelity() -> None:
    teacher = np.array([0.1, 0.9, 0.4, 0.7, 0.2])
    metrics = fidelity_metrics(
        teacher, teacher.copy(), teacher_threshold=0.5, surrogate_threshold=0.5
    )
    assert np.isclose(metrics["spearman"], 1.0)
    assert metrics["decision_agreement"] == 1.0


def test_anticorrelated_surrogate_is_exposed() -> None:
    teacher = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    metrics = fidelity_metrics(
        teacher, 1.0 - teacher, teacher_threshold=0.5, surrogate_threshold=0.5
    )
    assert np.isclose(metrics["spearman"], -1.0)
    assert metrics["decision_agreement"] == 0.0  # every verdict flips


def test_agreement_uses_each_sides_own_threshold() -> None:
    # The surrogate scores on a different scale; volume-matched thresholds make the
    # verdicts comparable even though the raw values never overlap.
    teacher = np.array([0.1, 0.2, 0.8, 0.9])
    surrogate = teacher * 0.1  # same ranking, tenth of the scale
    metrics = fidelity_metrics(teacher, surrogate, teacher_threshold=0.5, surrogate_threshold=0.05)
    assert np.isclose(metrics["spearman"], 1.0)
    assert metrics["decision_agreement"] == 1.0


def test_constant_scores_yield_zero_not_nan() -> None:
    teacher = np.array([0.3, 0.7, 0.5])
    flat = np.full(3, 0.5)
    metrics = fidelity_metrics(teacher, flat, teacher_threshold=0.5, surrogate_threshold=0.5)
    assert metrics["spearman"] == 0.0  # degenerate ranking is reported as no signal
