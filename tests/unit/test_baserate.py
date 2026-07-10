"""Base-rate arithmetic: Bayes precision and its two inversions, hand-checked."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.baserate import (
    bayes_precision,
    break_even_prior,
    required_fpr,
    sweep_priors,
)


def test_bayes_precision_hand_computed() -> None:
    # pi=0.5, TPR=0.5, FPR=0.01: 0.25 / (0.25 + 0.005) = 0.98039...
    assert bayes_precision(0.5, 0.01, 0.5) == pytest.approx(0.25 / 0.255)
    # pi=0.01: 0.005 / (0.005 + 0.0099) — the same detector, ~3x worse queue.
    assert bayes_precision(0.5, 0.01, 0.01) == pytest.approx(0.005 / 0.0149)


def test_precision_is_monotone_in_prevalence() -> None:
    priors = np.logspace(-5, -1, 9)
    precisions = [bayes_precision(0.3, 0.001, float(p)) for p in priors]
    assert precisions == sorted(precisions)  # rarer attacks can only hurt the queue


def test_perfect_and_degenerate_edges() -> None:
    assert bayes_precision(0.5, 0.0, 0.001) == 1.0  # no false positives: pure queue
    assert bayes_precision(0.0, 0.0, 0.001) == 0.0  # nothing fires: defined as zero
    assert bayes_precision(0.5, 0.01, 0.0) == 0.0  # no attacks exist: every alert lies


def test_required_fpr_inverts_bayes_precision() -> None:
    tpr, prior, target = 0.3, 0.0001, 0.9
    fpr = required_fpr(tpr, prior, target)
    assert bayes_precision(tpr, fpr, prior) == pytest.approx(target)


def test_required_fpr_shrinks_with_prevalence() -> None:
    # An order of magnitude rarer attack demands an order of magnitude cleaner FPR.
    loose = required_fpr(0.3, 0.001, 0.9)
    tight = required_fpr(0.3, 0.0001, 0.9)
    assert tight == pytest.approx(loose / 10, rel=0.02)


def test_break_even_prior_is_the_fifty_percent_point() -> None:
    tpr, fpr = 0.2, 0.01
    prior = break_even_prior(tpr, fpr)
    assert prior == pytest.approx(fpr / (tpr + fpr))
    assert bayes_precision(tpr, fpr, prior) == pytest.approx(0.5)


def test_sweep_priors_volume_arithmetic() -> None:
    rows = sweep_priors(tpr=0.5, fpr=0.001, priors=[0.01], flows_per_day=1_000_000)
    (row,) = rows
    assert row["attacks_caught_per_day"] == pytest.approx(1_000_000 * 0.01 * 0.5)
    assert row["false_alerts_per_day"] == pytest.approx(1_000_000 * 0.99 * 0.001)
    assert row["alerts_per_day"] == pytest.approx(
        row["attacks_caught_per_day"] + row["false_alerts_per_day"]
    )
    assert row["precision"] == pytest.approx(row["attacks_caught_per_day"] / row["alerts_per_day"])
