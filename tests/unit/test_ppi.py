"""Prediction-powered inference: the estimator's algebra and its validity/efficiency claims.

These pin the two properties the report leans on — PPI is *unbiased* (the rectifier
cancels model bias) and *tighter than classical* (a useful residual has less variance
than the label) — plus the degenerate cases where it must fall back to classical.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.ppi import (
    classical_mean_ci,
    effective_sample_size_gain,
    naive_mean_ci,
    ppi_mean_ci,
)


def test_constant_score_reduces_ppi_to_classical() -> None:
    # A score that never varies carries no information: PPI must give back classical.
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.35).astype(float)
    f_lab = np.full(50, 0.7)
    f_unl = np.full(2000, 0.7)
    classical = classical_mean_ci(y[:50], alpha=0.1)
    ppi = ppi_mean_ci(y[:50], f_lab, f_unl, alpha=0.1)
    assert np.isclose(ppi.point, classical.point)
    assert np.isclose(ppi.halfwidth, classical.halfwidth)


def test_perfect_predictions_make_ppi_far_tighter() -> None:
    # With f == y, the rectifier variance is zero, so PPI's width collapses to the
    # (large-N) model-mean term — much tighter than classical's small-n interval.
    rng = np.random.default_rng(1)
    pool = (rng.random(4000) < 0.3).astype(float)
    n = 100
    y_lab = pool[:n]
    classical = classical_mean_ci(y_lab, alpha=0.1)
    ppi = ppi_mean_ci(y_lab, y_lab.copy(), pool.copy(), alpha=0.1)
    assert ppi.halfwidth < 0.3 * classical.halfwidth


def test_ppi_point_is_unbiased_under_a_biased_model() -> None:
    # A deliberately biased score f = 0.5*y + 0.4. Naive averages the bias in;
    # PPI's rectifier subtracts it, recovering the true prevalence exactly when the
    # audit is the whole pool.
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.3).astype(float)
    f = 0.5 * y + 0.4
    naive = naive_mean_ci(f, alpha=0.1)
    ppi = ppi_mean_ci(y, f, f, alpha=0.1)
    assert np.isclose(ppi.point, y.mean())  # rectifier cancels the bias exactly
    assert not np.isclose(naive.point, y.mean())  # naive keeps the model's bias


def test_classical_ci_is_the_sample_mean() -> None:
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    ci = classical_mean_ci(y, alpha=0.1)
    assert np.isclose(ci.point, 0.6)
    assert ci.lo < ci.point < ci.hi
    assert ci.covers(0.6) and not ci.covers(0.6 + 10 * ci.halfwidth)


def test_effective_sample_size_gain_bounds() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert effective_sample_size_gain(y, y.copy()) == float("inf")  # perfect residual
    constant = np.full_like(y, 0.5)
    assert np.isclose(effective_sample_size_gain(y, constant), 1.0)  # no information


def test_empty_inputs_are_safe() -> None:
    empty = np.array([])
    assert not np.isfinite(classical_mean_ci(empty, 0.1).point)
    assert not np.isfinite(ppi_mean_ci(empty, empty, np.array([0.2]), 0.1).point)
