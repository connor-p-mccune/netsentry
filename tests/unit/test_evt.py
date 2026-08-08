"""Extreme-value tail fitting: parameter recovery, the quantile inversion, and the endpoint.

A GPD fit is easy to get subtly wrong and hard to notice, because a wrong shape still
produces a plausible-looking threshold. These pin it against ground truth three ways:
draws from a known GPD (does the fit recover what generated it), an independent
implementation (SciPy), and the closed-form tail of a distribution whose quantiles are
known exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netsentry.evaluation.evt import (
    GPDFit,
    Population,
    empirical_threshold,
    fit_gpd,
    gpd_quantile,
    mean_excess,
    pot_threshold,
    simulate_estimators,
)


def _gpd_sample(rng: np.random.Generator, n: int, xi: float, sigma: float) -> np.ndarray:
    """Inverse-CDF draw from GPD(xi, sigma) — the ground truth a fit must recover."""
    u = rng.random(n)
    if abs(xi) < 1e-12:
        return -sigma * np.log1p(-u)
    return (sigma / xi) * ((1.0 - u) ** (-xi) - 1.0)


@pytest.mark.parametrize(("xi", "sigma"), [(0.35, 1.0), (0.0, 2.0), (-0.25, 1.5)])
def test_fit_recovers_the_parameters_that_generated_the_sample(xi: float, sigma: float) -> None:
    rng = np.random.default_rng(11)
    sample = _gpd_sample(rng, 20_000, xi, sigma)
    fit = fit_gpd(sample, 0.0)
    assert fit is not None
    assert abs(fit.xi - xi) < 0.05
    assert abs(fit.sigma - sigma) < 0.12 * sigma


def test_fit_agrees_with_scipys_independent_implementation() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    sample = _gpd_sample(rng, 8_000, 0.2, 1.0)
    fit = fit_gpd(sample, 0.0)
    assert fit is not None
    xi_ref, _loc, sigma_ref = scipy_stats.genpareto.fit(sample, floc=0.0)
    assert abs(fit.xi - xi_ref) < 0.03
    assert abs(fit.sigma - sigma_ref) < 0.05


def test_fit_uses_only_the_exceedances_above_the_declared_threshold() -> None:
    rng = np.random.default_rng(2)
    # Bulk that is nothing like a GPD, plus a genuine exponential tail above 5.
    bulk = rng.normal(0.0, 0.3, 4_000)
    tail = 5.0 + rng.exponential(1.0, 2_000)
    fit = fit_gpd(np.concatenate([bulk, tail]), 5.0)
    assert fit is not None
    assert fit.n_exceed == 2_000
    assert abs(fit.xi) < 0.08  # the bulk did not drag the shape away from exponential


def test_fit_returns_none_when_the_tail_has_no_distinct_exceedances() -> None:
    assert fit_gpd(np.array([0.5, 0.5, 0.5, 0.5]), 0.4) is None
    assert fit_gpd(np.array([1.0, 2.0, 3.0]), 5.0) is None


def test_negative_shape_reports_a_finite_upper_endpoint() -> None:
    fit = GPDFit(xi=-0.5, sigma=1.0, u=10.0, n_exceed=100, n_total=1_000, log_likelihood=0.0)
    assert fit.upper_endpoint == 12.0  # u - sigma/xi
    # The quantile approaches the endpoint rather than running off: the tail has ended.
    assert gpd_quantile(fit, 1e-12) == pytest.approx(12.0, abs=1e-4)
    assert gpd_quantile(fit, 1e-12) < 12.0


def test_positive_shape_has_no_upper_endpoint() -> None:
    fit = GPDFit(xi=0.3, sigma=1.0, u=0.0, n_exceed=100, n_total=1_000, log_likelihood=0.0)
    assert math.isinf(fit.upper_endpoint)
    assert gpd_quantile(fit, 1e-8) > gpd_quantile(fit, 1e-4)


def test_gpd_quantile_inverts_the_tail_of_a_known_exponential() -> None:
    """P(X > x) = tail_rate * exp(-(x-u)/sigma); the inversion must reproduce it exactly."""
    fit = GPDFit(xi=0.0, sigma=2.0, u=1.0, n_exceed=100, n_total=1_000, log_likelihood=0.0)
    x = gpd_quantile(fit, 0.001)
    assert 0.1 * math.exp(-(x - 1.0) / 2.0) == pytest.approx(0.001, rel=1e-9)


def test_gpd_quantile_declines_to_extrapolate_below_the_tail_threshold() -> None:
    fit = GPDFit(xi=0.1, sigma=1.0, u=3.0, n_exceed=50, n_total=1_000, log_likelihood=0.0)
    assert gpd_quantile(fit, 0.5) == 3.0  # a budget looser than the tail itself


def test_pot_threshold_falls_back_to_the_empirical_quantile_when_the_tail_will_not_fit() -> None:
    constant = np.full(500, 0.25)
    threshold, fit = pot_threshold(constant, 0.01)
    assert fit is None
    assert threshold == empirical_threshold(constant, 0.01)


def test_mean_excess_is_linear_in_the_threshold_for_a_gpd_tail() -> None:
    """The diagnostic's defining property: e(u) = (sigma + xi*u)/(1 - xi) is a straight line."""
    rng = np.random.default_rng(9)
    sample = _gpd_sample(rng, 60_000, 0.2, 1.0)
    us = np.array([0.5, 1.0, 1.5, 2.0])
    curve = mean_excess(sample, us)
    slopes = np.diff(curve) / np.diff(us)
    assert np.allclose(slopes, slopes[0], rtol=0.2)


def test_populations_tail_probability_inverts_their_own_quantiles() -> None:
    for pop in (
        Population("exponential", 0.0, ""),
        Population("heavy (Pareto, xi=1/3)", 1 / 3, ""),
        Population("uniform", -1.0, ""),
    ):
        rng = np.random.default_rng(4)
        sample = pop.sample(rng, 200_000)
        for q in (0.01, 0.001):
            x = float(np.quantile(sample, 1.0 - q))
            assert pop.tail_probability(np.array([x]))[0] == pytest.approx(q, rel=0.15)


def test_simulation_reports_both_estimators_on_a_budget_they_can_both_reach() -> None:
    rows = simulate_estimators(
        Population("exponential", 0.0, ""),
        0.001,
        n=2_000,
        trials=25,
        tail_quantile=0.95,
        grid_points=120,
        seed=0,
    )
    assert {r.method for r in rows} == {"empirical quantile", "EVT (peaks-over-threshold)"}
    # Both should land within a factor of a few of the budget on an easy population.
    assert all(0.1 < r.median_ratio < 10.0 for r in rows)


def test_empirical_threshold_degenerates_to_the_sample_maximum_below_one_expected_flow() -> None:
    rng = np.random.default_rng(1)
    sample = rng.random(500)
    # n * budget < 1: there is no order statistic to read, only the largest score seen.
    assert empirical_threshold(sample, 0.0001) == float(sample.max())
