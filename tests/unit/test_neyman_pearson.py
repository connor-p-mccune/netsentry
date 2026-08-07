"""Neyman-Pearson thresholds: the finite-sample guarantee, proved on the closed form.

The report's whole claim is one identity — the population FPR of an order-statistic
threshold is Beta-distributed, so its violation probability is a binomial tail. These pin
that identity against an independent Monte-Carlo estimate, pin the certified rule as the
*largest* admissible one (a smaller one would be valid but needlessly deaf), and pin the
sample-size floor below which no threshold certifies the budget at all.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from netsentry.evaluation.neyman_pearson import (
    expected_fpr,
    holdout_violation_rate,
    log_binom_cdf,
    min_calibration_size,
    naive_count,
    np_admissible_count,
    rates_above,
    simulate_violation_rate,
    threshold_from_count,
    violation_probability,
)


def test_log_binom_cdf_matches_a_direct_sum_on_small_inputs() -> None:
    n, p = 20, 0.3
    for m in range(n + 1):
        direct = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(m + 1))
        assert math.isclose(math.exp(log_binom_cdf(n, p, m)), direct, rel_tol=1e-10)


def test_log_binom_cdf_is_exactly_one_at_the_full_range() -> None:
    assert math.isclose(math.exp(log_binom_cdf(50, 0.01, 50)), 1.0, rel_tol=1e-12)


def test_log_binom_cdf_survives_a_tail_that_underflows_in_linear_space() -> None:
    # (1 - 0.001)^1e6 is about e^-1000: zero in float64, finite in log space.
    value = log_binom_cdf(1_000_000, 0.001, 0)
    assert value < -900.0 and math.isfinite(value)


def test_violation_probability_matches_the_beta_sampling_distribution() -> None:
    """The identity the guarantee rests on, checked against simulation.

    Draw n uniform benign scores, take the order statistic that lets m through, and the
    fraction of the *population* above it is exactly 1 - U_(n-m). Its chance of exceeding
    alpha should be the binomial tail the closed form reports.
    """
    rng = np.random.default_rng(0)
    n, m, alpha = 200, 4, 0.05
    draws = rng.random((20_000, n))
    thresholds = np.sort(draws, axis=1)[:, n - m - 1]
    true_fpr = 1.0 - thresholds  # uniform scores: P(score > t) = 1 - t
    measured = float(np.mean(true_fpr > alpha))
    assert abs(measured - violation_probability(n, alpha, m)) < 0.01


def test_naive_quantile_rule_exceeds_its_budget_about_half_the_time() -> None:
    # The headline indictment: an empirical quantile is a coin flip, not a budget.
    n, alpha = 10_000, 0.001
    assert 0.4 < violation_probability(n, alpha, naive_count(n, alpha)) < 0.7


def test_violation_probability_increases_with_a_more_permissive_threshold() -> None:
    probs = [violation_probability(5_000, 0.01, m) for m in range(0, 120, 10)]
    assert all(b >= a for a, b in pairwise(probs))


def test_admissible_count_is_the_largest_rule_inside_the_promise() -> None:
    n, alpha, delta = 5_000, 0.01, 0.05
    m = np_admissible_count(n, alpha, delta)
    assert m is not None
    assert violation_probability(n, alpha, m) <= delta
    # ...and one step more permissive would break it, which is what makes it the largest.
    assert violation_probability(n, alpha, m + 1) > delta


def test_admissible_count_is_stricter_than_the_empirical_quantile() -> None:
    n, alpha, delta = 20_000, 0.001, 0.05
    m = np_admissible_count(n, alpha, delta)
    assert m is not None and m < naive_count(n, alpha)


def test_admissible_count_is_none_below_the_sample_size_floor() -> None:
    alpha, delta = 0.001, 0.05
    floor = min_calibration_size(alpha, delta)
    assert np_admissible_count(floor - 50, alpha, delta) is None
    assert np_admissible_count(floor + 50, alpha, delta) == 0


def test_min_calibration_size_is_the_point_where_the_conservative_rule_just_holds() -> None:
    alpha, delta = 0.01, 0.05
    n = min_calibration_size(alpha, delta)
    assert violation_probability(n, alpha, 0) <= delta
    assert violation_probability(n - 1, alpha, 0) > delta


def test_relaxing_delta_admits_a_more_permissive_threshold() -> None:
    counts = [np_admissible_count(20_000, 0.001, d) for d in (0.01, 0.05, 0.20)]
    assert all(c is not None for c in counts)
    assert counts[0] <= counts[1] <= counts[2]  # type: ignore[operator]


def test_expected_fpr_of_the_quantile_rule_sits_above_the_budget() -> None:
    # (floor(n*alpha) + 1) / (n + 1) > alpha — the bias is structural, not a small-sample fluke.
    for n in (1_000, 10_000, 100_000):
        assert expected_fpr(n, naive_count(n, 0.01)) > 0.01


def test_threshold_from_count_lets_exactly_that_many_flows_through() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for m in range(0, 5):
        t = threshold_from_count(scores, m)
        assert int(np.sum(scores > t)) == m


def test_threshold_from_count_at_zero_sits_above_every_calibration_flow() -> None:
    scores = np.array([0.2, 0.9, 0.5])
    assert threshold_from_count(scores, 0) == 0.9


def test_rates_above_uses_strict_exceedance_so_the_boundary_flow_is_not_alerted() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    tpr, fpr = rates_above(y, scores, 0.5)
    assert tpr == 0.5 and fpr == 0.0  # the two 0.5 flows sit on the line and are cleared


def test_rates_above_handles_a_single_class_stream() -> None:
    tpr, fpr = rates_above(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5)
    assert tpr == 0.0 and fpr > 0.0


def test_simulate_violation_rate_reproduces_the_closed_form() -> None:
    # The rank simulation is the independent check on the binomial identity.
    for m in (0, 3, 12):
        exact = simulate_violation_rate(2_000, 0.01, m, n_sims=8_000, seed=1)
        assert abs(exact - violation_probability(2_000, 0.01, m)) < 0.02


def test_simulate_violation_rate_is_invariant_to_a_monotone_score_transform() -> None:
    """The claim that licenses simulating with uniforms: the rule reads only ranks."""
    rng = np.random.default_rng(3)
    n, m, alpha = 500, 5, 0.02
    uniform = rng.random((4_000, n))
    warped = uniform**3  # strictly increasing: same ranks, wildly different distribution
    t_u = np.sort(uniform, axis=1)[:, n - m - 1]
    t_w = np.sort(warped, axis=1)[:, n - m - 1]
    # The population FPR is the same random variable under either scale.
    assert np.allclose(1.0 - t_u, 1.0 - t_w ** (1 / 3))
    assert abs(np.mean((1.0 - t_u) > alpha) - violation_probability(n, alpha, m)) < 0.02


def test_holdout_measurement_inflates_the_violation_rate_of_a_certified_rule() -> None:
    """A finite holdout cannot validate a finite-sample bound, and errs in one direction.

    A certified rule sits below budget by construction, so holdout sampling noise can only
    push the *measured* FPR across the line — never back. The measurement therefore reads
    high, which looks like the guarantee failing when it is the instrument failing.
    """
    rng = np.random.default_rng(7)
    n_cal, alpha, delta = 1_500, 0.02, 0.05
    pool = rng.random(2 * n_cal)
    m = np_admissible_count(n_cal, alpha, delta)
    assert m is not None
    measured, _ = holdout_violation_rate(pool, alpha, m, n_cal, n_splits=200, seed=7)
    exact = simulate_violation_rate(n_cal, alpha, m, n_sims=8_000, seed=7)
    assert exact <= delta
    assert measured > exact
