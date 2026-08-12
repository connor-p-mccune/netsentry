"""Partial-pooled rate estimation: the interval math, the prior fit, and the shrinkage weights.

The pieces here all have analytic answers or hard invariants — a Beta-Binomial with a Beta(1,1)
prior collapses to the Laplace rule of succession, shrinkage is monotone in sample size, and a
Wald interval's failure at zero successes is exactly what Wilson is here to fix — so they are
pinned rather than eyeballed off a plot.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.rare_rates import (
    beta_binomial_logpmf,
    coverage_simulation,
    fit_beta_prior,
    flows_needed,
    jeffreys_interval,
    posterior_interval,
    shrinkage_weight,
    wilson_interval,
)


def test_wilson_stays_open_where_the_wald_interval_collapses() -> None:
    # 0 of 12 successes: the Wald interval has zero width and claims certainty. Wilson does not.
    lo, hi = wilson_interval(0, 12)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi > 0.2  # a twelfth of nothing is very far from proving the rate is zero


def test_wilson_narrows_as_evidence_accumulates() -> None:
    widths = [hi - lo for lo, hi in (wilson_interval(n // 2, n) for n in (10, 100, 1000, 10000))]
    assert widths == sorted(widths, reverse=True)


def test_wilson_brackets_the_observed_proportion() -> None:
    lo, hi = wilson_interval(30, 100)
    assert lo < 0.30 < hi


def test_wilson_on_no_trials_admits_total_ignorance() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_jeffreys_pins_the_boundaries_at_zero_and_one() -> None:
    assert jeffreys_interval(0, 20)[0] == 0.0
    assert jeffreys_interval(20, 20)[1] == 1.0


def test_beta_binomial_with_a_uniform_prior_is_uniform_over_outcomes() -> None:
    # theta ~ Beta(1,1) makes every count 0..n equally likely a priori: P(k) = 1/(n+1).
    n = 5
    probs = [np.exp(beta_binomial_logpmf(k, n, 1.0, 1.0)) for k in range(n + 1)]
    assert np.allclose(probs, 1 / (n + 1))


def test_beta_binomial_probabilities_sum_to_one() -> None:
    n = 9
    total = sum(np.exp(beta_binomial_logpmf(k, n, 2.0, 5.0)) for k in range(n + 1))
    assert total == pytest.approx(1.0)


def test_beta_binomial_rejects_impossible_counts() -> None:
    assert beta_binomial_logpmf(7, 3, 1.0, 1.0) == -np.inf
    assert beta_binomial_logpmf(1, 3, -1.0, 1.0) == -np.inf


def test_posterior_with_a_uniform_prior_is_the_rule_of_succession() -> None:
    # Beta(1,1) + k of n successes gives a posterior mean of (k+1)/(n+2) -- Laplace's rule.
    mean, lo, hi = posterior_interval(3, 10, 1.0, 1.0)
    assert mean == pytest.approx(4 / 12)
    assert lo < mean < hi


def test_posterior_moves_toward_the_prior_mean_when_data_is_scarce() -> None:
    alpha, beta = 6.0, 4.0  # prior mean 0.6, strength 10
    scarce, _, _ = posterior_interval(0, 2, alpha, beta)
    plentiful, _, _ = posterior_interval(0, 2000, alpha, beta)
    assert scarce > 0.4  # two failures barely dent a prior worth ten observations
    assert plentiful < 0.01  # two thousand do


def test_shrinkage_is_monotone_in_sample_size_and_bounded() -> None:
    weights = [shrinkage_weight(2.0, 8.0, n) for n in (1, 10, 100, 10_000)]
    assert weights == sorted(weights, reverse=True)
    assert 0.0 < weights[-1] < weights[0] <= 1.0


def test_shrinkage_is_one_half_when_the_prior_is_worth_the_sample() -> None:
    # A prior of strength 10 against 10 observations splits the estimate evenly.
    assert shrinkage_weight(4.0, 6.0, 10) == pytest.approx(0.5)


def test_fit_beta_prior_recovers_the_generating_mean() -> None:
    rng = np.random.default_rng(0)
    trials = [200] * 40
    thetas = rng.beta(3.0, 7.0, size=40)  # mean 0.3
    successes = [int(rng.binomial(n, t)) for n, t in zip(trials, thetas, strict=True)]
    alpha, beta = fit_beta_prior(successes, trials)
    assert alpha / (alpha + beta) == pytest.approx(0.3, abs=0.06)


def test_fit_beta_prior_finds_a_tight_prior_when_the_classes_agree() -> None:
    # Every class at the same rate: the fitted prior should be concentrated, so shrinkage is
    # strong. Widely disagreeing classes should give a diffuse one.
    agreeing = fit_beta_prior([50] * 12, [100] * 12)
    disagreeing = fit_beta_prior([2, 98, 3, 97, 1, 99] * 2, [100] * 12)
    assert sum(agreeing) > sum(disagreeing)


def test_fit_beta_prior_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_beta_prior([1, 2], [10])


def test_flows_needed_scales_with_the_inverse_square_of_the_precision() -> None:
    # Halving the half-width costs four times the flows.
    assert flows_needed(0.5, 0.05) == pytest.approx(flows_needed(0.5, 0.1) * 4, rel=0.02)


def test_flows_needed_is_largest_at_a_rate_of_one_half() -> None:
    assert flows_needed(0.5, 0.05) > flows_needed(0.1, 0.05) > flows_needed(0.01, 0.05)


def test_credible_intervals_cover_at_their_nominal_rate_under_the_assumed_model() -> None:
    # The guarantee is conditional on the prior: generate from it, and coverage must hold.
    cov_b, cov_w, width_b, width_w = coverage_simulation(
        alpha=2.0, beta=8.0, trials_per_class=[5, 20, 100], n_replicates=300, seed=1
    )
    assert cov_b == pytest.approx(0.95, abs=0.04)
    assert cov_w >= 0.93  # Wilson is conservative here, as expected
    assert width_b < width_w  # and pays for it in width
