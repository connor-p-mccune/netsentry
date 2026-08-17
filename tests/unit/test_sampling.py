"""Sampling designs and the estimator that makes them answerable.

The high-value tests are the unbiasedness of the Horvitz-Thompson estimator under a *skewed*
design (a plain average would pass under uniform sampling and hide the bug), the variance
formula's agreement with the empirical variance, and the property the whole report turns on:
a design with a zero inclusion probability cannot be corrected by any weighting.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.sampling import (
    greedy_probabilities,
    horvitz_thompson,
    naive_total,
    priority_probabilities,
    stratified_probabilities,
    uniform_probabilities,
)


def _population(n: int = 4000, rate: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Attack indicators plus a noisy 'pre-filter' score that correlates with them."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < rate).astype(int)
    scores = np.clip(0.15 + 0.6 * labels + rng.normal(0, 0.15, n), 0.01, 0.99)
    return labels, scores


# --------------------------------------------------------------------------------------
# The designs.
# --------------------------------------------------------------------------------------


def test_uniform_probabilities_match_the_budget() -> None:
    assert np.allclose(uniform_probabilities(100, 0.05), 0.05)


def test_priority_probabilities_respect_the_budget_and_the_floor() -> None:
    _, scores = _population()
    probabilities = priority_probabilities(scores, budget=0.05, floor=0.002)
    assert probabilities.min() >= 0.002
    assert probabilities.max() <= 1.0
    # The realised rate has to land near the budget, or the designs are not being compared at
    # equal cost and every detection number in the report is meaningless.
    assert 0.045 < float(probabilities.mean()) < 0.075


def test_priority_probabilities_favour_the_suspicious_flows() -> None:
    labels, scores = _population()
    probabilities = priority_probabilities(scores, budget=0.05, floor=0.002)
    assert float(probabilities[labels == 1].mean()) > 3 * float(probabilities[labels == 0].mean())


def test_the_floor_is_what_keeps_every_flow_reachable() -> None:
    _, scores = _population()
    with_floor = priority_probabilities(scores, budget=0.01, floor=0.002)
    without_floor = priority_probabilities(scores, budget=0.01, floor=0.0)
    assert (with_floor > 0).all()
    assert float(without_floor.min()) < float(with_floor.min())


def test_stratified_allocation_covers_every_stratum() -> None:
    strata = np.array(["a"] * 900 + ["b"] * 90 + ["c"] * 10)
    probabilities = stratified_probabilities(strata, 0.1, allocation="proportional")
    for label in ("a", "b", "c"):
        assert float(probabilities[strata == label].mean()) == pytest.approx(0.1, abs=1e-6)


def test_neyman_allocation_spends_where_the_variance_is() -> None:
    # Two strata of equal size, one with a wide spread of scores and one nearly constant.
    strata = np.array(["wide"] * 500 + ["narrow"] * 500)
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.random(500), np.full(500, 0.5) + rng.normal(0, 0.01, 500)])
    probabilities = stratified_probabilities(strata, 0.1, allocation="neyman", values=values)
    assert float(probabilities[strata == "wide"].mean()) > float(
        probabilities[strata == "narrow"].mean()
    )


def test_greedy_selection_takes_exactly_the_budget_and_nothing_below_the_cut() -> None:
    _, scores = _population(n=1000)
    probabilities = greedy_probabilities(scores, 0.1)
    assert set(np.unique(probabilities).tolist()) <= {0.0, 1.0}
    assert abs(float(probabilities.mean()) - 0.1) < 0.02
    assert float(scores[probabilities == 1].min()) >= float(scores[probabilities == 0].max())


def test_greedy_leaves_flows_that_can_never_be_observed() -> None:
    # The property the report's central claim rests on: a zero inclusion probability is not a
    # small number, it is an unbounded Horvitz-Thompson weight, i.e. no estimator at all.
    _, scores = _population(n=500)
    assert float(greedy_probabilities(scores, 0.1).min()) == 0.0


# --------------------------------------------------------------------------------------
# The estimator.
# --------------------------------------------------------------------------------------


def test_horvitz_thompson_is_unbiased_under_a_skewed_design() -> None:
    # Unbiasedness under *uniform* sampling is trivial and would hide a wrong weighting. This
    # runs the skewed design, where a naive average is badly wrong and only the 1/pi weights
    # recover the truth.
    labels, scores = _population(n=3000, seed=2)
    truth = int(labels.sum())
    probabilities = priority_probabilities(scores, budget=0.1, floor=0.005)
    rng = np.random.default_rng(3)
    estimates = []
    for _ in range(300):
        taken = rng.random(len(labels)) < probabilities
        total, _ = horvitz_thompson((labels * taken)[taken], probabilities[taken])
        estimates.append(total)
    assert float(np.mean(estimates)) == pytest.approx(truth, rel=0.05)


def test_the_variance_estimator_agrees_with_the_empirical_variance() -> None:
    # A confidence interval built on a wrong variance formula still looks like an interval.
    labels, scores = _population(n=2000, seed=4)
    probabilities = priority_probabilities(scores, budget=0.2, floor=0.01)
    rng = np.random.default_rng(5)
    estimates, variances = [], []
    for _ in range(400):
        taken = rng.random(len(labels)) < probabilities
        total, variance = horvitz_thompson((labels * taken)[taken], probabilities[taken])
        estimates.append(total)
        variances.append(variance)
    empirical = float(np.var(estimates, ddof=1))
    assert 0.6 * empirical < float(np.mean(variances)) < 1.6 * empirical


def test_the_interval_covers_the_truth_about_as_often_as_it_claims() -> None:
    labels, scores = _population(n=2500, seed=6)
    truth = int(labels.sum())
    probabilities = priority_probabilities(scores, budget=0.15, floor=0.01)
    rng = np.random.default_rng(7)
    covered = []
    for _ in range(400):
        taken = rng.random(len(labels)) < probabilities
        total, variance = horvitz_thompson((labels * taken)[taken], probabilities[taken])
        half = 1.96 * float(np.sqrt(max(variance, 0.0)))
        covered.append(abs(total - truth) <= half)
    assert float(np.mean(covered)) > 0.88


def test_the_naive_estimator_is_correct_under_uniform_and_wrong_under_priority() -> None:
    labels, scores = _population(n=3000, seed=8)
    truth = int(labels.sum())
    rng = np.random.default_rng(9)

    uniform = uniform_probabilities(len(labels), 0.1)
    taken = rng.random(len(labels)) < uniform
    assert naive_total((labels * taken)[taken], uniform) == pytest.approx(truth, rel=0.25)

    priority = priority_probabilities(scores, budget=0.1, floor=0.005)
    taken = rng.random(len(labels)) < priority
    biased = naive_total((labels * taken)[taken], priority)
    assert biased > 1.5 * truth  # the sampler's skill, counted twice


def test_horvitz_thompson_of_an_empty_sample_is_zero_rather_than_an_error() -> None:
    total, variance = horvitz_thompson(np.zeros(0), np.zeros(0))
    assert total == 0.0 and variance == 0.0
