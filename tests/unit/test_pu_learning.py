"""Positive-unlabeled learning: the estimator algebra and the corrected budget arithmetic.

The report's claims rest on pure machinery — the SCAR labeling, the Elkan-Noto ``c``
estimator, the posterior odds weighting, the duplicated weighted design, and the
de-contaminated FPR quantile. Each is pinned here on constructed data, including the
budget-distortion value case (a hidden attack in the 'benign' pool over-tightens the naive
cut; the PU denominator does not fall for it).
"""

from __future__ import annotations

import numpy as np

from netsentry.training.pu_learning import (
    correct_scores,
    estimate_c,
    expand_weighted,
    prevalence_from_pu,
    pu_threshold_for_budget,
    scar_labels,
    unlabeled_posterior_weights,
)


def test_scar_labels_only_marks_true_positives_at_the_requested_fraction() -> None:
    y = np.array([1] * 40 + [0] * 60)
    s = scar_labels(y, 0.25, np.random.default_rng(0))
    assert s.sum() == 10  # 25% of the 40 positives
    assert np.all(y[s == 1] == 1)  # a confirmed label can only sit on a real attack


def test_scar_labels_is_deterministic_under_the_same_rng_seed() -> None:
    y = np.array([1, 0] * 50)
    first = scar_labels(y, 0.5, np.random.default_rng(7))
    second = scar_labels(y, 0.5, np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_estimate_c_is_the_mean_g_over_labeled_positives() -> None:
    g = np.array([0.5, 0.3, 0.4, 0.9, 0.1])
    s = np.array([1, 1, 1, 0, 0])
    assert np.isclose(estimate_c(g, s), 0.4)
    assert estimate_c(g, np.zeros(5)) == 1.0  # no labeled positives: degenerate fallback


def test_estimate_c_recovers_the_label_frequency_on_separable_data() -> None:
    # With a calibrated g = c * p(y|x) and near-0/1 posteriors, e1 lands on c.
    rng = np.random.default_rng(1)
    c = 0.3
    posterior = np.where(rng.random(4000) < 0.4, 0.999, 0.001)  # separable world
    y = (rng.random(4000) < posterior).astype(int)
    s = scar_labels(y, c, rng)
    g = c * posterior
    assert abs(estimate_c(g, s) - c) < 0.01


def test_correct_scores_divides_by_c_and_caps_at_one_preserving_order() -> None:
    g = np.array([0.05, 0.2, 0.4, 0.9])
    corrected = correct_scores(g, 0.5)
    assert np.allclose(corrected, [0.1, 0.4, 0.8, 1.0])
    assert np.all(np.diff(corrected) >= 0)  # monotone: ranking untouched by construction


def test_unlabeled_posterior_weights_match_the_hand_worked_odds_ratio() -> None:
    # c = 0.5: w = 1 * g/(1-g). g = 1/3 gives w = 0.5; a huge g clips to 1.
    w = unlabeled_posterior_weights(np.array([1.0 / 3.0, 0.999]), 0.5, score_clip=1e-6)
    assert np.isclose(w[0], 0.5)
    assert w[1] == 1.0


def test_expand_weighted_keeps_unit_mass_per_unlabeled_row() -> None:
    x = np.arange(10, dtype=float).reshape(5, 2)
    s = np.array([1, 0, 0, 1, 0])
    w = np.array([0.2, 0.7, 0.4])
    x_out, y_out, w_out = expand_weighted(x, s, w)
    assert len(x_out) == 2 + 2 * 3  # positives once, unlabeled twice
    assert np.all(y_out[:2] == 1) and np.all(w_out[:2] == 1.0)  # confirmed rows untouched
    # Each unlabeled row's positive + negative copies carry exactly its original mass.
    assert np.allclose(w_out[2:5] + w_out[5:8], 1.0)
    assert np.all(y_out[2:5] == 1) and np.all(y_out[5:8] == 0)


def test_prevalence_from_pu_is_mean_g_over_c_capped() -> None:
    assert np.isclose(prevalence_from_pu(0.06, 0.3), 0.2)
    assert prevalence_from_pu(0.9, 0.3) == 1.0


def test_pu_threshold_ignores_hidden_attack_mass_in_the_denominator() -> None:
    # Top score is a hidden attack (benign mass 0). Naive bookkeeping counts it as a
    # false positive and pins the cut at 0.9; the PU denominator lets the cut relax to
    # 0.8 within the same budget — the report's value case, constructed.
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    pu_mass = np.array([0.0, 1.0, 1.0, 1.0])
    naive_mass = np.ones(4)
    assert pu_threshold_for_budget(scores, pu_mass, budget=0.34) == 0.8
    assert pu_threshold_for_budget(scores, naive_mass, budget=0.34) == 0.9


def test_pu_threshold_respects_the_estimated_budget_on_random_data() -> None:
    rng = np.random.default_rng(2)
    scores = rng.random(500)
    mass = rng.random(500)
    budget = 0.05
    t = pu_threshold_for_budget(scores, mass, budget)
    realized = mass[scores >= t].sum() / mass.sum()
    assert realized <= budget + 1e-12


def test_pu_threshold_degenerate_mass_returns_the_strictest_cut() -> None:
    scores = np.array([0.4, 0.2])
    assert pu_threshold_for_budget(scores, np.zeros(2), 0.01) == 0.4
