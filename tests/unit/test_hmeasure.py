"""The H-measure: the properties that make it a valid coherent metric (Hand 2009).

A metric that invalidated the project's honesty thesis would be worse than none, so these
pin the anchors of the H-measure's definition — perfect separation scores 1, a trivial
classifier scores 0, it lives in [0, 1], and (like AUC) it depends only on the ranking of
scores, not their scale.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.hmeasure import h_measure


def test_perfect_separation_scores_one() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert h_measure(y, scores) > 0.999


def test_constant_score_scores_zero() -> None:
    # A classifier that says the same thing for everyone is trivial: H = 0.
    y = np.array([0, 0, 1, 1, 0, 1])
    assert h_measure(y, np.full(6, 0.5)) == 0.0


def test_random_scores_are_near_zero() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.3).astype(int)
    assert h_measure(y, rng.random(4000)) < 0.02


def test_in_unit_interval_for_a_middling_classifier() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.4).astype(int)
    scores = 0.3 * y + rng.random(3000)  # informative but overlapping
    h = h_measure(y, scores)
    assert 0.0 < h < 1.0


def test_invariant_to_monotone_score_transform() -> None:
    # The H-measure is built from the ROC hull, so a strictly increasing transform of the
    # scores cannot change it (up to quadrature noise).
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.35).astype(int)
    scores = 0.4 * y + rng.random(2000)
    assert np.isclose(h_measure(y, scores), h_measure(y, np.expm1(scores)), atol=2e-3)


def test_single_class_is_safe() -> None:
    assert h_measure(np.zeros(10, dtype=int), np.linspace(0, 1, 10)) == 0.0
