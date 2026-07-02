"""Active-learning acquisition tests: uncertainty picks the boundary flows,
random is seed-reproducible, and both handle the budget-exceeds-pool edge case."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.active_learning import (
    RoundPoint,
    _labels_to_reach,
    select_random,
    select_uncertain,
)


def test_uncertainty_selects_probabilities_nearest_half() -> None:
    probs = np.array([0.01, 0.48, 0.52, 0.95, 0.5, 0.2])
    unlabeled = np.arange(len(probs))
    chosen = select_uncertain(probs, unlabeled, k=3)
    # The three flows nearest 0.5 are indices 4 (0.50), 1 (0.48), 2 (0.52).
    assert set(chosen.tolist()) == {1, 2, 4}


def test_uncertainty_respects_the_unlabeled_mask() -> None:
    probs = np.array([0.5, 0.5, 0.9, 0.55])
    unlabeled = np.array([2, 3])  # the two most-uncertain (0,1) are already labeled
    chosen = select_uncertain(probs, unlabeled, k=1)
    assert chosen.tolist() == [3]  # 0.55 is closer to 0.5 than 0.9


def test_uncertainty_returns_all_when_budget_exceeds_pool() -> None:
    probs = np.array([0.4, 0.6])
    unlabeled = np.array([0, 1])
    chosen = select_uncertain(probs, unlabeled, k=5)
    assert set(chosen.tolist()) == {0, 1}


def test_random_is_seed_reproducible_and_disjoint_from_labeled() -> None:
    unlabeled = np.arange(100, 200)
    a = select_random(unlabeled, 10, np.random.default_rng(0))
    b = select_random(unlabeled, 10, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    assert set(a.tolist()).issubset(set(unlabeled.tolist()))
    assert len(set(a.tolist())) == 10  # sampled without replacement


def test_random_returns_all_when_budget_exceeds_pool() -> None:
    unlabeled = np.array([1, 2, 3])
    chosen = select_random(unlabeled, 10, np.random.default_rng(1))
    assert set(chosen.tolist()) == {1, 2, 3}


def test_labels_to_reach_finds_first_crossing() -> None:
    curve = [
        RoundPoint(500, 0.40, 0.1),
        RoundPoint(1000, 0.55, 0.2),
        RoundPoint(1500, 0.60, 0.3),
    ]
    assert _labels_to_reach(curve, 0.55) == 1000
    assert _labels_to_reach(curve, 0.99) is None
