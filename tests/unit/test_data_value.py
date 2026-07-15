"""KNN-Shapley data valuation: the closed-form recursion (checked against brute-force
exact Shapley), and the mislabel-recovery reads."""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

from netsentry.evaluation.data_value import flip_recovery, knn_shapley_values


def _knn_utility(
    subset: tuple[int, ...], x: np.ndarray, y: np.ndarray, x_q: np.ndarray, y_q: int, k: int
) -> float:
    """Utility of a training subset for a 1-NN/K-NN vote on one query point."""
    if not subset:
        return 0.0
    idx = np.array(subset)
    dist = np.linalg.norm(x[idx] - x_q, axis=1)
    order = idx[np.argsort(dist, kind="stable")]
    kk = min(k, len(order))
    # Standard KNN-Shapley utility (Jia et al.): normalise by K, so a subset smaller
    # than K cannot reach full utility (missing neighbours contribute 0).
    return float(np.sum(y[order[:kk]] == y_q) / k)


def _brute_force_shapley(
    x: np.ndarray, y: np.ndarray, x_q: np.ndarray, y_q: int, k: int
) -> np.ndarray:
    """Exact Shapley by enumerating every subset — the ground truth for small n."""
    n = len(x)
    phi = np.zeros(n)
    others = list(range(n))
    for j in range(n):
        rest = [i for i in others if i != j]
        for size in range(len(rest) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(rest, size):
                with_j = _knn_utility((*combo, j), x, y, x_q, y_q, k)
                without = _knn_utility(combo, x, y, x_q, y_q, k)
                phi[j] += weight * (with_j - without)
    return phi


def test_recursion_matches_brute_force_exact_shapley() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 2))
    y = np.array([0, 1, 0, 1, 1, 0])
    x_q = rng.standard_normal((1, 2))
    y_q = 1
    for k in (1, 2, 3):
        fast = knn_shapley_values(x, y, x_q, np.array([y_q]), k=k)
        exact = _brute_force_shapley(x, y, x_q[0], y_q, k=k)
        assert np.allclose(fast, exact, atol=1e-9), f"mismatch at K={k}"


def test_matching_neighbour_scores_positive_opposite_negative() -> None:
    # Query at the origin; a same-label point sits on it, an opposite-label point too.
    x = np.array([[0.0], [0.0], [5.0]])
    y = np.array([1, 0, 1])
    values = knn_shapley_values(x, y, np.array([[0.0]]), np.array([1]), k=1)
    assert values[0] > 0.0  # the near, correctly-labelled point helps
    assert values[1] < 0.0  # the near, wrongly-labelled point hurts


def test_values_average_over_queries() -> None:
    x = np.array([[0.0], [1.0]])
    y = np.array([1, 0])
    one = knn_shapley_values(x, y, np.array([[0.0]]), np.array([1]), k=1)
    two = knn_shapley_values(x, y, np.array([[0.0], [0.0]]), np.array([1, 1]), k=1)
    assert np.allclose(one, two)  # identical queries -> identical mean value


def test_empty_inputs_are_safe() -> None:
    y = np.array([1, 0, 1])
    no_train = knn_shapley_values(np.zeros((0, 2)), np.array([]), np.ones((3, 2)), y, k=2)
    assert no_train.shape == (0,)
    no_query = knn_shapley_values(np.ones((3, 2)), y, np.zeros((0, 2)), np.array([]), k=2)
    assert np.allclose(no_query, 0.0)


def test_flip_recovery_perfect_when_flips_are_most_negative() -> None:
    values = np.array([-0.9, -0.8, 0.1, 0.2, 0.3])
    is_flipped = np.array([True, True, False, False, False])
    r = flip_recovery(values, is_flipped)
    assert r["precision_at_flips"] == 1.0 and r["auc"] == 1.0 and r["n_flips"] == 2.0


def test_flip_recovery_chance_when_values_carry_no_signal() -> None:
    values = np.array([0.1, 0.1, 0.1, 0.1])
    is_flipped = np.array([True, False, True, False])
    r = flip_recovery(values, is_flipped)
    assert r["auc"] == 0.5


def test_flip_recovery_no_flips_is_safe() -> None:
    r = flip_recovery(np.array([0.1, -0.2, 0.3]), np.array([False, False, False]))
    assert r["n_flips"] == 0.0 and r["precision_at_flips"] == 0.0
