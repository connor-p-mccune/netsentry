"""Optimal trees: the branch-and-bound answer must equal exhaustive enumeration.

An "optimal" tree is worth nothing without evidence that the search is correct, and the only
convincing evidence is agreement with brute force on problems small enough to enumerate. The
reference implementation below is deliberately naive — every tree of every shape, no pruning —
so a bug in a bound cannot hide in both.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from netsentry.explain.optimal_tree import (
    Node,
    apply_predicates,
    build_predicates,
    optimal_tree,
    tree_objective,
)


# --------------------------------------------------------------------------------------
# Brute force: the reference the search must match
# --------------------------------------------------------------------------------------
def _all_trees(n_predicates: int, depth: int) -> list[Node]:
    """Every tree of depth at most ``depth`` over the predicate set (both leaf labels)."""
    trees: list[Node] = [Node(label=0), Node(label=1)]
    if depth <= 0:
        return trees
    children = _all_trees(n_predicates, depth - 1)
    for predicate in range(n_predicates):
        for left, right in itertools.product(children, repeat=2):
            trees.append(Node(predicate=predicate, left=left, right=right))
    return trees


def _brute_force(
    binary: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: float, depth: int
) -> float:
    """Objective of the best tree, by enumerating all of them."""
    return min(
        tree_objective(tree, binary, y, weights, penalty)
        for tree in _all_trees(binary.shape[1], depth)
    )


def _problem(seed: int, n_rows: int = 40, n_predicates: int = 4) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    binary = rng.integers(0, 2, size=(n_rows, n_predicates)).astype(np.uint8)
    y = (binary[:, 0] ^ (binary[:, 1] & binary[:, 2])).astype(int)
    y = np.where(rng.random(n_rows) < 0.1, 1 - y, y)  # a little label noise
    weights = np.full(n_rows, 1.0 / n_rows)
    return binary, y, weights


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("penalty", [0.0, 0.01, 0.05])
def test_branch_and_bound_matches_exhaustive_enumeration(seed: int, penalty: float) -> None:
    """The load-bearing test: the pruned search finds exactly what brute force finds."""
    binary, y, weights = _problem(seed)
    result = optimal_tree(binary, y, weights, penalty=penalty, max_depth=2, node_budget=10_000_000)
    assert result.certified
    assert result.objective == pytest.approx(_brute_force(binary, y, weights, penalty, 2))


def test_the_returned_tree_actually_achieves_the_reported_objective(seed: int = 7) -> None:
    """A search that reports a good number but returns a different tree would be worthless."""
    binary, y, weights = _problem(seed)
    result = optimal_tree(binary, y, weights, penalty=0.01, max_depth=3, node_budget=10_000_000)
    assert tree_objective(result.tree, binary, y, weights, 0.01) == pytest.approx(result.objective)


def test_a_heavy_penalty_buys_a_single_leaf() -> None:
    """When a leaf costs more than any error it could remove, the optimum is the stump."""
    binary, y, weights = _problem(0)
    result = optimal_tree(binary, y, weights, penalty=10.0, max_depth=3, node_budget=100_000)
    assert result.tree.n_leaves() == 1


def test_a_free_penalty_lets_the_tree_grow_to_fit() -> None:
    binary, y, weights = _problem(0)
    heavy = optimal_tree(binary, y, weights, penalty=0.05, max_depth=3, node_budget=1_000_000)
    free = optimal_tree(binary, y, weights, penalty=0.0, max_depth=3, node_budget=1_000_000)
    assert free.tree.n_leaves() >= heavy.tree.n_leaves()


def test_a_separable_problem_is_solved_exactly() -> None:
    binary = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 0, 1, 1])
    weights = np.full(4, 0.25)
    result = optimal_tree(binary, y, weights, penalty=0.01, max_depth=2, node_budget=10_000)
    assert result.tree.predict(binary).tolist() == y.tolist()
    assert result.objective == pytest.approx(0.02)  # zero error, two leaves


def test_the_certificate_is_withdrawn_when_the_budget_runs_out() -> None:
    """An 'optimal' tree found under a truncated search must not claim to be optimal."""
    binary, y, weights = _problem(0, n_rows=200, n_predicates=8)
    result = optimal_tree(binary, y, weights, penalty=0.0, max_depth=4, node_budget=50)
    assert not result.certified


def test_deeper_search_is_never_worse() -> None:
    binary, y, weights = _problem(3, n_predicates=5)
    objectives = [
        optimal_tree(
            binary, y, weights, penalty=0.005, max_depth=d, node_budget=5_000_000
        ).objective
        for d in (1, 2, 3)
    ]
    assert objectives == sorted(objectives, reverse=True) or len(set(objectives)) == 1


# --------------------------------------------------------------------------------------
# Binarisation
# --------------------------------------------------------------------------------------
def test_predicates_are_thresholds_on_the_chosen_features() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    y = (x[:, 0] > 0).astype(int)
    binary, labels, spec = build_predicates(x, y, [f"f{i}" for i in range(5)], 2, 3)
    assert binary.shape == (200, len(labels)) and len(spec) == len(labels)
    assert set(binary.ravel().tolist()) <= {0, 1}
    assert all(">" in label for label in labels)


def test_the_strongest_feature_is_selected() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    y = (x[:, 2] > 0).astype(int)
    x[:, 2] += y * 6.0  # feature 2 separates the classes and nothing else does
    _, labels, _ = build_predicates(x, y, ["a", "b", "c", "d"], 1, 2)
    assert all(label.startswith("c ") for label in labels)


def test_test_rows_are_binarised_with_the_training_thresholds() -> None:
    """Re-fitting thresholds on test rows would be leakage wearing a preprocessing costume."""
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(200, 3))
    y = (x_train[:, 0] > 0).astype(int)
    _, _, spec = build_predicates(x_train, y, ["a", "b", "c"], 2, 2)
    x_test = rng.normal(size=(50, 3))
    binary_test = apply_predicates(x_test, spec)
    assert binary_test.shape == (50, len(spec))
    for column, (j, threshold) in enumerate(spec):
        assert np.array_equal(binary_test[:, column], (x_test[:, j] > threshold).astype(np.uint8))


# --------------------------------------------------------------------------------------
# The tree object
# --------------------------------------------------------------------------------------
def test_a_leaf_predicts_its_label_for_everything() -> None:
    leaf = Node(label=1)
    assert leaf.n_leaves() == 1
    assert leaf.predict(np.zeros((5, 2), dtype=np.uint8)).tolist() == [1] * 5


def test_a_split_routes_true_right_and_false_left() -> None:
    tree = Node(predicate=0, left=Node(label=0), right=Node(label=1))
    binary = np.array([[0], [1], [0]], dtype=np.uint8)
    assert tree.predict(binary).tolist() == [0, 1, 0]
    assert tree.n_leaves() == 2


def test_the_description_names_the_predicate_it_splits_on() -> None:
    tree = Node(predicate=0, left=Node(label=0), right=Node(label=1))
    text = "\n".join(tree.describe(["Flow Duration > 3"]))
    assert "Flow Duration > 3" in text and "ATTACK" in text and "benign" in text
