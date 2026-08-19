"""NSGA-II's parts, and the geometric claim the report is built on.

The load-bearing test here is the last one: a Pareto-optimal point in a concave stretch of the
front is selected by *no* weighting of the objectives. That is a theorem, so it can be asserted
exactly rather than approximately, and it is the whole reason the report computes a front.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.pareto import (
    GENOME,
    Gene,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    hypervolume,
    pad_features,
    weighted_sum_reachable,
)

# --------------------------------------------------------------------------------------
# Domination and sorting.
# --------------------------------------------------------------------------------------


def test_domination_requires_no_worse_everywhere_and_better_somewhere() -> None:
    assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))  # a genuine trade-off
    assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))  # equality is not better


def test_non_dominated_sort_recovers_known_fronts() -> None:
    objectives = np.array(
        [
            [1.0, 5.0],  # front 0
            [2.0, 3.0],  # front 0
            [4.0, 1.0],  # front 0
            [3.0, 6.0],  # front 1 (dominated by [2, 3])
            [5.0, 7.0],  # front 2 (dominated by [3, 6])
        ]
    )
    fronts = fast_non_dominated_sort(objectives)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[1] == [3]
    assert fronts[2] == [4]


def test_every_individual_lands_in_exactly_one_front() -> None:
    rng = np.random.default_rng(0)
    objectives = rng.random((40, 3))
    fronts = fast_non_dominated_sort(objectives)
    flat = [index for front in fronts for index in front]
    assert sorted(flat) == list(range(40))


# --------------------------------------------------------------------------------------
# Crowding distance.
# --------------------------------------------------------------------------------------


def test_crowding_distance_protects_the_extremes() -> None:
    objectives = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(objectives)
    assert np.isinf(distance[0]) and np.isinf(distance[2])
    assert np.isfinite(distance[1])


def test_crowding_distance_prefers_the_sparse_neighbourhood() -> None:
    # Two points sit almost on top of each other and one sits alone; the lonely one must score
    # higher, or the front collapses towards its crowded middle over generations.
    objectives = np.array([[0.0, 1.0], [0.40, 0.60], [0.42, 0.58], [1.0, 0.0]])
    distance = crowding_distance(objectives)
    assert distance[1] < np.inf and distance[2] < np.inf
    lonely = crowding_distance(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))[1]
    assert lonely > distance[1]


# --------------------------------------------------------------------------------------
# Hypervolume.
# --------------------------------------------------------------------------------------


def test_hypervolume_of_one_point_is_the_box_it_dominates() -> None:
    assert hypervolume(np.array([[1.0, 2.0]]), np.array([4.0, 6.0])) == pytest.approx(12.0)
    assert hypervolume(np.array([[1.0, 1.0, 1.0]]), np.array([3.0, 3.0, 3.0])) == pytest.approx(8.0)


def test_hypervolume_of_two_points_uses_inclusion_exclusion() -> None:
    # Boxes [1,4]x[3,4] (area 3) and [2,4]x[1,4] (area 6) over reference (4, 4), overlapping
    # in [2,4]x[3,4] (area 2): the union is 3 + 6 - 2 = 7.
    front = np.array([[1.0, 3.0], [2.0, 1.0]])
    assert hypervolume(front, np.array([4.0, 4.0])) == pytest.approx(7.0)


def test_hypervolume_grows_when_the_front_improves() -> None:
    reference = np.array([1.0, 1.0, 1.0])
    worse = np.array([[0.5, 0.5, 0.5]])
    better = np.array([[0.5, 0.5, 0.5], [0.2, 0.8, 0.4]])
    assert hypervolume(better, reference) > hypervolume(worse, reference)


def test_points_beyond_the_reference_contribute_nothing() -> None:
    reference = np.array([1.0, 1.0])
    assert hypervolume(np.array([[2.0, 2.0]]), reference) == 0.0
    inside = hypervolume(np.array([[0.5, 0.5]]), reference)
    mixed = hypervolume(np.array([[0.5, 0.5], [3.0, 3.0]]), reference)
    assert inside == pytest.approx(mixed)


# --------------------------------------------------------------------------------------
# The geometric claim.
# --------------------------------------------------------------------------------------


def test_a_convex_front_is_entirely_reachable_by_weighted_sums() -> None:
    front = np.array([[0.0, 10.0], [4.0, 4.0], [10.0, 0.0]])  # the middle point is below the chord
    reachable = weighted_sum_reachable(front, 4000, np.random.default_rng(1))
    assert reachable.all()


def test_a_concave_front_member_is_reachable_by_no_weighting_at_all() -> None:
    """The report's central claim, as an exact statement.

    ``(6, 6)`` is Pareto-optimal — neither endpoint dominates it — and it sits *above* the
    chord joining them, so every linear functional prefers an endpoint. No weight vector
    selects it, and no amount of extra sampling changes that.
    """
    front = np.array([[0.0, 10.0], [6.0, 6.0], [10.0, 0.0]])
    assert not dominates(front[0], front[1]) and not dominates(front[2], front[1])
    reachable = weighted_sum_reachable(front, 20000, np.random.default_rng(2))
    assert reachable[0] and reachable[2]
    assert not reachable[1]


def test_reachability_is_scale_invariant() -> None:
    # Objectives live in different units (a rate, milliseconds, a rate), so the reachability
    # test normalises first; stretching one axis must not change which points are reachable.
    front = np.array([[0.0, 10.0], [6.0, 6.0], [10.0, 0.0]])
    stretched = front * np.array([1.0, 1000.0])
    a = weighted_sum_reachable(front, 8000, np.random.default_rng(3))
    b = weighted_sum_reachable(stretched, 8000, np.random.default_rng(3))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------------------
# The search space.
# --------------------------------------------------------------------------------------


def test_genes_decode_to_their_declared_bounds() -> None:
    for gene in GENOME:
        assert gene.decode(0.0) == pytest.approx(gene.low, rel=1e-6)
        assert gene.decode(1.0) == pytest.approx(gene.high, rel=1e-6)
        assert gene.low <= gene.decode(0.37) <= gene.high


def test_log_scaled_genes_spend_half_the_range_below_the_geometric_mean() -> None:
    gene = Gene("x", 1.0, 1000.0, log=True)
    assert gene.decode(0.5) == pytest.approx(np.sqrt(1000.0), rel=1e-6)


def test_integer_genes_decode_to_integers() -> None:
    gene = Gene("trees", 40, 500, integer=True)
    assert float(gene.decode(0.3)).is_integer()


def test_the_evasion_perturbation_only_moves_the_named_columns() -> None:
    matrix = np.ones((4, 5))
    padded = pad_features(matrix, np.array([1, 3]), 1.5)
    assert padded[:, [0, 2, 4]].tolist() == np.ones((4, 3)).tolist()
    assert padded[:, [1, 3]].tolist() == (np.ones((4, 2)) * 1.5).tolist()
