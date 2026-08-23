"""The transport primitives, and the two properties the study's conclusion rests on.

Two things here are easy to get wrong and impossible to notice from a plausible-looking
report. The one-dimensional solver has a closed form, so an implementation that is merely
*approximately* right can be caught outright -- and it is the reference the sliced estimator
and the entropic solver are both graded against. And the entropic solver's cost must approach
the exact assignment as the regularisation falls; a solver that converges to something else
would still produce a smooth, publishable-looking curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.monitoring.transport import (
    ArmSummary,
    RegRow,
    barycentric_targets,
    exact_assignment,
    quantile_transport_map,
    random_directions,
    sinkhorn,
    sliced_permutation_test,
    sliced_wasserstein,
    squared_cost_matrix,
    step_toward,
    wasserstein_1d,
)

# --------------------------------------------------------------------------------------
# One dimension, where the answer is known in closed form.
# --------------------------------------------------------------------------------------


def test_transport_of_a_pure_shift_is_the_shift() -> None:
    """W_p between a sample and a translate of itself is exactly the translation."""
    sample = np.random.default_rng(0).normal(size=500)
    assert wasserstein_1d(sample, sample + 2.5) == pytest.approx(2.5, abs=1e-9)
    assert wasserstein_1d(sample, sample + 2.5, p=2.0) == pytest.approx(2.5, abs=1e-9)


def test_transport_of_a_sample_against_itself_is_zero() -> None:
    sample = np.random.default_rng(1).normal(size=64)
    assert wasserstein_1d(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_transport_is_symmetric_and_handles_unequal_sizes() -> None:
    """The quantile integral is exact for any pair of sizes, not only equal ones."""
    rng = np.random.default_rng(2)
    left = rng.normal(size=37)
    right = rng.normal(loc=1.0, size=91)
    assert wasserstein_1d(left, right) == pytest.approx(wasserstein_1d(right, left))


def test_transport_matches_the_brute_force_assignment_in_one_dimension() -> None:
    """The monotone plan is optimal in 1-D: the sorted pairing must equal the LP optimum."""
    rng = np.random.default_rng(3)
    left = rng.normal(size=40)
    right = rng.normal(loc=0.8, size=40)
    _, brute = exact_assignment(np.abs(left[:, None] - right[None, :]))
    assert wasserstein_1d(left, right) == pytest.approx(brute, rel=1e-9)


def test_the_quantile_map_carries_the_source_onto_the_target() -> None:
    """Pushing the source through its own map reproduces the target's distribution."""
    rng = np.random.default_rng(4)
    source = rng.normal(size=800)
    target = rng.normal(loc=3.0, scale=2.0, size=800)
    mapped = quantile_transport_map(source, target, source)
    assert wasserstein_1d(mapped, target) < 0.05
    # ...and it is monotone, which is what makes it *the* optimal map rather than a map.
    order = np.argsort(source)
    assert np.all(np.diff(mapped[order]) >= -1e-9)


# --------------------------------------------------------------------------------------
# Sliced transport.
# --------------------------------------------------------------------------------------


def test_the_sliced_distance_is_zero_between_a_sample_and_itself() -> None:
    rng = np.random.default_rng(5)
    sample = rng.normal(size=(120, 6))
    directions = random_directions(6, 32, rng)
    assert sliced_wasserstein(sample, sample, directions=directions) == pytest.approx(0.0)


def test_the_sliced_distance_grows_with_the_separation() -> None:
    rng = np.random.default_rng(6)
    sample = rng.normal(size=(200, 4))
    directions = random_directions(4, 64, rng)
    near = sliced_wasserstein(sample, sample + 0.5, directions=directions)
    far = sliced_wasserstein(sample, sample + 2.0, directions=directions)
    assert 0.0 < near < far


def test_the_vectorised_equal_size_path_matches_the_general_one() -> None:
    """The fast path exists for the permutation test; it must not change the answer."""
    rng = np.random.default_rng(7)
    left = rng.normal(size=(64, 5))
    right = rng.normal(loc=0.7, size=(64, 5))
    directions = random_directions(5, 16, rng)
    fast = sliced_wasserstein(left, right, directions=directions)
    slow = float(
        np.mean(
            [
                wasserstein_1d(left @ directions[k], right @ directions[k])
                for k in range(len(directions))
            ]
        )
    )
    assert fast == pytest.approx(slow, rel=1e-9)


def test_the_permutation_test_does_not_fire_on_a_stationary_stream() -> None:
    """A monitor whose false-alarm rate is unknown is not a monitor."""
    rng = np.random.default_rng(8)
    pooled = rng.normal(size=(300, 5))
    test = sliced_permutation_test(
        pooled[:150], pooled[150:], rng=rng, projections=32, permutations=49
    )
    assert not test.rejects(0.05)


def test_the_permutation_test_fires_on_a_real_shift() -> None:
    rng = np.random.default_rng(9)
    test = sliced_permutation_test(
        rng.normal(size=(150, 5)),
        rng.normal(loc=1.0, size=(150, 5)),
        rng=rng,
        projections=32,
        permutations=49,
    )
    assert test.rejects(0.05)


def test_the_permutation_p_value_is_never_zero() -> None:
    """A finite permutation set cannot license `p = 0`, and reporting it is a real error."""
    rng = np.random.default_rng(10)
    test = sliced_permutation_test(
        rng.normal(size=(80, 3)),
        rng.normal(loc=50.0, size=(80, 3)),
        rng=rng,
        projections=8,
        permutations=19,
    )
    assert test.p_value == pytest.approx(1.0 / 20.0)


# --------------------------------------------------------------------------------------
# The full coupling.
# --------------------------------------------------------------------------------------


def test_the_exact_assignment_is_optimal_against_every_permutation() -> None:
    """Small enough to enumerate: the Hungarian answer must be the minimum over all plans."""
    from itertools import permutations

    rng = np.random.default_rng(11)
    cost = rng.random((6, 6))
    _, best = exact_assignment(cost)
    brute = min(sum(cost[i, order[i]] for i in range(6)) / 6 for order in permutations(range(6)))
    assert best == pytest.approx(brute)


def test_sinkhorn_approaches_the_exact_optimum_as_the_regularisation_falls() -> None:
    """The whole validation section depends on this limit actually holding."""
    rng = np.random.default_rng(12)
    source = rng.normal(size=(40, 3))
    target = rng.normal(loc=1.0, size=(40, 3))
    cost = squared_cost_matrix(source, target)
    _, exact = exact_assignment(cost)
    uniform = np.full(40, 1.0 / 40)
    median = float(np.median(cost))
    coarse = sinkhorn(uniform, uniform, cost, reg=0.5 * median, max_iter=400, tol=1e-9)
    fine = sinkhorn(uniform, uniform, cost, reg=0.01 * median, max_iter=400, tol=1e-9)
    assert coarse.cost > fine.cost >= exact - 1e-6
    assert abs(fine.cost - exact) < abs(coarse.cost - exact)


def test_sinkhorn_returns_a_plan_with_the_requested_marginals() -> None:
    """A plan that does not respect its marginals is not a transport plan."""
    rng = np.random.default_rng(13)
    cost = squared_cost_matrix(rng.normal(size=(30, 2)), rng.normal(size=(30, 2)))
    source = np.full(30, 1.0 / 30)
    result = sinkhorn(source, source, cost, reg=0.2 * float(np.median(cost)), max_iter=500)
    assert np.allclose(result.plan.sum(axis=1), source, atol=1e-4)
    assert np.allclose(result.plan.sum(axis=0), source, atol=1e-4)


def test_sinkhorn_survives_a_regularisation_that_underflows_the_naive_form() -> None:
    """The log-domain claim, pinned: the multiplicative kernel is all zeros at this setting."""
    rng = np.random.default_rng(14)
    cost = squared_cost_matrix(rng.normal(size=(20, 4)) * 30.0, rng.normal(size=(20, 4)) * 30.0)
    reg = 1e-3
    assert np.exp(-cost / reg).max() == 0.0  # the naive kernel is entirely underflowed
    result = sinkhorn(np.full(20, 0.05), np.full(20, 0.05), cost, reg=reg, max_iter=100)
    assert np.isfinite(result.cost) and result.cost > 0.0


def test_the_barycentric_projection_of_a_permutation_plan_is_the_partner() -> None:
    """With a hard plan the map has to return exactly the assigned target, not an average."""
    rng = np.random.default_rng(15)
    target = rng.normal(size=(5, 3))
    order = np.array([2, 0, 4, 1, 3])
    plan = np.zeros((5, 5))
    plan[np.arange(5), order] = 0.2
    assert np.allclose(barycentric_targets(plan, target), target[order])


def test_stepping_toward_a_target_respects_the_budget_and_stops_there() -> None:
    """The arms are only comparable if the budget is a cap on distance, not a fraction."""
    source = np.zeros((3, 2))
    target = np.array([[3.0, 4.0], [0.5, 0.0], [0.0, -10.0]])  # norms 5, 0.5, 10
    moved = step_toward(source, target, 1.0)
    distances = np.linalg.norm(moved - source, axis=1)
    assert distances[0] == pytest.approx(1.0)
    assert distances[1] == pytest.approx(0.5)  # already inside the budget: do not overshoot
    assert distances[2] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# The records the report reads from.
# --------------------------------------------------------------------------------------


def test_the_floor_multiple_prices_the_aggregate_against_sampling_noise() -> None:
    summary = ArmSummary(
        arm="x",
        plan_cost=1.0,
        distinct_targets=10,
        detection=0.1,
        aggregate=0.25,
        p_value=0.01,
        max_psi=0.3,
        is_coupling=True,
    )
    assert summary.floor_multiple(0.05) == pytest.approx(5.0)


def test_the_regularisation_gap_is_relative_to_the_exact_optimum() -> None:
    row = RegRow(scale=0.1, reg=1.0, cost=110.0, iterations=50, to_centroid=1.0, to_partner=2.0)
    assert row.gap(100.0) == pytest.approx(0.1)
