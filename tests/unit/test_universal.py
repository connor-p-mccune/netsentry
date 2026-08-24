"""The universal-perturbation fitter, and the property the defence rests on.

The interesting claims are structural rather than empirical, so they are tested that way: a
projection that respects its budget, a descent that only ever moves allowed coordinates, an
additive-only mode that never emits a negative entry, and -- the one the report's defence
section depends on -- that a non-decreasing score cannot be lowered by a non-negative shift.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.robustness.universal import (
    BudgetRow,
    UniversalStudy,
    centroid_direction,
    fit_universal,
    project,
)


def _linear(weights: np.ndarray):  # type: ignore[no-untyped-def]
    """A score that falls when the weighted features fall -- attackable by construction."""

    def score(rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows, dtype=float) @ weights

    return score


# --------------------------------------------------------------------------------------
# The budget.
# --------------------------------------------------------------------------------------


def test_projection_leaves_a_vector_inside_the_ball_alone() -> None:
    vector = np.array([0.3, 0.4])  # norm 0.5
    assert np.allclose(project(vector, 1.0), vector)


def test_projection_scales_a_vector_onto_the_ball() -> None:
    projected = project(np.array([3.0, 4.0]), 1.0)
    assert float(np.linalg.norm(projected)) == pytest.approx(1.0)
    assert np.allclose(projected, np.array([0.6, 0.8]))


def test_projection_survives_a_zero_vector() -> None:
    assert np.allclose(project(np.zeros(4), 2.0), np.zeros(4))


# --------------------------------------------------------------------------------------
# The fitter.
# --------------------------------------------------------------------------------------


def test_the_fitted_vector_lowers_the_batch_score() -> None:
    rng = np.random.default_rng(0)
    flows = rng.normal(size=(64, 4)) + 3.0
    score = _linear(np.array([1.0, 1.0, 0.0, 0.0]))
    vector = fit_universal(
        score, flows, np.array([0, 1, 2, 3]), budget=2.0, steps=20, step_size=0.25
    )
    assert float(np.mean(score(flows + vector))) < float(np.mean(score(flows)))


def test_the_fitter_never_moves_a_coordinate_it_was_not_given() -> None:
    """The threat model is a coordinate set; a fitter that ignores it measures nothing."""
    rng = np.random.default_rng(1)
    flows = rng.normal(size=(32, 5)) + 2.0
    vector = fit_universal(
        _linear(np.ones(5)), flows, np.array([0, 2]), budget=3.0, steps=15, step_size=0.5
    )
    assert vector[1] == 0.0 and vector[3] == 0.0 and vector[4] == 0.0
    assert vector[0] != 0.0 or vector[2] != 0.0


def test_the_fitted_vector_respects_its_budget() -> None:
    rng = np.random.default_rng(2)
    flows = rng.normal(size=(32, 3)) + 5.0
    vector = fit_universal(
        _linear(np.ones(3)), flows, np.arange(3), budget=1.0, steps=40, step_size=0.5
    )
    assert float(np.linalg.norm(vector)) <= 1.0 + 1e-9


def test_the_additive_only_mode_never_emits_a_negative_shift() -> None:
    """Padding adds; it does not subtract. The defence is only meaningful against this mode."""
    rng = np.random.default_rng(3)
    flows = rng.normal(size=(32, 4)) + 2.0
    vector = fit_universal(
        _linear(np.array([1.0, -1.0, 1.0, -1.0])),
        flows,
        np.arange(4),
        budget=2.0,
        steps=20,
        step_size=0.25,
        non_negative=True,
    )
    assert np.all(vector >= 0.0)


def test_a_non_decreasing_score_cannot_be_lowered_by_an_additive_vector() -> None:
    """The defence, as a property rather than as a measurement.

    A model non-decreasing in every attacked coordinate cannot be pushed down by a non-negative
    shift, so the fitter must return the zero vector -- there is no allowed step that helps.
    """
    rng = np.random.default_rng(4)
    flows = rng.normal(size=(48, 3))
    monotone = _linear(np.array([1.0, 2.0, 0.5]))  # increasing in every coordinate
    vector = fit_universal(
        monotone, flows, np.arange(3), budget=4.0, steps=25, step_size=0.5, non_negative=True
    )
    assert np.allclose(vector, 0.0)
    assert float(np.mean(monotone(flows + vector))) == pytest.approx(
        float(np.mean(monotone(flows)))
    )


def test_the_fitter_stops_when_no_step_helps() -> None:
    """A local optimum has to terminate the loop, not consume the whole step budget."""
    flows = np.zeros((8, 2))
    vector = fit_universal(
        _linear(np.zeros(2)), flows, np.arange(2), budget=5.0, steps=50, step_size=1.0
    )
    assert np.allclose(vector, 0.0)


def test_the_centroid_direction_points_from_the_attacks_to_the_benign_mean() -> None:
    attacks = np.array([[0.0, 0.0], [2.0, 2.0]])
    benign = np.array([[10.0, 20.0], [10.0, 20.0]])
    direction = centroid_direction(attacks, benign, np.array([0]))
    assert direction[0] == pytest.approx(9.0)
    assert direction[1] == 0.0  # coordinate 1 is outside the threat model


# --------------------------------------------------------------------------------------
# The records the report reads from.
# --------------------------------------------------------------------------------------


def _study(rows: list[BudgetRow]) -> UniversalStudy:
    return UniversalStudy(
        budgets=rows,
        transfers=[],
        defences=[],
        recipe=[],
        baseline_detection=0.2,
        baseline_psi=0.5,
        n_fit=10,
        n_held_out=10,
        n_controllable=2,
        n_features=4,
        profile="fpr_1pct",
        headline_budget=2.0,
        queries_universal=1,
        queries_per_flow=1,
    )


def test_the_sweep_preserves_order_and_deduplicates() -> None:
    rows = [
        BudgetRow("a", 1.0, 0.1, 0.1, 1.0),
        BudgetRow("b", 1.0, 0.2, 0.2, 2.0),
        BudgetRow("a", 2.0, 0.05, 0.05, 3.0),
    ]
    assert _study(rows).sweep() == [1.0, 2.0]


def test_a_curve_selects_one_direction_across_the_sweep() -> None:
    rows = [
        BudgetRow("a", 1.0, 0.1, 0.1, 1.0),
        BudgetRow("b", 1.0, 0.9, 0.9, 2.0),
        BudgetRow("a", 2.0, 0.05, 0.05, 3.0),
    ]
    assert _study(rows).curve("a", "detection").tolist() == [0.1, 0.05]


def test_a_missing_cell_returns_none_rather_than_raising() -> None:
    assert _study([BudgetRow("a", 1.0, 0.1, 0.1, 1.0)]).at("b", 1.0) is None
