"""The Shapley machinery, checked against the axioms it is supposed to satisfy.

The report's sharpest claim -- that a conditional attribution must split credit evenly between
two identical features while an interventional one must give the unused copy zero -- is a
consequence of the axioms rather than an empirical observation. So the axioms get tested
directly: efficiency, symmetry, and the null-player property, on games small enough to solve by
hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.explain.shap_estimand import (
    _split_counts,
    coalition_values,
    conditional_shapley,
    exact_shapley,
    rank_agreement,
    shapley_from_values,
    top_features,
    top_overlap,
)

# --------------------------------------------------------------------------------------
# The Shapley formula.
# --------------------------------------------------------------------------------------


def test_the_formula_matches_a_two_player_game_solved_by_hand() -> None:
    """phi_1 = a/2 + (c - b)/2 for v(1)=a, v(2)=b, v(1,2)=c -- the textbook closed form."""
    a, b, c = 3.0, 5.0, 11.0
    values = {0b00: 0.0, 0b01: a, 0b10: b, 0b11: c}
    phi = shapley_from_values(values, 2)
    assert phi[0] == pytest.approx(a / 2 + (c - b) / 2)
    assert phi[1] == pytest.approx(b / 2 + (c - a) / 2)
    assert phi.sum() == pytest.approx(c)  # efficiency


def test_a_null_player_receives_nothing() -> None:
    """A feature that changes no coalition's value must get exactly zero."""
    values = {0b00: 1.0, 0b01: 4.0, 0b10: 1.0, 0b11: 4.0}
    phi = shapley_from_values(values, 2)
    assert phi[1] == pytest.approx(0.0)
    assert phi[0] == pytest.approx(3.0)


def test_symmetric_players_receive_the_same_share() -> None:
    """The axiom the duplicate experiment's prediction rests on."""
    values = {0b00: 0.0, 0b01: 2.0, 0b10: 2.0, 0b11: 6.0}
    phi = shapley_from_values(values, 2)
    assert phi[0] == pytest.approx(phi[1])
    assert phi.sum() == pytest.approx(6.0)


# --------------------------------------------------------------------------------------
# The interventional value function.
# --------------------------------------------------------------------------------------


def _linear(weights: np.ndarray):  # type: ignore[no-untyped-def]
    def score(rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows, dtype=float) @ weights

    return score


def test_the_coalition_value_holds_the_flow_and_samples_the_rest() -> None:
    background = np.array([[0.0, 0.0], [2.0, 4.0]])
    x = np.array([10.0, 100.0])
    values = coalition_values(_linear(np.array([1.0, 1.0])), x, background)
    assert values[0b00] == pytest.approx(3.0)  # both from the background: mean(0+0, 2+4)
    assert values[0b01] == pytest.approx(10.0 + 2.0)  # first held, second sampled
    assert values[0b11] == pytest.approx(110.0)  # both held: the flow's own score


def test_interventional_shapley_of_a_linear_model_is_the_weighted_deviation() -> None:
    """For an additive model the attribution has a closed form: w_j * (x_j - E[X_j])."""
    rng = np.random.default_rng(0)
    background = rng.normal(size=(64, 4))
    weights = np.array([2.0, -1.0, 0.5, 0.0])
    x = rng.normal(size=4)
    phi = exact_shapley(_linear(weights), x, background)
    expected = weights * (x - background.mean(axis=0))
    assert np.allclose(phi, expected, atol=1e-9)


def test_a_feature_the_model_ignores_gets_exactly_zero_interventionally() -> None:
    rng = np.random.default_rng(1)
    background = rng.normal(size=(32, 3))
    phi = exact_shapley(_linear(np.array([1.0, 0.0, 1.0])), rng.normal(size=3), background)
    assert phi[1] == pytest.approx(0.0, abs=1e-12)


def test_interventional_attribution_is_efficient() -> None:
    rng = np.random.default_rng(2)
    background = rng.normal(size=(48, 4))
    weights = np.array([1.5, -2.0, 0.25, 3.0])
    x = rng.normal(size=4)
    score = _linear(weights)
    phi = exact_shapley(score, x, background)
    baseline = float(np.mean(score(background)))
    assert phi.sum() + baseline == pytest.approx(float(score(x[None, :])[0]))


# --------------------------------------------------------------------------------------
# The conditional value function, and the claim the report makes about it.
# --------------------------------------------------------------------------------------


def test_the_conditional_attribution_splits_credit_between_identical_features() -> None:
    """The report's provable claim, checked numerically on a model that ignores the copy.

    The scorer reads only the first column. Interventionally the copy is a null player and gets
    zero; conditionally the two are exchangeable -- knowing either determines the other -- so
    symmetry forces an even split. Both halves of that are asserted here.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=256)
    background = np.column_stack([base, base, rng.normal(size=256)])
    x = np.array([1.4, 1.4, 0.3])
    score = _linear(np.array([1.0, 0.0, 0.0]))
    interventional = exact_shapley(score, x, background)
    conditional = conditional_shapley(score, x, background, neighbours=8)
    assert interventional[1] == pytest.approx(0.0, abs=1e-12)
    assert conditional[0] == pytest.approx(conditional[1], rel=0.05)
    assert conditional[0] + conditional[1] == pytest.approx(interventional[0], rel=0.15)


def test_the_conditional_value_of_the_empty_coalition_is_the_unconditional_mean() -> None:
    rng = np.random.default_rng(4)
    background = rng.normal(size=(64, 3))
    score = _linear(np.array([1.0, 1.0, 1.0]))
    from netsentry.explain.shap_estimand import conditional_values

    values = conditional_values(score, np.zeros(3), background, neighbours=4)
    assert values[0b000] == pytest.approx(float(np.mean(score(background))))


def test_the_conditional_attribution_is_efficient_when_every_feature_is_held() -> None:
    """Holding all features must reproduce the flow's own score, whatever k is."""
    rng = np.random.default_rng(5)
    background = rng.normal(size=(64, 3))
    from netsentry.explain.shap_estimand import conditional_values

    x = background[7]
    values = conditional_values(_linear(np.array([1.0, 2.0, 3.0])), x, background, neighbours=1)
    assert values[0b111] == pytest.approx(float(x @ np.array([1.0, 2.0, 3.0])))


# --------------------------------------------------------------------------------------
# What the API actually returns.
# --------------------------------------------------------------------------------------


def test_the_top_list_ranks_by_magnitude_not_by_sign() -> None:
    values = np.array([0.1, -0.9, 0.4])
    assert top_features(values, 2) == [1, 2]


def test_overlap_and_rank_agreement_behave_at_the_extremes() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0])
    assert top_overlap(left, left, 2) == pytest.approx(1.0)
    assert top_overlap(left, -left[::-1], 1) == pytest.approx(0.0)
    assert rank_agreement(left, left) == pytest.approx(1.0)
    assert rank_agreement(left, -left) == pytest.approx(-1.0)


def test_rank_agreement_is_defined_for_a_constant_attribution() -> None:
    """A flow whose contributions are all equal must not produce a nan in a report table."""
    assert rank_agreement(np.zeros(5), np.arange(5.0)) == 0.0


# --------------------------------------------------------------------------------------
# The ground truth the duplicate experiment rests on.
# --------------------------------------------------------------------------------------


class _Booster:
    def __init__(self, dump: dict[str, object]) -> None:
        self._dump = dump

    def dump_model(self) -> dict[str, object]:
        return self._dump


def test_split_counts_reads_the_dumped_model() -> None:
    """'The model never splits on this feature' has to be verified, not assumed."""
    tree = {
        "tree_structure": {
            "split_feature": 0,
            "left_child": {"leaf_value": 1.0},
            "right_child": {
                "split_feature": 2,
                "left_child": {"leaf_value": 0.0},
                "right_child": {"leaf_value": 2.0},
            },
        }
    }
    counts = _split_counts(_Booster({"tree_info": [tree, tree]}), 4)
    assert counts.tolist() == [2, 0, 2, 0]
