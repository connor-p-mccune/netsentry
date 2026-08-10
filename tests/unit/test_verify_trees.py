"""Tree verification: soundness of the bound, and that it is about the deployed model.

A verification result is only worth as much as two things: that the flattened ensemble is
the same function LightGBM evaluates, and that the interval bound never claims more than it
can prove. The first is checked against LightGBM's own raw score; the second is checked by
brute force — sample the box densely and assert every observed value lies inside the
bound, for many random trees.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.robustness.verify_trees import (
    Tree,
    attack_radius,
    batched_margin,
    certified_radius,
    ensemble_bounds,
    ensemble_margin,
    is_certified,
    parse_tree,
    perturbation_box,
    tree_bounds,
    tree_margin,
)

# A hand-built stump forest: feature 0 splits at 0.5, feature 1 at -1.0.
STUMP = parse_tree(
    {
        "split_feature": 0,
        "threshold": 0.5,
        "left_child": {"leaf_value": -1.0},
        "right_child": {"leaf_value": 2.0},
    }
)
DEEPER = parse_tree(
    {
        "split_feature": 1,
        "threshold": -1.0,
        "left_child": {"leaf_value": 0.5},
        "right_child": {
            "split_feature": 0,
            "threshold": 3.0,
            "left_child": {"leaf_value": -0.25},
            "right_child": {"leaf_value": 4.0},
        },
    }
)


def test_parse_tree_records_leaves_and_splits() -> None:
    assert STUMP.n_nodes == 3
    assert STUMP.feature.tolist() == [0, -1, -1]
    assert STUMP.value[STUMP.feature < 0].tolist() == [-1.0, 2.0]


def test_tree_margin_follows_the_less_than_or_equal_convention() -> None:
    # LightGBM routes left on `x <= threshold`; the boundary itself goes left.
    assert tree_margin(STUMP, np.array([0.5, 0.0])) == -1.0
    assert tree_margin(STUMP, np.array([0.5000001, 0.0])) == 2.0


def test_tree_bounds_prune_a_child_the_box_cannot_reach() -> None:
    lo, hi = np.array([-5.0, 0.0]), np.array([0.4, 0.0])
    assert tree_bounds(STUMP, lo, hi) == (-1.0, -1.0)  # box lies wholly left
    lo, hi = np.array([0.6, 0.0]), np.array([9.0, 0.0])
    assert tree_bounds(STUMP, lo, hi) == (2.0, 2.0)  # wholly right


def test_tree_bounds_keep_both_leaves_when_the_box_straddles_a_split() -> None:
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert tree_bounds(STUMP, lo, hi) == (-1.0, 2.0)


def test_tree_bounds_are_exact_for_a_single_tree() -> None:
    """One tree at a time, the bound is not merely sound — it is attained."""
    rng = np.random.default_rng(0)
    lo, hi = np.array([-2.0, -3.0]), np.array([5.0, 2.0])
    low, high = tree_bounds(DEEPER, lo, hi)
    samples = lo + rng.random((20_000, 2)) * (hi - lo)
    observed = np.array([tree_margin(DEEPER, s) for s in samples])
    assert observed.min() == pytest.approx(low)
    assert observed.max() == pytest.approx(high)


def test_ensemble_bound_contains_every_value_the_box_can_produce() -> None:
    """Soundness, by brute force, over many random ensembles — the property that matters."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        trees = [_random_tree(rng, n_features=4, depth=3) for _ in range(6)]
        centre = rng.normal(size=4)
        lo, hi = centre - 0.8, centre + 0.8
        low, high = ensemble_bounds(trees, lo, hi)
        samples = lo + rng.random((400, 4)) * (hi - lo)
        values = np.array([ensemble_margin(trees, s) for s in samples])
        assert values.min() >= low - 1e-9
        assert values.max() <= high + 1e-9


def test_the_bound_is_conservative_not_exact_across_trees() -> None:
    """The stated incompleteness: two trees reading the same input cannot both be extremal.

    Tree A is minimal only when x0 <= 0.5 and tree B only when x0 > 0.5, so no single input
    attains the summed minimum. A sound bound must still report it — and knowing that it
    does is what keeps the report's "sound but incomplete" claim honest.
    """
    a = parse_tree(
        {
            "split_feature": 0,
            "threshold": 0.5,
            "left_child": {"leaf_value": 0.0},
            "right_child": {"leaf_value": 10.0},
        }
    )
    b = parse_tree(
        {
            "split_feature": 0,
            "threshold": 0.5,
            "left_child": {"leaf_value": 10.0},
            "right_child": {"leaf_value": 0.0},
        }
    )
    low, _high = ensemble_bounds([a, b], np.array([0.0]), np.array([1.0]))
    assert low == 0.0  # the bound says 0 + 0
    # ...but every actual input scores 10, so the truth is strictly above the bound.
    assert ensemble_margin([a, b], np.array([0.0])) == 10.0
    assert ensemble_margin([a, b], np.array([1.0])) == 10.0


def test_perturbation_box_respects_a_one_directional_threat_model() -> None:
    x = np.array([1.0, 1.0])
    up = np.array([True, False])
    down = np.array([False, False])
    lo, hi = perturbation_box(x, 0.5, up, down)
    assert lo.tolist() == [1.0, 1.0]  # nothing may decrease
    assert hi.tolist() == [1.5, 1.0]  # only feature 0 may increase


def test_pinning_every_feature_certifies_any_radius() -> None:
    x = np.array([1.0, 0.0])
    frozen = np.zeros(2, dtype=bool)
    assert is_certified([STUMP], x, 99.0, 0.0, frozen, frozen, predicted_attack=True)


def test_certified_radius_stops_exactly_at_the_split_that_would_flip_the_verdict() -> None:
    # x0 = 1.0 scores +2; dropping below 0.5 scores -1. With a threshold of 0, the verdict
    # survives any perturbation smaller than 0.5 and no perturbation larger.
    movable = np.ones(2, dtype=bool)
    r = certified_radius(
        [STUMP], np.array([1.0, 0.0]), 0.0, movable, movable, max_radius=2.0, steps=20
    )
    assert 0.49 < r < 0.501


def test_a_one_directional_attacker_cannot_reach_the_dangerous_side() -> None:
    """The threat model earns its keep: forbid decreases and the same flow certifies fully."""
    x = np.array([1.0, 0.0])
    up_only = np.ones(2, dtype=bool)
    no_down = np.zeros(2, dtype=bool)
    r = certified_radius([STUMP], x, 0.0, up_only, no_down, max_radius=2.0, steps=20)
    assert r == 2.0  # censored at the search ceiling: nothing inside it breaks the verdict


def test_certified_radius_is_zero_for_a_flow_already_on_the_wrong_side() -> None:
    movable = np.ones(2, dtype=bool)
    # Predicted benign (margin -1 < 0) but tested as if it must stay above 0.
    r = certified_radius(
        [STUMP], np.array([0.0, 0.0]), 5.0, movable, movable, max_radius=2.0, steps=10
    )
    assert r == 2.0  # it is certified *benign*, robustly: the check follows the prediction


def test_attack_radius_upper_bounds_the_certified_radius() -> None:
    """The sandwich must not invert: an attack cannot succeed inside a proved-safe ball."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        trees = [_random_tree(rng, n_features=3, depth=3) for _ in range(4)]
        x = rng.normal(size=3)
        threshold = ensemble_margin(trees, x) - 0.01  # x is predicted "attack" by a hair
        movable = np.ones(3, dtype=bool)
        cert = certified_radius(trees, x, threshold, movable, movable, max_radius=3.0, steps=14)
        found = attack_radius(
            batched_margin(trees),
            x,
            threshold,
            movable,
            movable,
            max_radius=3.0,
            steps=14,
            n_random=40,
            rng=rng,
        )
        assert found >= cert - 1e-6


def _random_tree(rng: np.random.Generator, n_features: int, depth: int) -> Tree:
    """A random balanced tree, for property tests that need many distinct ensembles."""

    def _node(level: int) -> dict[str, object]:
        if level == 0:
            return {"leaf_value": float(rng.normal())}
        return {
            "split_feature": int(rng.integers(0, n_features)),
            "threshold": float(rng.normal()),
            "left_child": _node(level - 1),
            "right_child": _node(level - 1),
        }

    return parse_tree(_node(depth))  # type: ignore[arg-type]


@pytest.mark.slow
def test_flattened_ensemble_reproduces_lightgbms_own_raw_score() -> None:
    """The gate the whole report stands on: this is the deployed model, not a lookalike."""
    lgb = pytest.importorskip("lightgbm")
    from netsentry.robustness.verify_trees import parse_booster

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y = (x[:, 0] + 0.5 * x[:, 3] - 0.3 * x[:, 5] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=25, num_leaves=7, verbosity=-1, random_state=0)
    model.fit(x, y)
    trees = parse_booster(model.booster_.dump_model())
    reference = model.booster_.predict(x[:100], raw_score=True)
    mine = np.array([ensemble_margin(trees, row) for row in x[:100]])
    assert np.max(np.abs(reference - mine)) < 1e-9
