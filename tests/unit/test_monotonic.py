"""Monotone constraints: the property must actually hold, and the attack must be a real one.

The report claims an entire evasion family is impossible by construction. That claim rests on
three things being true: the constraint vector marks the right features, the fitted model
genuinely never lowers its score when a constrained feature rises, and the attack used to
check it is strong enough that failing to evade means something. Each is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config import Settings
from netsentry.models.monotonic import (
    constraint_vector,
    inflation_attack,
    provably_inflation_robust,
    violates_monotonicity,
)
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.verify_trees import Tree

NAMES = ["Flow Duration", "Total Fwd Packets", "SYN Flag Count", "Init_Win_bytes_forward"]


# --------------------------------------------------------------------------------------
# The constraint vector
# --------------------------------------------------------------------------------------
def test_only_attacker_controllable_features_are_constrained() -> None:
    vector = constraint_vector(NAMES, ["Flow Duration", "Total Fwd Packets"])
    assert vector == [1, 1, 0, 0]


def test_the_constraint_is_non_decreasing_not_non_increasing() -> None:
    """+1, because the constrained score is the *attack* probability: padding must not help."""
    assert set(constraint_vector(NAMES, ["Flow Duration"])) <= {0, 1}


def test_an_unknown_controllable_name_constrains_nothing() -> None:
    assert constraint_vector(NAMES, ["Nonexistent Feature"]) == [0, 0, 0, 0]


def test_the_vector_has_one_entry_per_feature() -> None:
    assert len(constraint_vector(NAMES, NAMES)) == len(NAMES)


# --------------------------------------------------------------------------------------
# The property, end to end through a fitted model
# --------------------------------------------------------------------------------------
@pytest.fixture
def adversarial_data() -> tuple[np.ndarray, np.ndarray]:
    """A problem where the unconstrained fit is non-monotone in feature 0 by construction.

    The label is 1 in a *band* of feature 0, so raising it past the band lowers the attack
    score — exactly the structure a padding attacker exploits.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=(1200, 2))
    y = ((x[:, 0] > 0.0) & (x[:, 0] < 1.5)).astype(int)
    return x, y


def test_an_unconstrained_model_can_be_talked_out_of_an_alert_by_padding(
    settings: Settings, adversarial_data: tuple[np.ndarray, np.ndarray]
) -> None:
    x, y = adversarial_data
    settings.supervised.n_estimators = 60
    model = SupervisedClassifier(settings).fit(x, y)

    def score(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(matrix))[:, 1]

    alerting = x[(y == 1) & (score(x) > 0.5)][:50]
    attacked = inflation_attack(score, alerting, np.array([0]), np.array([0.5, 2.0]), rounds=2)
    assert float(np.mean(score(attacked) > 0.5)) < 0.5  # most alerts padded away


def test_a_constrained_model_cannot_be(
    settings: Settings, adversarial_data: tuple[np.ndarray, np.ndarray]
) -> None:
    """The same data, the same attack, the constraint switched on."""
    x, y = adversarial_data
    settings.supervised.n_estimators = 60
    model = SupervisedClassifier(settings, monotone_constraints=[1, 0]).fit(x, y)

    def score(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(matrix))[:, 1]

    alerting = x[(y == 1) & (score(x) > 0.5)][:50]
    if not len(alerting):
        pytest.skip("constrained model raised no alerts on this fixture")
    attacked = inflation_attack(score, alerting, np.array([0]), np.array([0.5, 2.0]), rounds=2)
    assert float(np.mean(score(attacked) > 0.5)) == 1.0  # every alert survives


def test_the_random_probe_refutes_the_unconstrained_model_and_not_the_constrained_one(
    settings: Settings, adversarial_data: tuple[np.ndarray, np.ndarray]
) -> None:
    x, y = adversarial_data
    settings.supervised.n_estimators = 60
    rng = np.random.default_rng(1)
    steps = np.array([0.5, 2.0])

    def scorer(constraints: list[int] | None) -> object:
        fitted = SupervisedClassifier(settings, monotone_constraints=constraints).fit(x, y)
        return lambda matrix: np.asarray(fitted.predict_proba(matrix))[:, 1]

    free = violates_monotonicity(scorer(None), x[:200], np.array([0]), steps, rng)
    held = violates_monotonicity(scorer([1, 0]), x[:200], np.array([0]), steps, rng)
    assert free.any()
    assert not held.any()


# --------------------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------------------
def _stump(feature: int, threshold: float, left: float, right: float) -> Tree:
    """A one-split tree: feature <= threshold goes left."""
    return Tree(
        feature=np.array([feature, -1, -1], dtype=np.int32),
        threshold=np.array([threshold, 0.0, 0.0]),
        left=np.array([1, -1, -1], dtype=np.int32),
        right=np.array([2, -1, -1], dtype=np.int32),
        value=np.array([0.0, left, right]),
    )


def test_a_rising_stump_is_provably_robust_to_unbounded_inflation() -> None:
    """Raising the feature only moves the flow to the higher leaf, so the score cannot fall."""
    trees = [_stump(0, 1.0, left=-2.0, right=3.0)]
    assert provably_inflation_robust(trees, np.array([2.0, 0.0]), np.array([0]), 0.0, 1e6)


def test_a_falling_stump_is_not() -> None:
    trees = [_stump(0, 1.0, left=3.0, right=-2.0)]
    assert not provably_inflation_robust(trees, np.array([0.5, 0.0]), np.array([0]), 0.0, 1e6)


def test_inflating_an_unconstrained_column_cannot_break_a_flow_that_ignores_it() -> None:
    """Only the listed columns are inflated; a tree splitting elsewhere is untouched."""
    trees = [_stump(1, 1.0, left=3.0, right=-2.0)]
    assert provably_inflation_robust(trees, np.array([0.0, 0.5]), np.array([0]), 0.0, 1e6)


def test_the_bound_is_evaluated_against_the_threshold_it_was_given() -> None:
    trees = [_stump(0, 1.0, left=2.0, right=5.0)]
    x = np.array([2.0])
    assert provably_inflation_robust(trees, x, np.array([0]), 4.0, 1e6)
    assert not provably_inflation_robust(trees, x, np.array([0]), 6.0, 1e6)


# --------------------------------------------------------------------------------------
# The attack
# --------------------------------------------------------------------------------------
def test_the_attack_never_decreases_a_feature() -> None:
    """An attacker can add bytes; it cannot un-send them. The search must respect that."""
    x = np.zeros((5, 3))

    def score(matrix: np.ndarray) -> np.ndarray:
        return -matrix.sum(axis=1)  # every addition lowers the score, so it always moves

    attacked = inflation_attack(score, x, np.array([0, 1, 2]), np.array([1.0]), rounds=3)
    assert np.all(attacked >= x)


def test_the_attack_stops_when_no_addition_helps() -> None:
    x = np.zeros((4, 2))

    def score(matrix: np.ndarray) -> np.ndarray:
        return matrix.sum(axis=1)  # every addition raises the score: nothing to do

    assert np.array_equal(inflation_attack(score, x, np.array([0, 1]), np.array([1.0]), 5), x)


def test_more_rounds_never_make_the_attacker_worse_off() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, size=(20, 2))

    def score(matrix: np.ndarray) -> np.ndarray:
        return np.cos(matrix[:, 0] * 3.0) + matrix[:, 1] * 0.1

    scores = [
        float(np.mean(score(inflation_attack(score, x, np.array([0]), np.array([0.3]), r))))
        for r in (0, 1, 2, 4)
    ]
    assert scores == sorted(scores, reverse=True)
