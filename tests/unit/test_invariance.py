"""Invariance: an all-benign day must not screen anything, and the penalty must mean something.

The report's headline is that 42% of features point in opposite directions on different
capture days. That number is only meaningful if a day containing one class is excluded rather
than scored as zero, and if IRM's penalty is actually the quantity Arjovsky et al. define. Both
are pinned here against hand-constructed cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.training.invariance import (
    FeatureStability,
    environment_strength,
    feature_stability,
    fit_irm,
    informative_environments,
    irm_penalty,
    screen_breakdown,
    screen_invariant,
)


# --------------------------------------------------------------------------------------
# Environment strength
# --------------------------------------------------------------------------------------
def test_a_perfectly_separating_feature_has_maximum_strength() -> None:
    y = np.array([0, 0, 1, 1])
    assert environment_strength(np.array([1.0, 2.0, 3.0, 4.0]), y) == pytest.approx(0.5)


def test_the_sign_says_which_way_the_feature_points() -> None:
    y = np.array([0, 0, 1, 1])
    rising = environment_strength(np.array([1.0, 2.0, 3.0, 4.0]), y)
    falling = environment_strength(np.array([4.0, 3.0, 2.0, 1.0]), y)
    assert rising > 0 > falling
    assert rising == pytest.approx(-falling)


def test_a_single_class_environment_has_no_defined_relationship() -> None:
    assert environment_strength(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])) == 0.0


# --------------------------------------------------------------------------------------
# Which environments can screen anything
# --------------------------------------------------------------------------------------
def test_an_all_benign_day_is_dropped_from_the_screen() -> None:
    """CIC-IDS2017's Monday, and the bug it would silently cause if kept."""
    y = np.array([0, 0, 0, 0, 1, 0, 1])
    envs = np.array(["Mon", "Mon", "Mon", "Tue", "Tue", "Wed", "Wed"])
    assert informative_environments(y, envs) == ["Tue", "Wed"]


def test_dropping_a_single_class_day_changes_the_verdict() -> None:
    """Keeping it would push every feature's dispersion up and reject the whole vector."""
    x = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [1.0], [2.0]])
    y = np.array([0, 0, 0, 0, 1, 0, 1])
    envs = np.array(["Mon", "Mon", "Mon", "Tue", "Tue", "Wed", "Wed"])
    [stability] = feature_stability(x, y, envs, ["f"])
    assert len(stability.strengths) == 2  # Monday contributed nothing
    assert stability.sign_agreement


def test_every_environment_being_single_class_leaves_nothing_to_screen() -> None:
    y = np.array([0, 0, 1, 1])
    envs = np.array(["a", "a", "b", "b"])
    assert informative_environments(y, envs) == []


# --------------------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------------------
def test_a_feature_that_flips_direction_fails_sign_agreement() -> None:
    x = np.array([[1.0], [2.0], [2.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    envs = np.array(["a", "a", "b", "b"])
    [stability] = feature_stability(x, y, envs, ["f"])
    assert not stability.sign_agreement
    assert screen_invariant([stability], 0.0, np.inf) == []


def test_a_consistent_strong_feature_passes() -> None:
    x = np.array([[1.0], [9.0], [1.0], [9.0]])
    y = np.array([0, 1, 0, 1])
    envs = np.array(["a", "a", "b", "b"])
    [stability] = feature_stability(x, y, envs, ["f"])
    assert screen_invariant([stability], 0.1, 0.75) == ["f"]


def test_a_signal_free_feature_is_rejected_despite_trivially_agreeing_with_itself() -> None:
    weak = FeatureStability("noise", "in_flight", [0.0, 0.0], 0.0, True, 0.0)
    assert screen_invariant([weak], 0.02, 0.75) == []


def test_a_feature_whose_strength_swings_is_rejected() -> None:
    swingy = FeatureStability("swingy", "in_flight", [0.45, 0.02], 0.235, True, 0.91)
    assert screen_invariant([swingy], 0.02, 0.75) == []
    assert screen_invariant([swingy], 0.02, 0.95) == ["swingy"]


def test_the_breakdown_partitions_every_feature_exactly_once() -> None:
    stability = [
        FeatureStability("passes", "in_flight", [0.3, 0.3], 0.3, True, 0.0),
        FeatureStability("flips", "in_flight", [0.3, -0.3], 0.3, False, 0.0),
        FeatureStability("weak", "complete", [0.001, 0.001], 0.001, True, 0.0),
        FeatureStability("swingy", "complete", [0.4, 0.02], 0.21, True, 0.9),
    ]
    counts = screen_breakdown(stability, 0.02, 0.75)
    assert sum(counts.values()) == len(stability)
    assert counts == {"passed": 1, "flipped sign": 1, "too weak": 1, "unstable magnitude": 1}


# --------------------------------------------------------------------------------------
# IRM
# --------------------------------------------------------------------------------------
def test_the_penalty_vanishes_when_the_classifier_is_optimal_for_the_environment() -> None:
    """A logistic model at its own optimum has zero gradient at the dummy scale."""
    logits = np.array([-4.0, 4.0])
    y = 1.0 / (1.0 + np.exp(-logits))  # labels equal to the model's own probabilities
    assert irm_penalty(logits, y) == pytest.approx(0.0, abs=1e-12)


def test_the_penalty_is_positive_when_an_environment_wants_a_different_scale() -> None:
    logits = np.array([-4.0, 4.0])
    assert irm_penalty(logits, np.array([0.0, 1.0])) > 0.0


def test_the_penalty_is_never_negative() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.normal(0, 3, size=50)
        assert irm_penalty(z, rng.integers(0, 2, size=50).astype(float)) >= 0.0


def test_erm_learns_a_separable_problem() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] > 0).astype(float)
    envs = np.array(["a"] * 200 + ["b"] * 200)
    head = fit_irm(x, y, envs, penalty_weight=0.0, steps=400, learning_rate=0.5, l2=0.0, seed=0)
    assert float(np.mean((head.predict_proba(x) >= 0.5) == y)) > 0.9


def test_the_penalty_term_actually_reduces_the_penalty() -> None:
    """If turning the penalty on does not lower it, the sweep measures nothing."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 3))
    # Environment b has a feature that predicts the label only there: the spurious case.
    y = (x[:, 0] > 0).astype(float)
    envs = np.array(["a"] * 200 + ["b"] * 200)
    x[200:, 2] = np.where(y[200:] > 0, 3.0, -3.0)
    kwargs = {"steps": 300, "learning_rate": 0.3, "l2": 0.0, "seed": 0}
    erm = fit_irm(x, y, envs, penalty_weight=0.0, **kwargs)
    irm = fit_irm(x, y, envs, penalty_weight=100.0, **kwargs)

    def total(head: object) -> float:
        return sum(
            irm_penalty(head.logits(x[i]), y[i])  # type: ignore[attr-defined]
            for i in (slice(0, 200), slice(200, 400))
        )

    assert total(irm) < total(erm)


def test_training_is_deterministic_under_a_fixed_seed() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 2))
    y = (x[:, 0] > 0).astype(float)
    envs = np.array(["a"] * 100 + ["b"] * 100)
    kwargs = {"penalty_weight": 1.0, "steps": 50, "learning_rate": 0.2, "l2": 0.0, "seed": 7}
    assert np.allclose(fit_irm(x, y, envs, **kwargs).weights, fit_irm(x, y, envs, **kwargs).weights)
