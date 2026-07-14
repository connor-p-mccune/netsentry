"""Model-extraction primitives: query-response defenses, surrogate fit, fidelity, and
the transfer-attack search — the pure pieces the report orchestrates."""

from __future__ import annotations

import numpy as np

from netsentry.robustness.extraction import (
    LABEL_ONLY,
    PROBABILITIES,
    ROUNDED,
    answered_query,
    fidelity,
    search_delta,
    train_surrogate,
)


def test_answered_query_probabilities_is_identity() -> None:
    scores = np.array([0.13, 0.87, 0.5])
    assert np.allclose(answered_query(scores, PROBABILITIES, round_decimals=1), scores)


def test_answered_query_rounding_reduces_precision() -> None:
    scores = np.array([0.137, 0.849])
    rounded = answered_query(scores, ROUNDED, round_decimals=1)
    assert np.allclose(rounded, [0.1, 0.8])


def test_answered_query_label_only_is_a_hard_decision() -> None:
    scores = np.array([0.2, 0.5, 0.9, 0.49])
    labels = answered_query(scores, LABEL_ONLY, round_decimals=1)
    assert np.allclose(labels, [0.0, 1.0, 1.0, 0.0])  # >= 0.5 is the attack side


def test_fidelity_is_decision_agreement_not_truth() -> None:
    victim = np.array([0.9, 0.1, 0.8, 0.2])
    # Surrogate agrees on the argmax cut for every row, though the scores differ.
    surrogate = np.array([0.55, 0.45, 0.51, 0.0])
    assert fidelity(victim, surrogate) == 1.0
    # One disagreement (row 1: victim benign, surrogate attack) -> 3/4.
    surrogate_flip = np.array([0.9, 0.9, 0.8, 0.2])
    assert fidelity(victim, surrogate_flip) == 0.75


def test_fidelity_empty_is_zero() -> None:
    assert fidelity(np.array([]), np.array([])) == 0.0


def test_train_surrogate_recovers_a_linear_boundary_from_soft_labels() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 4))
    # A soft victim: sigmoid of the first coordinate.
    victim = 1.0 / (1.0 + np.exp(-3.0 * x[:, 0]))
    surrogate = train_surrogate(x, victim, PROBABILITIES, seed=0)
    pred = surrogate.score(x)
    # The stolen scorer ranks rows the same way the victim does.
    assert np.corrcoef(pred, victim)[0, 1] > 0.9
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_train_surrogate_label_only_degenerate_pool_returns_constant() -> None:
    x = np.zeros((10, 3))
    answers = np.zeros(10)  # all one class -> nothing to separate
    surrogate = train_surrogate(x, answers, LABEL_ONLY, seed=0)
    assert surrogate.estimator is None
    assert np.allclose(surrogate.score(x), 0.0)


def test_search_delta_finds_a_score_reducing_perturbation() -> None:
    # Score is +x on one controllable feature; the search should push it negative.
    def score_fn(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    x = np.ones((5, 3))
    ctrl = np.array([0])
    rng = np.random.default_rng(1)
    delta = search_delta(score_fn, x, ctrl, eps=1.0, iterations=50, rng=rng)
    assert np.all(score_fn(x + delta) < score_fn(x))
    # The perturbation only touches controllable columns and respects the L2 ball.
    assert np.allclose(delta[:, 1:], 0.0)
    assert np.all(np.linalg.norm(delta, axis=1) <= 1.0 + 1e-9)


def test_search_delta_zero_budget_is_a_no_op() -> None:
    def score_fn(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    x = np.ones((3, 2))
    rng = np.random.default_rng(0)
    delta = search_delta(score_fn, x, np.array([0]), eps=0.0, iterations=10, rng=rng)
    assert np.allclose(delta, 0.0)
