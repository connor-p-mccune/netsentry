"""Uncertainty decomposition: the two terms must be distinguishable, not just computable.

The whole report rests on aleatoric and epistemic uncertainty measuring genuinely different
things. These pin the two extreme cases that separate them — members that individually
hedge (all aleatoric) versus members that are each certain and disagree (all epistemic) —
plus the non-negativity that Jensen's inequality guarantees and a bug would break.
"""

from __future__ import annotations

import math

import numpy as np

from netsentry.evaluation.uncertainty import binary_entropy, decompose, risk_coverage


def test_binary_entropy_is_zero_at_certainty_and_maximal_at_a_coin_flip() -> None:
    assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-9
    assert math.isclose(float(binary_entropy(np.array([0.5]))[0]), math.log(2), rel_tol=1e-9)


def test_unanimous_confident_members_leave_no_uncertainty_at_all() -> None:
    members = np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]])
    total, aleatoric, epistemic = decompose(members)
    assert np.all(total < 0.06) and np.all(epistemic < 1e-9)


def test_members_that_each_hedge_produce_aleatoric_uncertainty_only() -> None:
    """Everyone says 'coin flip': the data is ambiguous, and no amount of it would help."""
    members = np.full((5, 3), 0.5)
    total, aleatoric, epistemic = decompose(members)
    assert np.allclose(total, math.log(2))
    assert np.allclose(aleatoric, math.log(2))
    assert np.allclose(epistemic, 0.0, atol=1e-9)


def test_confident_members_that_disagree_produce_epistemic_uncertainty_only() -> None:
    """The discriminating case: each member is sure, and they are sure of opposite things."""
    members = np.array([[1.0, 0.0], [0.0, 1.0]])
    total, aleatoric, epistemic = decompose(members)
    assert np.allclose(total, math.log(2), atol=1e-6)
    assert np.allclose(aleatoric, 0.0, atol=1e-6)
    assert np.allclose(epistemic, math.log(2), atol=1e-6)


def test_epistemic_uncertainty_is_never_negative() -> None:
    """Jensen's inequality: entropy is concave, so H(mean) >= mean(H). A bug breaks this."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        members = rng.random((rng.integers(2, 12), 200))
        _total, _aleatoric, epistemic = decompose(members)
        assert np.all(epistemic >= 0.0)


def test_the_three_terms_add_up_exactly() -> None:
    rng = np.random.default_rng(1)
    members = rng.random((7, 500))
    total, aleatoric, epistemic = decompose(members)
    assert np.allclose(total, aleatoric + epistemic)


def test_a_single_member_ensemble_has_no_epistemic_uncertainty_by_construction() -> None:
    members = np.array([[0.3, 0.7, 0.9]])
    total, aleatoric, epistemic = decompose(members)
    assert np.allclose(epistemic, 0.0)
    assert np.allclose(total, aleatoric)


def test_decomposition_is_invariant_to_the_order_of_the_members() -> None:
    rng = np.random.default_rng(2)
    members = rng.random((6, 50))
    a = decompose(members)
    b = decompose(members[::-1])
    assert all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


def test_risk_coverage_drops_the_error_rate_when_the_signal_ranks_mistakes() -> None:
    # Scores: the first two flows are wrong; the abstention signal flags exactly those.
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    signal = np.array([1.0, 1.0, 0.0, 0.0])  # high = uncertain
    cov, err = risk_coverage(scores, signal, y, 0.5, np.array([0.5, 1.0]))
    assert err[0] == 0.0  # keeping the confident half removes both mistakes
    assert err[1] == 0.5  # keeping everything restores them
    assert cov.tolist() == [0.5, 1.0]


def test_risk_coverage_is_flat_when_the_signal_is_uninformative() -> None:
    rng = np.random.default_rng(3)
    scores = rng.random(400)
    y = rng.integers(0, 2, 400)
    noise = rng.random(400)  # unrelated to whether the model is right
    _cov, err = risk_coverage(scores, noise, y, 0.5, np.array([0.5, 1.0]))
    assert abs(err[0] - err[1]) < 0.1


def test_risk_coverage_at_full_coverage_is_the_plain_error_rate() -> None:
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    y = np.array([1, 0, 0, 0])  # one mistake: flow 2
    _cov, err = risk_coverage(scores, np.zeros(4), y, 0.5, np.array([1.0]))
    assert err[0] == 0.25
