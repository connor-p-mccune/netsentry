"""Off-policy evaluation: unbiasedness, the double-robustness property, and support.

An OPE estimator that is subtly wrong still returns plausible currency figures, so these
pin the properties that define each one. The central test is the one that matters in
practice: IPS recovers the true value of a policy that was never run, and only because the
logging policy explored — remove the exploration and the same estimator silently reports
a number for a question the data cannot answer.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.ope import (
    Economics,
    deterministic_propensity,
    dm_value,
    dr_value,
    effective_sample_size,
    epsilon_greedy_propensity,
    importance_weights,
    ips_value,
    policy_value,
    snips_value,
    support_violation_rate,
)

ECON = Economics(value_per_catch=500.0, cost_per_review=25.0)


def test_reward_is_zero_for_every_flow_nobody_reviewed() -> None:
    # The property that makes the formulation honest under partial feedback: the skip arm
    # needs no outcome, because nothing was spent and nothing was found.
    r = ECON.reward(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert r.tolist() == [0.0, 0.0, -25.0, 475.0]


def test_epsilon_greedy_propensity_is_bounded_away_from_zero_and_one() -> None:
    p = epsilon_greedy_propensity(np.array([0.1, 0.9]), 0.5, 0.1)
    assert np.allclose(p, [0.05, 0.95])
    assert np.all(p > 0.0) and np.all(p < 1.0)


def test_zero_exploration_reproduces_a_plain_deterministic_threshold() -> None:
    scores = np.array([0.1, 0.6, 0.9])
    assert np.allclose(
        epsilon_greedy_propensity(scores, 0.5, 0.0), deterministic_propensity(scores, 0.5)
    )


def test_ips_recovers_the_value_of_a_policy_that_was_never_deployed() -> None:
    """The whole point: estimate a lower threshold's value from logs of a higher one.

    Unbiasedness is an expectation, so this averages over many logged streams; a biased
    estimator would sit outside the tolerance no matter how many replicates are drawn.
    """
    rng = np.random.default_rng(0)
    n = 4_000
    scores = rng.random(n)
    is_attack = (rng.random(n) < 0.3 * scores).astype(int)  # higher score, likelier attack
    logged_p = epsilon_greedy_propensity(scores, 0.8, 0.3)
    target_p = deterministic_propensity(scores, 0.4)  # reviews far more than the logger
    truth = policy_value(target_p, is_attack, ECON)

    estimates = []
    for _ in range(300):
        actions = (rng.random(n) < logged_p).astype(int)
        rewards = ECON.reward(actions, is_attack)
        w = importance_weights(actions, logged_p, target_p)
        estimates.append(ips_value(rewards, w))
    assert abs(float(np.mean(estimates)) - truth) < 0.03 * abs(truth)


def test_importance_weight_is_one_when_the_target_is_the_logging_policy() -> None:
    scores = np.array([0.2, 0.7, 0.9])
    p = epsilon_greedy_propensity(scores, 0.5, 0.2)
    for actions in ([0, 0, 0], [1, 1, 1], [1, 0, 1]):
        assert np.allclose(importance_weights(np.array(actions), p, p), 1.0)


def test_doubly_robust_is_exact_when_the_reward_model_is_perfect() -> None:
    """Half of "doubly": a correct reward model makes the residual correction vanish."""
    rng = np.random.default_rng(1)
    n = 500
    scores = rng.random(n)
    is_attack = (scores > 0.7).astype(int)  # deterministic, so a perfect model exists
    perfect = ECON.value_per_catch * is_attack - ECON.cost_per_review
    logged_p = epsilon_greedy_propensity(scores, 0.6, 0.4)
    target_p = deterministic_propensity(scores, 0.3)
    actions = (rng.random(n) < logged_p).astype(int)
    rewards = ECON.reward(actions, is_attack)
    w = importance_weights(actions, logged_p, target_p)
    truth = policy_value(target_p, is_attack, ECON)
    assert dr_value(actions, rewards, w, target_p, perfect) == truth


def test_doubly_robust_stays_unbiased_when_the_reward_model_is_nonsense() -> None:
    """The other half: wrong reward model, right propensities, still unbiased."""
    rng = np.random.default_rng(2)
    n = 3_000
    scores = rng.random(n)
    is_attack = (rng.random(n) < 0.4 * scores).astype(int)
    nonsense = np.full(n, 900.0)  # wildly wrong about every flow
    logged_p = epsilon_greedy_propensity(scores, 0.7, 0.4)
    target_p = deterministic_propensity(scores, 0.35)
    truth = policy_value(target_p, is_attack, ECON)
    estimates = []
    for _ in range(300):
        actions = (rng.random(n) < logged_p).astype(int)
        rewards = ECON.reward(actions, is_attack)
        w = importance_weights(actions, logged_p, target_p)
        estimates.append(dr_value(actions, rewards, w, target_p, nonsense))
    assert abs(float(np.mean(estimates)) - truth) < 0.05 * abs(truth)


def test_direct_method_inherits_whatever_its_reward_model_believes() -> None:
    # No propensities involved: the estimate is the model integrated under the policy.
    target_p = np.array([1.0, 1.0, 0.0])
    assert dm_value(target_p, np.array([100.0, 200.0, 999.0])) == 100.0


def test_snips_is_invariant_to_rescaling_the_reward_and_ips_is_not() -> None:
    rewards = np.array([10.0, -5.0, 20.0])
    weights = np.array([2.0, 0.5, 3.0])
    assert np.isclose(snips_value(rewards * 7.0, weights), 7.0 * snips_value(rewards, weights))
    # Self-normalisation divides out a uniformly inflated weight; IPS does not.
    assert np.isclose(snips_value(rewards, weights * 4.0), snips_value(rewards, weights))
    assert not np.isclose(ips_value(rewards, weights * 4.0), ips_value(rewards, weights))


def test_support_violation_is_total_when_the_logging_policy_never_explores() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    logged = deterministic_propensity(scores, 0.5)  # reviews only the 0.9 flow
    target = deterministic_propensity(scores, 0.15)  # wants to review three
    # Two of the three flows it would review were never reviewable under the log.
    assert support_violation_rate(logged, target) == 2 / 3


def test_exploration_removes_the_support_violation_entirely() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    logged = epsilon_greedy_propensity(scores, 0.5, 0.05)
    target = deterministic_propensity(scores, 0.15)
    assert support_violation_rate(logged, target) == 0.0


def test_effective_sample_size_falls_when_one_flow_dominates_the_weights() -> None:
    assert effective_sample_size(np.ones(100)) == 100.0
    dominated = np.concatenate([np.ones(99), [1000.0]])
    assert effective_sample_size(dominated) < 5.0


def test_policy_value_prices_a_review_all_policy_against_the_attack_rate() -> None:
    is_attack = np.array([1, 0, 0, 0])  # 25% attacks
    review_all = np.ones(4)
    # 0.25 * 500 - 25 = 100 per flow.
    assert policy_value(review_all, is_attack, ECON) == 100.0
    assert policy_value(np.zeros(4), is_attack, ECON) == 0.0
