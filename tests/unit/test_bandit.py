"""The bandit learners, the reward bookkeeping, and the regret reference.

Two things here are easy to get wrong and impossible to notice from a plausible-looking report:
feeding a learner the counterfactual reward (which quietly turns partial feedback into full
supervision and makes every bandit look brilliant), and choosing a regret reference that flatters
whatever you built. Both get their own test.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.bandit import (
    EPSILON,
    FIXED,
    LINUCB,
    REVIEW,
    SKIP,
    THOMPSON,
    ArmRun,
    BanditStudy,
    EpsilonGreedy,
    LinearBandit,
    best_threshold_rewards,
    build_context,
    learner_chooser,
    random_chooser,
    run_policy,
    threshold_chooser,
)
from netsentry.evaluation.ope import Economics

ECON = Economics(value_per_catch=500.0, cost_per_review=25.0)


def _stream(n: int = 400, rate: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels where reviewing high scores pays and reviewing low ones does not."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < rate).astype(int)
    scores = np.clip(rng.beta(2.0, 8.0, n) + labels * 0.6, 0.0, 1.0)
    return scores, labels


def _contexts(scores: np.ndarray) -> np.ndarray:
    return np.vstack([build_context(float(s), 0.0) for s in scores])


# --------------------------------------------------------------------------------------
# The context.
# --------------------------------------------------------------------------------------


def test_the_context_carries_an_intercept_and_the_scores() -> None:
    context = build_context(0.4, -1.0)
    assert context[0] == 1.0  # without an intercept the learner cannot express a base rate
    assert context[1] == pytest.approx(0.4)
    assert context[2] == pytest.approx(-1.0)
    assert context[3] == pytest.approx(0.16)  # the curvature term


# --------------------------------------------------------------------------------------
# The learners.
# --------------------------------------------------------------------------------------


def test_linucb_explores_before_it_has_evidence() -> None:
    """With no data the confidence bonus decides, so both actions get tried early.

    Note what the model does *without* updates: the two actions have identical value and
    identical width, so `argmax` breaks the tie toward skipping, forever. Exploration comes
    from the widths diverging as one action accumulates observations -- which is to say a
    bandit only explores when it is being fed, and a test that forgets to feed it will report
    that LinUCB never explores at all.
    """
    model = LinearBandit(4, alpha=1.0, rng=np.random.default_rng(0))
    scores, labels = _stream(60)
    run = run_policy("linucb", learner_chooser(model), _contexts(scores), labels, ECON, model)
    assert set(run.actions.tolist()) == {SKIP, REVIEW}


def test_linucb_learns_that_a_punished_action_is_bad() -> None:
    model = LinearBandit(4, alpha=0.05, rng=np.random.default_rng(1))
    context = build_context(0.5, 0.0)
    for _ in range(200):
        model.update(context, REVIEW, -1.0)
        model.update(context, SKIP, 0.0)
    assert model.choose(context) == SKIP


def test_linucb_is_deterministic_given_the_stream() -> None:
    """Its exploration is a function of what it has seen, not of a coin.

    This is why the report's spread column is exactly zero for LinUCB and not for Thompson.
    """
    scores, labels = _stream(300)
    contexts = _contexts(scores)
    totals = []
    for seed in (1, 2, 3):
        model = LinearBandit(4, 1.0, np.random.default_rng(seed))
        totals.append(run_policy("x", learner_chooser(model), contexts, labels, ECON, model).total)
    assert len(set(totals)) == 1


def test_thompson_sampling_is_not_deterministic() -> None:
    scores, labels = _stream(300)
    contexts = _contexts(scores)
    totals = set()
    for seed in (1, 2, 3):
        model = LinearBandit(4, 1.0, np.random.default_rng(seed), thompson=True)
        totals.add(run_policy("x", learner_chooser(model), contexts, labels, ECON, model).total)
    assert len(totals) > 1


def test_epsilon_greedy_explores_at_about_its_rate() -> None:
    """The control's exploration is blind, which is exactly what makes it a control."""
    model = EpsilonGreedy(4, epsilon=0.5, rng=np.random.default_rng(2))
    context = build_context(0.9, 0.0)
    for _ in range(300):  # teach it decisively that reviewing is bad here
        model.update(context, REVIEW, -1.0)
    reviews = sum(model.choose(context) == REVIEW for _ in range(2000))
    assert 0.2 < reviews / 2000 < 0.35  # ~epsilon/2 of the time it explores into the bad action


# --------------------------------------------------------------------------------------
# The bookkeeping: partial feedback, and the reward the learner is allowed to see.
# --------------------------------------------------------------------------------------


class _Recorder:
    """A learner that records what it was told, so the feedback discipline can be asserted."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.seen: list[tuple[int, float]] = []

    def choose(self, context: np.ndarray) -> int:
        return self.action

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.seen.append((action, reward))


def test_a_learner_is_only_told_about_the_action_it_took() -> None:
    """The discipline the whole study rests on: no counterfactual reward, ever.

    Feeding both arms' outcomes turns a bandit into supervised learning and makes exploration
    free -- which is precisely the cost this report exists to measure.
    """
    scores, labels = _stream(200)
    recorder = _Recorder(SKIP)
    run_policy("skip", learner_chooser(recorder), _contexts(scores), labels, ECON, recorder)
    assert len(recorder.seen) == 200
    assert {action for action, _ in recorder.seen} == {SKIP}
    assert all(reward == 0.0 for _, reward in recorder.seen)  # skipping reveals nothing


def test_the_learner_sees_a_scaled_reward_while_the_report_sees_dollars() -> None:
    """Unscaled, the exploration term is negligible beside the first bad draw."""
    scores, labels = _stream(50, rate=0.0)
    recorder = _Recorder(REVIEW)
    run = run_policy("review", learner_chooser(recorder), _contexts(scores), labels, ECON, recorder)
    assert run.total == pytest.approx(-25.0 * 50)  # dollars, in the report
    assert all(reward == pytest.approx(-0.05) for _, reward in recorder.seen)  # scaled, internally


def test_the_arm_accounts_for_every_flow_exactly_once() -> None:
    scores, labels = _stream(300)
    run = run_policy("threshold", threshold_chooser(scores, 0.5), _contexts(scores), labels, ECON)
    assert run.caught + run.missed == int(labels.sum())
    assert run.reviewed == int((scores >= 0.5).sum())


def test_reviewing_an_attack_pays_and_reviewing_a_benign_flow_costs() -> None:
    contexts = _contexts(np.array([0.9, 0.9]))
    labels = np.array([1, 0])
    run = run_policy("all", random_chooser(1.0, np.random.default_rng(0)), contexts, labels, ECON)
    assert run.rewards[0] == pytest.approx(475.0)
    assert run.rewards[1] == pytest.approx(-25.0)


def test_skipping_scores_exactly_zero_whatever_the_flow_was() -> None:
    # The formulation that makes partial feedback coherent: the un-reviewed arm needs no
    # outcome, because nothing was spent and nothing was found.
    contexts = _contexts(np.array([0.9, 0.1]))
    run = run_policy(
        "none", random_chooser(0.0, np.random.default_rng(0)), contexts, np.array([1, 0]), ECON
    )
    assert run.total == 0.0


# --------------------------------------------------------------------------------------
# The regret reference.
# --------------------------------------------------------------------------------------


def test_the_hindsight_reference_is_the_best_threshold_not_the_best_action() -> None:
    """At a low attack rate 'review nothing' beats 'review everything', and is a useless bar."""
    scores, labels = _stream(2000, rate=0.01, seed=3)
    rewards, cut = best_threshold_rewards(scores, labels, ECON)
    assert rewards.sum() > 0.0  # a threshold policy makes money where a blanket one does not
    assert ECON.reward(np.ones(len(labels)), labels).sum() < rewards.sum()
    assert 0.0 <= cut <= 1.0


def test_no_threshold_policy_can_beat_the_hindsight_reference() -> None:
    scores, labels = _stream(1500, seed=4)
    best, _ = best_threshold_rewards(scores, labels, ECON)
    for cut in np.linspace(0.0, 1.0, 25):
        assert ECON.reward((scores >= cut).astype(int), labels).sum() <= best.sum() + 1e-9


# --------------------------------------------------------------------------------------
# The study's own arithmetic.
# --------------------------------------------------------------------------------------


def _study(rewards: dict[str, np.ndarray], reference: np.ndarray, attacks: int) -> BanditStudy:
    arms = [
        ArmRun(
            name=name,
            rewards=values,
            actions=(values != 0).astype(int),
            reviewed=int((values != 0).sum()),
            caught=int((values > 0).sum()),
            missed=attacks - int((values > 0).sum()),
        )
        for name, values in rewards.items()
    ]
    return BanditStudy(
        arms=arms,
        oracle_rewards=reference,
        best_fixed_rewards=reference,
        best_fixed_name="reference",
        n_flows=len(reference),
        prevalence=attacks / len(reference),
        economics=ECON,
        threshold=0.5,
        attacks=attacks,
    )


def test_a_policy_matching_the_reference_has_zero_regret() -> None:
    reference = np.array([475.0, -25.0, 0.0, 475.0])
    study = _study({FIXED: reference.copy()}, reference, attacks=2)
    assert study.regret(FIXED)[-1] == pytest.approx(0.0)


def test_regret_accumulates_where_the_policy_falls_behind() -> None:
    reference = np.array([475.0, 475.0, 475.0])
    study = _study({LINUCB: np.array([475.0, 0.0, 0.0])}, reference, attacks=3)
    assert study.regret(LINUCB).tolist() == pytest.approx([0.0, 475.0, 950.0])


def test_linear_regret_reads_as_an_exponent_near_one() -> None:
    """The diagnostic that separates 'learning slowly' from 'not learning at all'."""
    n = 3000
    reference = np.full(n, 1.0)
    study = _study({EPSILON: np.zeros(n)}, reference, attacks=0)  # falls behind every step
    assert study.regret_slope(EPSILON) == pytest.approx(1.0, abs=0.05)


def test_flat_regret_reads_as_an_exponent_near_zero() -> None:
    n = 3000
    reference = np.zeros(n)
    rewards = np.zeros(n)
    rewards[0] = -100.0  # one early mistake, nothing after it
    study = _study({THOMPSON: rewards}, reference, attacks=0)
    assert abs(study.regret_slope(THOMPSON)) < 0.05


def test_the_realised_alert_budget_counts_benign_reviews_only() -> None:
    reference = np.zeros(4)
    study = _study({LINUCB: np.array([475.0, -25.0, -25.0, 0.0])}, reference, attacks=1)
    assert study.realised_fpr(LINUCB) == pytest.approx(2 / 3)  # 2 benign reviews of 3 benign flows
