"""Deferral: the cost accounting must charge for reviews, and the control must degenerate.

Two claims carry the report. The learned rule must be *identical* to the cost-aware rule when
the analyst's skill is constant, since the gap between them is what the report reads as the
value of knowing the analyst; and deferring must cost something, or "defer everything" wins by
construction and the study says nothing. Both are pinned here, along with the modelling point
that separates an expected-loss rule from an accuracy-based one.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.defer import (
    _policy_scores,
    analyst_skill,
    analyst_verdicts,
    defer_mask,
    expected_loss_advantage,
    rank_normalise,
    system_cost,
)


# --------------------------------------------------------------------------------------
# The analyst
# --------------------------------------------------------------------------------------
def test_the_uniform_analyst_ignores_both_covariates() -> None:
    margin = np.linspace(0, 1, 20)
    novelty = np.linspace(1, 0, 20)
    skill = analyst_skill("uniform", margin, novelty, 0.8, 0.4)
    assert np.allclose(skill, 0.8)


def test_the_correlated_analyst_is_best_where_the_model_is_confident() -> None:
    margin = np.array([0.0, 0.5, 1.0])
    novelty = np.zeros(3)
    skill = analyst_skill("correlated", margin, novelty, 0.8, 0.4)
    assert skill[0] < skill[1] < skill[2]


def test_the_complementary_analyst_is_best_on_unfamiliar_flows() -> None:
    margin = np.zeros(3)
    novelty = np.array([0.0, 0.5, 1.0])
    skill = analyst_skill("complementary", margin, novelty, 0.8, 0.4)
    assert skill[0] < skill[1] < skill[2]


def test_skill_never_escapes_a_probability() -> None:
    grid = np.linspace(0, 1, 50)
    for kind in ("uniform", "correlated", "complementary"):
        skill = analyst_skill(kind, grid, grid, 0.9, 5.0)  # an absurd spread
        assert np.all(skill >= 0.0) and np.all(skill <= 1.0)


def test_an_unknown_analyst_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown analyst"):
        analyst_skill("oracle", np.zeros(3), np.zeros(3), 0.8, 0.1)


def test_a_perfect_analyst_reproduces_the_labels_and_a_hopeless_one_inverts_them() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 1, 0, 1])
    assert np.array_equal(analyst_verdicts(np.ones(5), y, rng), y)
    assert np.array_equal(analyst_verdicts(np.zeros(5), y, rng), 1 - y)


def test_analyst_accuracy_matches_the_skill_it_was_given() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=20_000)
    verdicts = analyst_verdicts(np.full(20_000, 0.75), y, rng)
    assert float(np.mean(verdicts == y)) == pytest.approx(0.75, abs=0.02)


# --------------------------------------------------------------------------------------
# The policies
# --------------------------------------------------------------------------------------
def test_rank_normalise_spans_the_unit_interval_regardless_of_scale() -> None:
    values = np.array([1.0, 10.0, 1e9, 2.0])
    ranked = rank_normalise(values)
    assert ranked.min() == 0.0 and ranked.max() == 1.0
    assert np.argmax(ranked) == 2  # order preserved


def test_the_budget_is_a_hard_cap() -> None:
    priority = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    mask = defer_mask(priority, 2)
    assert mask.sum() == 2
    assert list(np.flatnonzero(mask)) == [0, 2]  # the two highest priorities


def test_a_zero_budget_defers_nothing_and_an_oversized_one_defers_everything() -> None:
    priority = np.arange(4.0)
    assert defer_mask(priority, 0).sum() == 0
    assert defer_mask(priority, 99).all()


def test_the_learned_rule_is_the_cost_aware_rule_for_a_constant_analyst() -> None:
    """The control the whole report leans on, driven through the policy dispatcher itself.

    The last column of the report's table is the gap between these two policies, and the
    claim is that it measures only what knowing the analyst is worth. That is only true if
    the gap is exactly zero when there is nothing to know.
    """
    scores = np.array([0.9, 0.1, 0.5, 0.3])
    pred = np.array([1, 0, 0, 0])
    confidence = rank_normalise(np.where(pred == 1, scores, 1 - scores))
    costs = (25.0, 500.0, 25.0)
    rng = np.random.default_rng(0)
    args = (confidence, scores, pred, np.full(4, 0.8), 0.8, costs, rng)
    assert np.allclose(
        _policy_scores("cost-aware", *args), _policy_scores("learned advantage", *args)
    )


def test_the_learned_rule_prefers_flows_where_the_human_is_better() -> None:
    scores = np.full(3, 0.4)
    pred = np.zeros(3, dtype=int)
    skill = np.array([0.5, 0.95, 0.7])
    advantage = expected_loss_advantage(scores, pred, skill, 25.0, 500.0, 25.0)
    assert int(np.argmax(advantage)) == 1


def test_asymmetric_costs_change_which_flow_is_worth_escalating() -> None:
    """The modelling point: a 60%-confident *benign* call outranks a 60%-confident attack call.

    Both flows have the same probability of being wrong, so an accuracy-based rule ranks them
    equally. Only one of them risks a miss, and a miss costs twenty times a false alarm.
    """
    scores = np.array([0.4, 0.6])  # equally uncertain either side of even odds
    pred = np.array([0, 1])  # ... but one is called benign and one attack
    advantage = expected_loss_advantage(scores, pred, np.full(2, 0.9), 25.0, 500.0, 25.0)
    assert advantage[0] > advantage[1]


def test_review_time_is_subtracted_so_a_useless_escalation_scores_negative() -> None:
    certain_benign = expected_loss_advantage(
        np.array([0.001]), np.array([0]), np.array([0.9]), 25.0, 500.0, 25.0
    )
    assert certain_benign[0] < 0.0


def test_a_hopeless_analyst_never_earns_an_escalation() -> None:
    scores = np.linspace(0.01, 0.99, 25)
    pred = (scores >= 0.5).astype(int)
    assert np.all(expected_loss_advantage(scores, pred, np.zeros(25), 25.0, 500.0, 25.0) < 0)


# --------------------------------------------------------------------------------------
# The cost accounting
# --------------------------------------------------------------------------------------
def test_a_review_is_charged_even_when_the_human_is_right() -> None:
    """Without this term 'defer everything' wins by construction."""
    y = np.array([1, 1])
    wrong_model = np.array([0, 0])
    right_human = np.array([1, 1])
    cost = system_cost(y, wrong_model, right_human, np.array([True, True]), 25.0, 500.0, 25.0)
    assert cost == pytest.approx(50.0)  # two reviews, no misses left


def test_deferring_to_a_worse_human_makes_the_system_worse() -> None:
    y = np.array([1, 1, 0, 0])
    good_model = y.copy()
    bad_human = 1 - y
    kept = system_cost(y, good_model, bad_human, np.zeros(4, dtype=bool), 25.0, 500.0, 0.0)
    handed = system_cost(y, good_model, bad_human, np.ones(4, dtype=bool), 25.0, 500.0, 0.0)
    assert kept == 0.0 and handed > kept


def test_misses_are_priced_above_false_alarms() -> None:
    y = np.array([1, 0])
    no_defer = np.zeros(2, dtype=bool)
    miss = system_cost(y, np.array([0, 0]), y, no_defer, 25.0, 500.0, 25.0)
    alarm = system_cost(y, np.array([1, 1]), y, no_defer, 25.0, 500.0, 25.0)
    assert miss > alarm
