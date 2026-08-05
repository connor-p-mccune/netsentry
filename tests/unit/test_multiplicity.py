"""Predictive multiplicity: the Rashomon membership rule and the two disagreement measures.

The report's claims are properties of pure functions over decision matrices, provable
without training anything. These pin the definitions exactly — ambiguity is a union over
the set, discrepancy is a max over single models, and the vote fraction is the abstention
signal that routes contested flows to a human.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.multiplicity import (
    ambiguity,
    candidate_specs,
    discrepancy,
    rashomon_mask,
    route_by_vote,
    vote_fraction,
)


def test_rashomon_mask_keeps_models_within_relative_slack() -> None:
    # champion 0.50, epsilon 10% -> floor 0.45. 0.46 is in, 0.44 is out.
    mask = rashomon_mask(np.array([0.46, 0.44, 0.50]), 0.50, 0.10)
    assert mask.tolist() == [True, False, True]


def test_rashomon_mask_admits_models_that_beat_the_champion() -> None:
    # A competitor that scores *higher* is certainly near-optimal; excluding it would
    # understate the multiplicity.
    assert rashomon_mask(np.array([0.9]), 0.5, 0.01).tolist() == [True]


def test_rashomon_mask_with_zero_epsilon_admits_only_ties_or_better() -> None:
    mask = rashomon_mask(np.array([0.499, 0.5, 0.501]), 0.5, 0.0)
    assert mask.tolist() == [False, True, True]


def test_ambiguity_is_the_union_of_disagreements() -> None:
    champion = np.array([0, 0, 0, 0])
    # Model A flips flow 0; model B flips flow 1. The union is 2 of 4 flows.
    competitors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert ambiguity(champion, competitors) == 0.5


def test_discrepancy_is_the_worst_single_model_not_the_union() -> None:
    champion = np.array([0, 0, 0, 0])
    competitors = np.array([[1, 0, 0, 0], [0, 1, 1, 0]])
    # Union = 3/4 = 0.75, but the worst single model differs on only 2/4 = 0.5.
    assert ambiguity(champion, competitors) == 0.75
    assert discrepancy(champion, competitors) == 0.5


def test_discrepancy_never_exceeds_ambiguity() -> None:
    rng = np.random.default_rng(0)
    champion = rng.integers(0, 2, size=200)
    competitors = rng.integers(0, 2, size=(6, 200))
    assert discrepancy(champion, competitors) <= ambiguity(champion, competitors) + 1e-12


def test_unanimous_set_has_zero_ambiguity_and_discrepancy() -> None:
    champion = np.array([1, 0, 1, 0])
    competitors = np.tile(champion, (5, 1))
    assert ambiguity(champion, competitors) == 0.0
    assert discrepancy(champion, competitors) == 0.0


def test_empty_rashomon_set_reports_no_disagreement() -> None:
    champion = np.array([1, 0, 1])
    empty = np.zeros((0, 3))
    assert ambiguity(champion, empty) == 0.0
    assert discrepancy(champion, empty) == 0.0


def test_vote_fraction_is_the_per_flow_attack_share() -> None:
    decisions = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
    assert np.allclose(vote_fraction(decisions), [0.75, 0.0, 0.75])


def test_route_by_vote_sends_only_the_contested_band_to_review() -> None:
    votes = np.array([0.0, 0.5, 1.0, 0.5])
    is_attack = np.array([False, True, True, False])
    contested = np.array([False, True, False, True])
    out = route_by_vote(votes, is_attack, contested, 0.4, 0.6)
    assert out.review_rate == 0.5  # the two 0.5 flows
    assert out.auto_alert_rate == 0.25  # only the unanimous 1.0
    assert out.residual_ambiguity == 0.0  # both contested flows went to review


def test_route_by_vote_leaves_residual_ambiguity_when_the_band_is_too_narrow() -> None:
    votes = np.array([0.1, 0.5, 0.9])
    contested = np.array([True, True, True])
    out = route_by_vote(votes, np.array([False, True, True]), contested, 0.45, 0.55)
    assert out.review_rate == 1 / 3
    # The 0.1 and 0.9 flows are auto-decided despite being contested.
    assert np.isclose(out.residual_ambiguity, 1.0)


def test_route_by_vote_reports_precision_and_recall_of_the_auto_alerts() -> None:
    votes = np.array([1.0, 1.0, 0.0, 0.0])
    is_attack = np.array([True, False, True, False])
    contested = np.zeros(4, dtype=bool)
    out = route_by_vote(votes, is_attack, contested, 0.4, 0.6)
    assert out.auto_alert_precision == 0.5  # one of two auto-alerts is an attack
    assert out.auto_alert_recall == 0.5  # one of two attacks is auto-alerted


def test_route_by_vote_handles_an_empty_stream() -> None:
    out = route_by_vote(np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), 0.4, 0.6)
    assert out.review_rate == 0.0 and out.auto_alert_precision == 0.0


def test_candidate_specs_are_deterministic_and_distinctly_seeded(settings: object) -> None:
    from netsentry.config import Settings

    assert isinstance(settings, Settings)
    first = candidate_specs(settings.multiplicity, 42)
    second = candidate_specs(settings.multiplicity, 42)
    assert first == second
    assert len(first) == settings.multiplicity.n_models
    # Distinct seeds are what make the pool a pool rather than one model refit.
    assert len({s.seed for s in first}) == len(first)
    for spec in first:
        assert spec.subsample in settings.multiplicity.subsample_choices
        assert spec.num_leaves in settings.multiplicity.num_leaves_choices
        assert "seed=" in spec.label()
