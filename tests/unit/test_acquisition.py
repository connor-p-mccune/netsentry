"""Cost-aware acquisition: the tier arithmetic, the gates, and the diagnostic that explains them.

The escalation loop is the part that can be wrong while still producing a plausible frontier —
a flow judged by the wrong tier, or a cost that double-counts a family already bought — so the
tests pin the accounting exactly and then check that each gate escalates the population it
claims to.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from netsentry.evaluation.acquisition import (
    ADAPTIVE,
    ASYMMETRIC,
    RANDOM,
    Family,
    TierModel,
    adaptive_policy,
    build_families,
    cumulative_tiers,
    detection_retention,
    tier_columns,
    tier_cost,
    uncertainty_band,
)


def _tier(scores: np.ndarray, threshold: float) -> TierModel:
    return TierModel(
        columns=[],
        cost=0.0,
        threshold=threshold,
        scores_val=scores,
        scores_test=scores,
        tpr=0.0,
        fpr=0.0,
    )


# --------------------------------------------------------------------------------------
# Families and tiers.
# --------------------------------------------------------------------------------------


def test_families_are_ordered_cheapest_first() -> None:
    prices = {
        "TCP flags": 0.5,
        "header/window/bulk": 1.0,
        "volume/counts": 2.0,
        "packet size": 4.0,
        "flow rates": 6.0,
        "timing/IAT": 10.0,
    }
    families = build_families(prices)
    costs = [family.price for family in families]
    assert costs == sorted(costs)
    assert families[0].name == "TCP flags"
    assert families[-1].name == "timing/IAT"


def test_unpriced_families_get_a_default_rather_than_vanishing() -> None:
    families = build_families({"TCP flags": 1.0})
    assert len(families) > 1
    assert all(family.price > 0 for family in families)


def test_tiers_are_nested_and_costs_accumulate() -> None:
    families = [
        Family("cheap", ["a"], 1.0),
        Family("medium", ["b", "c"], 2.0),
        Family("dear", ["d"], 5.0),
    ]
    tiers = cumulative_tiers(families)
    assert [tier_cost(tier) for tier in tiers] == [1.0, 3.0, 8.0]
    assert tier_columns(tiers[-1]) == ["a", "b", "c", "d"]
    for smaller, larger in pairwise(tiers):
        assert set(tier_columns(smaller)) < set(tier_columns(larger))


# --------------------------------------------------------------------------------------
# The gates.
# --------------------------------------------------------------------------------------


def test_the_uncertainty_band_is_measured_in_rank_space() -> None:
    # Raw score distances are useless near a 0.1% operating point, where the scores pile up;
    # in rank space a band of 0.1 always selects about a tenth of the flows.
    scores = np.concatenate([np.linspace(0, 0.001, 900), np.linspace(0.9, 1.0, 100)])
    band = uncertainty_band(scores, threshold=0.95, width=0.1)
    assert 0.1 < band.mean() < 0.3


def test_a_wider_band_escalates_more() -> None:
    scores = np.linspace(0, 1, 1000)
    narrow = uncertainty_band(scores, 0.5, 0.05).sum()
    wide = uncertainty_band(scores, 0.5, 0.3).sum()
    assert narrow < wide


# --------------------------------------------------------------------------------------
# The escalation loop's accounting.
# --------------------------------------------------------------------------------------


def test_a_policy_that_escalates_nothing_costs_exactly_the_cheap_tier() -> None:
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 1, 500)
    tiers = [_tier(scores, 0.5), _tier(scores, 0.5)]
    point = adaptive_policy(tiers, [1.0, 10.0], (scores > 0.5).astype(int), 0.0, rng, gate="top")
    assert point.mean_cost == pytest.approx(1.0)
    assert point.escalated == 0.0


def test_a_zero_width_band_still_escalates_the_exact_ties() -> None:
    # Not a rounding artifact worth hiding: |rank - threshold rank| <= 0 catches the flows
    # sitting exactly on the threshold, and the accounting has to charge for them.
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 1, 500)
    point = adaptive_policy(
        [_tier(scores, 0.5), _tier(scores, 0.5)], [1.0, 10.0], (scores > 0.5).astype(int), 0.0, rng
    )
    assert 1.0 < point.mean_cost < 1.1


def test_a_policy_that_escalates_everything_costs_the_full_tier() -> None:
    rng = np.random.default_rng(1)
    scores = np.linspace(0, 1, 500)
    tiers = [_tier(scores, 0.5), _tier(scores, 0.5)]
    point = adaptive_policy(tiers, [1.0, 10.0], (scores > 0.5).astype(int), 1.0, rng, gate="top")
    assert point.mean_cost == pytest.approx(10.0)


def test_cost_never_double_counts_a_tier_already_bought() -> None:
    # Three tiers priced 1, 3, 8: a flow that escalates twice must cost 8, not 1 + 3 + 8.
    rng = np.random.default_rng(2)
    scores = np.ones(100)
    tiers = [_tier(scores, 0.5), _tier(scores, 0.5), _tier(scores, 0.5)]
    point = adaptive_policy(tiers, [1.0, 3.0, 8.0], np.ones(100, dtype=int), 1.0, rng, gate="top")
    assert point.mean_cost == pytest.approx(8.0)


def test_the_verdict_comes_from_the_last_tier_that_saw_the_flow() -> None:
    # The cheap tier says benign for everything; the expensive tier says attack. Escalated
    # flows must be detected, which is only true if the final verdict is the last tier's.
    rng = np.random.default_rng(3)
    cheap = _tier(np.zeros(100), 0.5)
    dear = _tier(np.ones(100), 0.5)
    labels = np.ones(100, dtype=int)
    escalated = adaptive_policy([cheap, dear], [1.0, 2.0], labels, 1.0, rng, gate="top")
    not_escalated = adaptive_policy([cheap, dear], [1.0, 2.0], labels, 0.0, rng, gate="top")
    assert escalated.tpr == 1.0
    assert not_escalated.tpr == 0.0


def test_each_gate_reports_its_own_policy_name() -> None:
    rng = np.random.default_rng(4)
    scores = np.linspace(0, 1, 200)
    tiers = [_tier(scores, 0.5), _tier(scores, 0.5)]
    labels = (scores > 0.5).astype(int)
    assert adaptive_policy(tiers, [1.0, 2.0], labels, 0.1, rng).policy == ADAPTIVE
    assert adaptive_policy(tiers, [1.0, 2.0], labels, 0.1, rng, gate="top").policy == ASYMMETRIC
    assert adaptive_policy(tiers, [1.0, 2.0], labels, 0.1, rng, gate="random").policy == RANDOM


def test_random_gating_spends_about_what_the_uncertainty_gate_spends() -> None:
    # The control is only a control if it matches the *spend*; otherwise it is a cheaper policy
    # losing to a dearer one and says nothing about the signal.
    rng = np.random.default_rng(5)
    scores = np.linspace(0, 1, 2000)
    tiers = [_tier(scores, 0.9), _tier(scores, 0.9)]
    labels = (scores > 0.9).astype(int)
    gated = adaptive_policy(tiers, [1.0, 5.0], labels, 0.2, rng)
    control = adaptive_policy(tiers, [1.0, 5.0], labels, 0.2, rng, gate="random")
    assert control.mean_cost == pytest.approx(gated.mean_cost, rel=0.15)


# --------------------------------------------------------------------------------------
# The diagnostic.
# --------------------------------------------------------------------------------------


def test_retention_is_total_when_the_tiers_rank_identically() -> None:
    scores = np.linspace(0, 1, 1000)
    labels = (scores > 0.98).astype(int)
    cheap = _tier(scores, 0.98)
    full = _tier(scores, 0.98)
    row = detection_retention(cheap, full, labels, keep=0.05)
    assert row.retained == pytest.approx(1.0)
    assert row.detected_by_full > 0


def test_retention_collapses_when_the_cheap_tier_ranks_backwards() -> None:
    # The failure the report diagnoses: a filter built on a cheap score that disagrees with the
    # expensive model forwards none of its detections, whatever the escalation policy.
    scores = np.linspace(0, 1, 1000)
    labels = (scores > 0.98).astype(int)
    full = _tier(scores, 0.98)
    cheap = _tier(1.0 - scores, 0.98)  # perfectly inverted ranking
    row = detection_retention(cheap, full, labels, keep=0.1)
    assert row.retained == pytest.approx(0.0)


def test_retention_is_undefined_rather_than_zero_when_nothing_is_detected() -> None:
    scores = np.linspace(0, 1, 100)
    full = _tier(scores, 2.0)  # a threshold nothing reaches
    row = detection_retention(_tier(scores, 0.5), full, np.ones(100, dtype=int), keep=0.5)
    assert row.detected_by_full == 0
    assert np.isnan(row.retained)
