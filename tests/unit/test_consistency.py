"""The harvester's three filters, the interval rungs, and the attribution ranking.

Every filter here is a regression test: the first version of the study reported the value each
one now rejects as a disagreement between reports. The attribution ranking is the other
load-bearing part -- picking by distance alone lets a wide interval explain every value in the
table, which is a curve that fits everything rather than a diagnosis.
"""

from __future__ import annotations

import pytest

from netsentry.evaluation.consistency import (
    Quotation,
    Variation,
    attribute,
    harvest,
    names_another_metric,
    rejection_reason,
    spread,
)

BAND = (0.40, 0.70)


def _harvest(body: str) -> tuple[list[Quotation], list[Quotation]]:
    return harvest({"r": body}, *BAND)


# --------------------------------------------------------------------------------------
# What counts as the incumbent's score.
# --------------------------------------------------------------------------------------


def test_a_plain_statement_is_harvested() -> None:
    kept, rejected = _harvest("The deployed model scores 0.529 PR-AUC on the later days.")
    assert [q.value for q in kept] == [0.529]
    assert rejected == []


def test_a_value_outside_the_band_is_ignored() -> None:
    """Without the band the pattern collects coverage levels and p-values near the word."""
    kept, rejected = _harvest("The baseline holds coverage at 0.900 across the split.")
    assert kept == [] and rejected == []


def test_a_value_far_from_its_qualifier_is_ignored() -> None:
    filler = "x" * 80
    kept, _ = _harvest(f"The deployed model {filler} and separately 0.529 appears.")
    assert kept == []


def test_the_same_value_is_counted_once_per_report() -> None:
    kept, _ = _harvest("The incumbent scores 0.529. Later: the incumbent scores 0.529 again.")
    assert len(kept) == 1


# --------------------------------------------------------------------------------------
# Filter one: a different metric.
# --------------------------------------------------------------------------------------


def test_a_roc_auc_is_not_an_average_precision() -> None:
    """ROC-AUC is insensitive to prevalence where PR-AUC is not; conflating them invents a gap."""
    kept, rejected = _harvest("The deployed model scores AUC 0.668 but H 0.180.")
    assert kept == []
    assert [q.reason for q in rejected] == ["metric"]


def test_pr_auc_is_not_mistaken_for_roc_auc() -> None:
    assert not names_another_metric("the incumbent reaches 0.529 PR-AUC here")
    assert names_another_metric("the incumbent reaches an AUROC of 0.693")


# --------------------------------------------------------------------------------------
# Filter two: the far side of a comparison.
# --------------------------------------------------------------------------------------


def test_the_qualifier_after_a_value_wins_when_it_is_nearer() -> None:
    """ "rises from X (static) to Y (retrained)" -- a backwards-only rule mis-attributes Y."""
    kept, rejected = _harvest("Mean PR-AUC rises from 0.433 (static) to 0.544 (retrained).")
    assert 0.544 not in [q.value for q in kept]
    assert "rival" in [q.reason for q in rejected]


def test_a_rival_further_away_does_not_steal_the_value() -> None:
    reason = rejection_reason(
        "the deployed model scores 0.529, unlike the MLP", "deployed", "0.529"
    )
    assert reason == ""


def test_a_rival_between_the_qualifier_and_the_value_steals_it() -> None:
    """The real sentence: "clean baseline day ... its local model posts 0.583 PR-AUC"."""
    kept, rejected = _harvest("It is the clean baseline day and its local model posts 0.583 here.")
    assert kept == []
    assert rejected and rejected[0].reason == "rival"


def test_a_rival_in_a_trailing_clause_does_not() -> None:
    """A comma is a clause break; "unlike the MLP" describes a contrast, not the number."""
    kept, _ = _harvest("The deployed model scores 0.529 PR-AUC, unlike the MLP.")
    assert [q.value for q in kept] == [0.529]


# --------------------------------------------------------------------------------------
# Filter three: the qualifier used in another sense.
# --------------------------------------------------------------------------------------


def test_a_baseline_day_is_not_a_baseline_model() -> None:
    assert rejection_reason("the baseline day carries 0.583 of it", "baseline", "0.583") == "sense"


def test_a_baseline_model_is_still_a_model() -> None:
    assert rejection_reason("the baseline model scores 0.529", "baseline", "0.529") == ""


# --------------------------------------------------------------------------------------
# Rungs: points and intervals.
# --------------------------------------------------------------------------------------


def _point(name: str, value: float) -> Variation:
    return Variation(name=name, describes="", value=value, canonical=0.5)


def _interval(name: str, low: float, high: float) -> Variation:
    return Variation(
        name=name, describes="", value=(low + high) / 2, canonical=0.5, low=low, high=high
    )


def test_a_point_rung_claims_none_of_the_axis() -> None:
    assert _point("p", 0.53).width == 0.0
    assert not _point("p", 0.53).random


def test_an_interval_rung_claims_its_range() -> None:
    rung = _interval("i", 0.51, 0.55)
    assert rung.random
    assert rung.width == pytest.approx(0.04)


def test_a_point_rung_covers_only_within_tolerance() -> None:
    rung = _point("p", 0.530)
    assert rung.covers(0.532, 0.005) and not rung.covers(0.560, 0.005)


def test_an_interval_rung_covers_anything_inside_it() -> None:
    rung = _interval("i", 0.51, 0.55)
    assert rung.covers(0.516, 0.0) and rung.covers(0.549, 0.0)
    assert not rung.covers(0.60, 0.0)


def test_distance_to_an_interval_is_measured_to_its_edge() -> None:
    rung = _interval("i", 0.51, 0.55)
    assert rung.distance(0.53) == 0.0
    assert rung.distance(0.57) == pytest.approx(0.02)


# --------------------------------------------------------------------------------------
# Attribution.
# --------------------------------------------------------------------------------------


def _quoted(value: float) -> list[Quotation]:
    return [Quotation(report="r", value=value, context="")]


def test_a_single_covering_rung_pins_the_value() -> None:
    rows = attribute(_quoted(0.53), [_point("near", 0.530), _point("far", 0.90)], 0.005)
    assert rows[0].specific and rows[0].knob == "near"


def test_several_covering_rungs_only_bracket_it() -> None:
    """A value consistent with two stories has been bracketed, not attributed."""
    rows = attribute(_quoted(0.53), [_point("a", 0.530), _point("b", 0.532)], 0.005)
    assert rows[0].explained and not rows[0].specific
    assert rows[0].candidates == 2


def test_a_point_rung_beats_an_interval_that_also_covers() -> None:
    """Picking by distance alone would let a wide enough range explain everything."""
    rows = attribute(_quoted(0.53), [_interval("wide", 0.40, 0.70), _point("tight", 0.530)], 0.005)
    assert rows[0].knob == "tight"


def test_a_value_no_rung_reaches_is_unexplained() -> None:
    rows = attribute(_quoted(0.53), [_point("a", 0.60), _point("b", 0.70)], 0.005)
    assert not rows[0].explained
    assert rows[0].gap == pytest.approx(-0.07)


def test_attribution_collects_every_report_stating_a_value() -> None:
    quotations = [
        Quotation(report="a", value=0.53, context=""),
        Quotation(report="b", value=0.53, context=""),
    ]
    rows = attribute(quotations, [_point("p", 0.53)], 0.005)
    assert rows[0].reports == ("a", "b")


def test_the_spread_is_the_distinct_values_in_order() -> None:
    quotations = [
        Quotation(report="a", value=0.537, context=""),
        Quotation(report="b", value=0.516, context=""),
        Quotation(report="c", value=0.537, context=""),
    ]
    assert spread(quotations) == [0.516, 0.537]
