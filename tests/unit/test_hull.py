"""The frontier, the interpolation, and the two threshold-free metrics.

The hull is the part with a wrong answer that looks plausible: pop on the wrong turn and the
same code returns the *lower* hull, which is the diagonal -- a "frontier" whose detection rate
equals its false-positive rate. That happened, so it has a regression test, along with the
closed forms for net benefit and normalised cost that make the other two sections checkable
rather than merely plotted.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.hull import (
    BenefitRow,
    hull_detection_at,
    net_benefit,
    normalised_cost,
    roc_points,
    threshold_detection_at,
    upper_hull,
)

# An ROC with a deliberate dip: (0.5, 0.6) sits below the chord from (0.1, 0.5) to (1.0, 1.0).
FPR = np.array([0.0, 0.1, 0.5, 1.0])
TPR = np.array([0.0, 0.5, 0.6, 1.0])


# --------------------------------------------------------------------------------------
# The hull.
# --------------------------------------------------------------------------------------


def test_the_hull_drops_a_point_below_its_own_chord() -> None:
    hull = upper_hull(FPR, TPR)
    assert sorted(FPR[hull].tolist()) == [0.0, 0.1, 1.0]


def test_the_hull_is_not_the_diagonal() -> None:
    """The regression test for popping on the wrong turn: the lower hull *is* the diagonal."""
    hull = upper_hull(FPR, TPR)
    assert not np.allclose(FPR[hull], TPR[hull])
    assert float(np.max(TPR[hull] - FPR[hull])) > 0.3


def test_every_roc_point_lies_on_or_below_the_hull() -> None:
    """The defining property, checked by interpolation rather than by inspection."""
    hull = upper_hull(FPR, TPR)
    for point in range(len(FPR)):
        best, _, _ = hull_detection_at(FPR, TPR, hull, float(FPR[point]))
        assert best >= TPR[point] - 1e-12


def test_a_convex_curve_keeps_all_of_its_points() -> None:
    fpr = np.array([0.0, 0.2, 0.6, 1.0])
    tpr = np.array([0.0, 0.7, 0.95, 1.0])
    assert len(upper_hull(fpr, tpr)) == 4


def test_two_points_at_the_same_budget_keep_the_better_one() -> None:
    """Ties in false-positive rate are common on a coarse ROC; the worse one is dominated."""
    fpr = np.array([0.0, 0.1, 0.1, 1.0])
    tpr = np.array([0.0, 0.4, 0.9, 1.0])
    hull = upper_hull(fpr, tpr)
    kept = {(round(float(fpr[i]), 6), round(float(tpr[i]), 6)) for i in hull}
    assert (0.1, 0.9) in kept
    assert (0.1, 0.4) not in kept


# --------------------------------------------------------------------------------------
# The interpolation, which is the randomised rule.
# --------------------------------------------------------------------------------------


def test_a_budget_at_a_vertex_needs_no_coin() -> None:
    """A degenerate mixture -- weight 0 or 1 -- is a plain threshold with extra steps."""
    hull = upper_hull(FPR, TPR)
    detection, _, weight = hull_detection_at(FPR, TPR, hull, 0.1)
    assert detection == pytest.approx(0.5)
    assert weight == pytest.approx(0.0) or weight == pytest.approx(1.0)


def test_a_budget_between_vertices_interpolates_linearly() -> None:
    """Halfway between two hull vertices, a fair coin reaches the midpoint exactly."""
    hull = upper_hull(FPR, TPR)
    detection, _, weight = hull_detection_at(FPR, TPR, hull, 0.55)
    expected = 0.5 + (0.55 - 0.1) / (1.0 - 0.1) * (1.0 - 0.5)
    assert detection == pytest.approx(expected)
    assert 0.0 < weight < 1.0


def test_a_budget_beyond_the_curve_saturates() -> None:
    hull = upper_hull(FPR, TPR)
    detection, _, _ = hull_detection_at(FPR, TPR, hull, 5.0)
    assert detection == pytest.approx(1.0)


def test_a_plain_threshold_never_exceeds_its_budget() -> None:
    thresholds = np.array([1.0, 0.8, 0.4, 0.0])
    for budget in (0.05, 0.1, 0.3, 0.9):
        _, realised, _ = threshold_detection_at(FPR, TPR, thresholds, budget)
        assert realised <= budget + 1e-12


def test_a_plain_threshold_takes_the_best_point_inside_the_budget() -> None:
    thresholds = np.array([1.0, 0.8, 0.4, 0.0])
    detection, realised, cut = threshold_detection_at(FPR, TPR, thresholds, 0.6)
    assert detection == pytest.approx(0.6)
    assert realised == pytest.approx(0.5)
    assert cut == pytest.approx(0.4)


def test_the_hull_is_never_worse_than_a_threshold_at_the_same_budget() -> None:
    """The dominance claim, as an invariant rather than as a table."""
    thresholds = np.array([1.0, 0.8, 0.4, 0.0])
    hull = upper_hull(FPR, TPR)
    for budget in (0.05, 0.1, 0.3, 0.55, 0.9):
        plain, _, _ = threshold_detection_at(FPR, TPR, thresholds, budget)
        best, _, _ = hull_detection_at(FPR, TPR, hull, budget)
        assert best >= plain - 1e-12


# --------------------------------------------------------------------------------------
# Net benefit and normalised cost, against their closed forms.
# --------------------------------------------------------------------------------------


def test_alerting_on_everything_has_the_closed_form_net_benefit() -> None:
    """NB(all) = prevalence - (1 - prevalence) * odds, exactly."""
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # prevalence 0.2
    scores = np.ones(10)
    for probability in (0.1, 0.25, 0.5):
        odds = probability / (1 - probability)
        assert net_benefit(y, scores, probability) == pytest.approx(0.2 - 0.8 * odds)


def test_a_perfect_classifier_reaches_the_prevalence() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert net_benefit(y, y.astype(float), 0.5) == pytest.approx(0.2)


def test_net_benefit_is_defined_at_the_edges() -> None:
    y = np.array([1, 0])
    assert net_benefit(y, np.ones(2), 0.0) == 0.0
    assert net_benefit(y, np.ones(2), 1.0) == 0.0


def test_normalised_cost_reduces_to_its_axes() -> None:
    assert normalised_cost(0.3, 0.8, 0.0) == pytest.approx(0.3)  # only false positives count
    assert normalised_cost(0.3, 0.8, 1.0) == pytest.approx(0.2)  # only misses count


def test_a_perfect_point_costs_nothing_at_any_skew() -> None:
    for skew in (0.0, 0.25, 0.5, 1.0):
        assert normalised_cost(0.0, 1.0, skew) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Wiring.
# --------------------------------------------------------------------------------------


def test_roc_points_returns_one_threshold_per_operating_point() -> None:
    y = np.array([0, 0, 1, 1])
    fpr, tpr, thresholds = roc_points(y, np.array([0.1, 0.4, 0.35, 0.8]))
    assert len(fpr) == len(tpr) == len(thresholds)
    assert fpr[0] == 0.0 and tpr[-1] == 1.0


def test_the_benefit_row_picks_a_winner_at_each_base_rate() -> None:
    row = BenefitRow(
        threshold_probability=0.1,
        model=0.05,
        alert_on_everything=0.09,
        alert_on_nothing=0.0,
        model_production=0.02,
        everything_production=-0.4,
    )
    assert row.best == "alert on everything"
    assert row.best_production == "the model"


def test_a_policy_that_loses_to_doing_nothing_is_named() -> None:
    row = BenefitRow(
        threshold_probability=0.9,
        model=-0.1,
        alert_on_everything=-6.0,
        alert_on_nothing=0.0,
        model_production=-0.1,
        everything_production=-6.0,
    )
    assert row.best == "alert on nothing"
