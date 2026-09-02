"""The breakdown point, the two robust calibrators, and what each one costs.

The breakdown property is the study's whole claim, so it is tested as an inequality rather than
illustrated: below the budget's share of the sample the attacker cannot reach the cut, above it
the cut lands wherever they put it. The two defences are tested on both halves of their trade --
robustness under attack *and* the bill on clean data -- because a calibrator that is robust by
being wrong is not a defence.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from netsentry.robustness.calibration_attack import (
    AttackPoint,
    BudgetCost,
    CalibrationAttackStudy,
    DefenceRow,
    breakdown_point,
    median_of_days,
    poisoned_threshold,
    trimmed_threshold,
)

BENIGN = np.linspace(0.0, 1.0, 1000)
BUDGET = 0.01


# --------------------------------------------------------------------------------------
# The breakdown point.
# --------------------------------------------------------------------------------------


def test_the_breakdown_point_is_the_budget() -> None:
    """A quantile breaks at its own tail mass, which is what the budget names."""
    assert breakdown_point(0.01) == pytest.approx(0.01)
    assert breakdown_point(0.001) == pytest.approx(0.001)


def test_a_tighter_budget_is_cheaper_to_attack() -> None:
    """The claim that makes the study worth writing, stated as a comparison."""
    assert breakdown_point(0.001) < breakdown_point(0.05)


def test_an_unpoisoned_sample_gives_the_plain_quantile() -> None:
    clean = poisoned_threshold(BENIGN, np.array([]), BUDGET)
    assert clean == pytest.approx(float(np.quantile(BENIGN, 0.99, method="higher")))


def test_injection_past_the_breakdown_point_hands_the_cut_to_the_attacker() -> None:
    injected = np.full(int(0.02 * len(BENIGN)), 5.0)
    assert poisoned_threshold(BENIGN, injected, BUDGET) == pytest.approx(5.0)


def test_injection_below_the_breakdown_point_does_not() -> None:
    injected = np.full(int(0.002 * len(BENIGN)), 5.0)
    assert poisoned_threshold(BENIGN, injected, BUDGET) < 5.0


def test_the_cut_rises_monotonically_with_the_injection() -> None:
    cuts = [poisoned_threshold(BENIGN, np.full(count, 5.0), BUDGET) for count in (0, 2, 5, 10, 20)]
    assert all(later >= earlier for earlier, later in pairwise(cuts))


# --------------------------------------------------------------------------------------
# The trimmed quantile: robust, and never free.
# --------------------------------------------------------------------------------------


def test_trimming_survives_contamination_smaller_than_the_trim() -> None:
    poisoned = np.concatenate([BENIGN, np.full(10, 5.0)])
    assert trimmed_threshold(poisoned, BUDGET, trim=0.05) < 5.0


def test_trimming_still_falls_to_contamination_larger_than_itself() -> None:
    poisoned = np.concatenate([BENIGN, np.full(200, 5.0)])
    assert trimmed_threshold(poisoned, BUDGET, trim=0.02) == pytest.approx(5.0)


def test_trimming_lowers_the_cut_on_clean_data() -> None:
    """The bill, paid every day on traffic nobody is attacking."""
    plain = poisoned_threshold(BENIGN, np.array([]), BUDGET)
    assert trimmed_threshold(BENIGN, BUDGET, trim=0.05) < plain


def test_trimming_everything_falls_back_rather_than_failing() -> None:
    assert np.isfinite(trimmed_threshold(BENIGN, BUDGET, trim=1.0))


def test_an_empty_sample_gives_an_unreachable_cut() -> None:
    """Better to alert on nothing than to alert on everything when calibration has no data."""
    assert trimmed_threshold(np.array([]), BUDGET, trim=0.02) == float("inf")


# --------------------------------------------------------------------------------------
# Median of days: free, total, and conditional.
# --------------------------------------------------------------------------------------


def test_a_median_over_days_outvotes_one_poisoned_day() -> None:
    scores = np.concatenate([BENIGN[:300], BENIGN[:300], np.full(300, 5.0)])
    days = np.repeat(np.array(["mon", "tue", "wed"]), 300)
    assert median_of_days(scores, days, BUDGET, minimum=10) < 5.0


def test_a_median_over_days_falls_when_every_day_is_poisoned() -> None:
    """Its breakdown point is in *days*, so an attacker who spreads out defeats it."""
    scores = np.tile(np.concatenate([BENIGN[:290], np.full(10, 5.0)]), 3)
    days = np.repeat(np.array(["mon", "tue", "wed"]), 300)
    assert median_of_days(scores, days, BUDGET, minimum=10) == pytest.approx(5.0)


def test_a_day_too_small_to_calibrate_is_skipped() -> None:
    scores = np.concatenate([BENIGN[:300], np.full(3, 5.0)])
    days = np.concatenate([np.repeat("mon", 300), np.repeat("tue", 3)])
    assert median_of_days(scores, days, BUDGET, minimum=50) < 5.0


def test_no_day_large_enough_falls_back_to_the_pooled_quantile() -> None:
    scores = BENIGN[:10]
    days = np.repeat("mon", 10)
    pooled = poisoned_threshold(scores, np.array([]), BUDGET)
    assert median_of_days(scores, days, BUDGET, minimum=500) == pytest.approx(pooled)


# --------------------------------------------------------------------------------------
# The records.
# --------------------------------------------------------------------------------------


def _point(detection: float = 0.15, psi: float = 0.01) -> AttackPoint:
    return AttackPoint(
        attacker="blind",
        fraction=0.01,
        injected=56,
        threshold=0.95,
        clean_threshold=0.87,
        detection=detection,
        clean_detection=0.20,
        realised_fpr=0.004,
        score_psi=psi,
        psi_threshold=0.2,
    )


def test_an_attack_point_reports_what_it_cost_the_detector() -> None:
    point = _point()
    assert point.detection_loss == pytest.approx(0.05)
    assert point.shift == pytest.approx(0.08)


def test_a_quiet_attack_is_the_one_worth_naming() -> None:
    assert not _point(psi=0.01).noticed
    assert _point(psi=0.5).noticed


def _defence(name: str, attacker: str, poisoned: float, clean_fpr: float = 0.017) -> DefenceRow:
    return DefenceRow(
        name=name,
        describes="",
        attacker=attacker,
        clean_threshold=0.8,
        clean_detection=0.20,
        clean_fpr=clean_fpr,
        poisoned_threshold=0.9,
        poisoned_detection=poisoned,
        budget=0.01,
    )


def test_a_defence_reports_the_share_of_detection_it_keeps() -> None:
    assert _defence("trim", "spread", 0.10).survived == pytest.approx(0.5)


def test_a_defence_reports_its_clean_bill() -> None:
    assert _defence("trim", "spread", 0.10, clean_fpr=0.017).clean_cost == pytest.approx(0.007)


def _study(defences: list[DefenceRow]) -> CalibrationAttackStudy:
    return CalibrationAttackStudy(
        points=[_point()],
        defences=defences,
        costs=[BudgetCost(0.01, 153, 56, 0.207)],
        budget=0.01,
        n_benign=5611,
        clean_threshold=0.87,
        clean_detection=0.20,
        deciding_flows=153,
    )


def test_a_rule_that_depends_on_the_attacker_shape_is_named() -> None:
    """Averaging over shapes would hide the case an operator actually faces."""
    study = _study(
        [
            _defence("median of days", "spread", 0.00),
            _defence("median of days", "confined", 0.19),
            _defence("trim", "spread", 0.07),
            _defence("trim", "confined", 0.07),
        ]
    )
    assert study.shape_dependent() == ["median of days"]


def test_the_best_defence_is_the_one_that_keeps_the_most() -> None:
    study = _study([_defence("median of days", "confined", 0.19), _defence("trim", "spread", 0.07)])
    assert study.best_defence().name == "median of days"


def test_a_budget_cost_carries_its_own_breakdown_point() -> None:
    assert BudgetCost(0.001, 9, 6, 0.09).breakdown == pytest.approx(0.001)
