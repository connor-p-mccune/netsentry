"""The factorial, the stressors, and the two failure shapes the design exists to name.

The interaction arithmetic is the part worth pinning: `both - first - second + neither` is easy
to write with a sign error, and a sign error would turn "the monitors saturate" into "the
monitors amplify" without anything else in the report changing.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.robustness.composition import (
    STRESSORS,
    Cell,
    CompositionStudy,
    Interaction,
    Scenario,
    apply_evasion,
    apply_outage,
    apply_rarity,
    apply_shift,
    scenarios,
)


def _cell(
    active: set[str],
    *,
    detection: float = 0.2,
    coverage: float = 0.95,
    fpr: float = 0.008,
    alert: float = 0.05,
    feature_psi: float = 0.05,
    score_psi: float = 0.05,
) -> Cell:
    return Cell(
        scenario=Scenario(frozenset(active)),
        rows=100,
        prevalence=0.25,
        realised_fpr=fpr,
        detection=detection,
        coverage=coverage,
        alert_rate=alert,
        feature_psi=feature_psi,
        score_psi=score_psi,
        budget=0.015,
        coverage_target=0.90,
        alert_ceiling=0.20,
        psi_threshold=0.2,
        detection_floor=0.10,
    )


# --------------------------------------------------------------------------------------
# The design.
# --------------------------------------------------------------------------------------


def test_the_factorial_has_a_cell_for_every_combination() -> None:
    assert len(scenarios(STRESSORS)) == 2 ** len(STRESSORS)


def test_the_cells_are_ordered_by_how_much_is_wrong() -> None:
    orders = [cell.order for cell in scenarios(("a", "b", "c"))]
    assert orders == sorted(orders)
    assert orders[0] == 0


def test_the_empty_scenario_is_named_rather_than_blank() -> None:
    assert Scenario(frozenset()).name == "nothing wrong"
    assert Scenario(frozenset({"b", "a"})).name == "a + b"


# --------------------------------------------------------------------------------------
# The stressors.
# --------------------------------------------------------------------------------------


def test_a_shift_keeps_only_the_latest_slice() -> None:
    x = np.arange(20.0).reshape(-1, 1)
    y = np.zeros(20, dtype=int)
    days = np.arange(20.0)
    kept_x, kept_y = apply_shift(x, y, days, 0.25)
    assert len(kept_y) == pytest.approx(6, abs=2)
    assert float(kept_x.min()) > 10.0


def test_a_shift_with_mismatched_days_changes_nothing() -> None:
    """A split without a day column must degrade to a no-op rather than to a wrong answer."""
    x = np.arange(6.0).reshape(-1, 1)
    y = np.zeros(6, dtype=int)
    kept_x, kept_y = apply_shift(x, y, np.array([]), 0.25)
    assert kept_x.shape == x.shape and len(kept_y) == 6


def test_an_outage_substitutes_the_training_median() -> None:
    x = np.ones((4, 3))
    damaged = apply_outage(x, np.array([1]), np.array([9.0, 5.0, 9.0]))
    assert np.all(damaged[:, 1] == 5.0)
    assert np.all(damaged[:, [0, 2]] == 1.0)


def test_an_outage_of_nothing_leaves_the_matrix_alone() -> None:
    x = np.ones((4, 3))
    assert np.array_equal(apply_outage(x, np.array([], dtype=int), np.zeros(3)), x)


def test_evasion_moves_attacks_only() -> None:
    x = np.zeros((4, 2))
    y = np.array([1, 0, 1, 0])
    shaped = apply_evasion(x, y, np.array([10.0, 10.0]), np.array([0]), 0.5)
    assert np.all(shaped[y == 1, 0] == 5.0)
    assert np.all(shaped[y == 0, 0] == 0.0)


def test_evasion_moves_only_controllable_features() -> None:
    x = np.zeros((2, 2))
    shaped = apply_evasion(x, np.array([1, 1]), np.array([10.0, 10.0]), np.array([0]), 1.0)
    assert np.all(shaped[:, 1] == 0.0)


def test_rarity_thins_attacks_to_the_target_base_rate() -> None:
    x = np.arange(200.0).reshape(-1, 1)
    y = np.array([1] * 100 + [0] * 100)
    _, thinned = apply_rarity(x, y, 0.05, np.random.default_rng(0))
    assert float(np.mean(thinned)) == pytest.approx(0.05, abs=0.02)


def test_rarity_never_invents_attacks() -> None:
    """Asking for a higher base rate than exists must leave the sample alone, not upsample it."""
    x = np.arange(20.0).reshape(-1, 1)
    y = np.array([1] * 2 + [0] * 18)
    _, kept = apply_rarity(x, y, 0.5, np.random.default_rng(1))
    assert int(np.sum(kept)) == 2


# --------------------------------------------------------------------------------------
# Guarantees and alarms.
# --------------------------------------------------------------------------------------


def test_a_healthy_cell_breaks_nothing_and_says_nothing() -> None:
    cell = _cell(set())
    assert cell.broken == [] and cell.alarms == [] and not cell.silent


def test_detection_below_the_floor_is_broken() -> None:
    assert "detection" in _cell(set(), detection=0.05).broken


def test_coverage_below_the_promise_is_broken() -> None:
    assert "coverage" in _cell(set(), coverage=0.5).broken


def test_a_breach_with_no_alarm_is_a_silent_failure() -> None:
    """The failure shape the design exists to find: broken guarantee, green dashboard."""
    assert _cell(set(), detection=0.05).silent


def test_a_breach_with_an_alarm_is_not_silent() -> None:
    assert not _cell(set(), detection=0.05, score_psi=0.5).silent


def test_a_monitor_at_exactly_the_threshold_fires() -> None:
    assert "feature PSI" in _cell(set(), feature_psi=0.2).alarms


# --------------------------------------------------------------------------------------
# The interaction arithmetic.
# --------------------------------------------------------------------------------------


def test_a_purely_additive_pair_has_no_interaction() -> None:
    row = Interaction("m", "a", "b", alone_first=0.3, alone_second=0.4, together=0.5, baseline=0.2)
    assert row.additive == pytest.approx(0.5)
    assert row.interaction == pytest.approx(0.0)


def test_a_saturating_monitor_has_a_negative_interaction() -> None:
    """Two failures moving a statistic less than their sum is the masking mechanism."""
    row = Interaction("m", "a", "b", alone_first=0.4, alone_second=0.4, together=0.5, baseline=0.0)
    assert row.interaction < 0


def test_a_compound_failure_is_more_than_twice_either_part() -> None:
    row = Interaction(
        "m", "a", "b", alone_first=0.01, alone_second=0.01, together=0.9, baseline=0.0
    )
    assert row.compound


def test_a_pair_dominated_by_one_part_is_not_compound() -> None:
    row = Interaction("m", "a", "b", alone_first=0.8, alone_second=0.0, together=0.8, baseline=0.0)
    assert not row.compound


# --------------------------------------------------------------------------------------
# The record.
# --------------------------------------------------------------------------------------


def _study(cells: list[Cell], interactions: list[Interaction] | None = None) -> CompositionStudy:
    return CompositionStudy(
        cells=cells,
        interactions=interactions or [],
        stressors=("a", "b"),
        n_features=10,
        outage_features=["f"],
    )


def test_a_compound_failure_is_one_no_single_stressor_causes() -> None:
    cells = [
        _cell(set()),
        _cell({"a"}),
        _cell({"b"}),
        _cell({"a", "b"}, detection=0.01),
    ]
    study = _study(cells)
    assert [cell.scenario.name for cell in study.compound_failures()] == ["a + b"]


def test_a_failure_that_a_single_stressor_already_causes_is_not_compound() -> None:
    cells = [
        _cell(set()),
        _cell({"a"}, detection=0.01),
        _cell({"b"}),
        _cell({"a", "b"}, detection=0.01),
    ]
    assert _study(cells).compound_failures() == []


def test_an_invisible_stressor_costs_detection_without_tripping_anything() -> None:
    cells = [_cell(set(), detection=0.20), _cell({"a"}, detection=0.14)]
    assert [cell.scenario.name for cell in _study(cells).invisible()] == ["a"]


def test_a_false_alarm_trips_a_monitor_without_breaking_anything() -> None:
    cells = [_cell(set()), _cell({"a"}, score_psi=0.5)]
    assert [cell.scenario.name for cell in _study(cells).false_alarms()] == ["a"]


def test_a_reversal_improves_one_guarantee_while_degrading_another() -> None:
    cells = [_cell(set(), detection=0.2, coverage=0.6), _cell({"a"}, detection=0.1, coverage=0.8)]
    assert [cell.scenario.name for cell in _study(cells).reversals()] == ["a"]


def test_the_subadditive_share_counts_only_monitor_readings() -> None:
    rows = [
        Interaction("score PSI", "a", "b", 0.4, 0.4, 0.5, 0.0),
        Interaction("feature PSI", "a", "b", 0.4, 0.4, 0.9, 0.0),
        Interaction("detection rate", "a", "b", 0.4, 0.4, 0.1, 0.0),
    ]
    assert _study([_cell(set())], rows).subadditive_share() == pytest.approx(0.5)


def test_a_study_with_no_monitor_interactions_reports_zero() -> None:
    assert _study([_cell(set())], []).subadditive_share() == 0.0
