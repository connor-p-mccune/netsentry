"""Release-gate bars: the leak re-check and the pass/fail policy logic."""

from __future__ import annotations

from netsentry.config import Settings
from netsentry.evaluation.gate import (
    GateCheck,
    GateResult,
    leaked_feature_names,
    performance_checks,
)


def _bars(settings: Settings, **overrides: float) -> list[GateCheck]:
    measured = {
        "pr_auc_value": 0.5,
        "prevalence": 0.25,
        "tpr_primary": 0.09,
        "primary_fpr": 0.001,
        "ece": 0.02,
    }
    measured.update(overrides)
    return performance_checks(settings.gate, **measured)  # type: ignore[arg-type]


def _by_name(checks: list[GateCheck], name: str) -> GateCheck:
    return next(c for c in checks if c.name == name)


def test_clean_feature_space_reports_no_leaks() -> None:
    names = ["Flow Duration", "Flow Bytes/s", "SYN Flag Count"]
    assert leaked_feature_names(names, port_allowed=False) == []


def test_surviving_identifier_and_encoded_port_are_caught() -> None:
    names = ["Flow Duration", "Timestamp", "Destination Port_80"]
    leaks = leaked_feature_names(names, port_allowed=False)
    assert "Timestamp" in leaks
    assert "Destination Port" in leaks  # the encoded variant still trips the check


def test_port_is_permitted_only_when_explicitly_encoded() -> None:
    names = ["Destination Port_443"]
    assert leaked_feature_names(names, port_allowed=True) == []
    assert leaked_feature_names(names, port_allowed=False) == ["Destination Port"]


def test_healthy_candidate_passes_every_bar(settings: Settings) -> None:
    assert all(c.passed for c in _bars(settings))


def test_pr_auc_floor_is_relative_to_prevalence(settings: Settings) -> None:
    # A random ranker's PR-AUC equals the prevalence; below lift x prevalence fails.
    failing = _by_name(_bars(settings, pr_auc_value=0.30, prevalence=0.25), "PR-AUC floor")
    assert not failing.passed
    passing = _by_name(_bars(settings, pr_auc_value=0.30, prevalence=0.10), "PR-AUC floor")
    assert passing.passed  # same score clears the bar at a lower base rate


def test_too_good_a_score_fails_the_gate(settings: Settings) -> None:
    check = _by_name(_bars(settings, pr_auc_value=0.9995), "too-good-to-be-true ceiling")
    assert not check.passed
    assert "leakage" in check.detail  # the gate names the suspected cause


def test_detection_and_calibration_floors(settings: Settings) -> None:
    assert not _by_name(_bars(settings, tpr_primary=0.01), "detection floor").passed
    assert not _by_name(_bars(settings, ece=0.5), "calibration quality").passed


def test_result_fails_when_any_check_fails() -> None:
    result = GateResult(
        [GateCheck("a", True, ""), GateCheck("b", False, ""), GateCheck("c", True, "")]
    )
    assert not result.ok
    assert result.n_failed == 1
    assert GateResult([GateCheck("a", True, "")]).ok
