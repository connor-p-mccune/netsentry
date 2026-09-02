"""The classifier, the audit's two error directions, and the splitting arithmetic.

The classification is what every number in the report rests on, so it is pinned feature by
feature rather than by counting. The splitting scale factors are the other load-bearing part:
flag counts deliberately do *not* divide, because a split session is several TCP connections and
each carries its own SYN -- the first version of this module divided them and overstated how much
splitting changes a feature vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.robustness.threat_model import (
    BACKWARD,
    ENVIRONMENTAL,
    FORWARD,
    JOINT,
    ModelRow,
    SplitRow,
    ThreatModelStudy,
    Verdict,
    audit,
    classify,
    split_flow,
    split_scaling,
)

# --------------------------------------------------------------------------------------
# Classification.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("Total Fwd Packets", FORWARD),
        ("Fwd Packet Length Max", FORWARD),
        ("Init_Win_bytes_forward", FORWARD),
        ("Total Backward Packets", BACKWARD),
        ("Bwd IAT Mean", BACKWARD),
        ("Init_Win_bytes_backward", BACKWARD),
        ("Flow Duration", JOINT),
        ("Packet Length Mean", JOINT),
        ("Down/Up Ratio", JOINT),
        ("Idle Mean", JOINT),
        ("Destination Port", ENVIRONMENTAL),
    ],
)
def test_each_feature_is_classified_by_who_produces_its_packets(
    feature: str, expected: str
) -> None:
    assert classify(feature) == expected


def test_a_transformed_column_name_is_stripped_before_classifying() -> None:
    """The pipeline prefixes columns; the classifier must see the underlying feature."""
    assert classify("numeric__Total Fwd Packets") == FORWARD


def test_classification_is_case_insensitive() -> None:
    assert classify("total fwd packets") == FORWARD


# --------------------------------------------------------------------------------------
# The audit's two error directions.
# --------------------------------------------------------------------------------------


def test_a_backward_feature_the_list_claims_is_over_claimed() -> None:
    verdict = next(row for row in audit(["Bwd IAT Mean"]) if row.feature == "Bwd IAT Mean")
    assert verdict.over_claimed and not verdict.under_claimed


def test_a_forward_feature_the_list_omits_is_under_claimed() -> None:
    """The dangerous direction: control the attacker has that the evaluation does not model."""
    verdict = next(row for row in audit([]) if row.feature == "Total Fwd Packets")
    assert verdict.under_claimed and not verdict.over_claimed


def test_a_forward_feature_the_list_claims_is_neither() -> None:
    verdict = next(
        row for row in audit(["Total Fwd Packets"]) if row.feature == "Total Fwd Packets"
    )
    assert not verdict.over_claimed and not verdict.under_claimed


def test_a_joint_feature_is_never_an_error_either_way() -> None:
    """Joint features are partially controllable, so claiming or omitting them is a judgement."""
    claimed = next(row for row in audit(["Flow Duration"]) if row.feature == "Flow Duration")
    omitted = next(row for row in audit([]) if row.feature == "Flow Duration")
    assert not (claimed.over_claimed or claimed.under_claimed)
    assert not (omitted.over_claimed or omitted.under_claimed)


def test_the_audit_covers_every_feature_exactly_once() -> None:
    verdicts = audit([])
    assert len({row.feature for row in verdicts}) == len(verdicts)


# --------------------------------------------------------------------------------------
# Splitting arithmetic.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature",
    ["Total Fwd Packets", "Flow Duration", "Total Length of Bwd Packets", "Subflow Fwd Bytes"],
)
def test_a_total_divides_among_the_pieces(feature: str) -> None:
    assert split_scaling(feature) == pytest.approx(1.0)


@pytest.mark.parametrize("feature", ["Flow Bytes/s", "Fwd Packet Length Mean", "Down/Up Ratio"])
def test_a_rate_or_a_mean_is_unchanged_by_splitting(feature: str) -> None:
    """The same bytes over the same seconds is the same bytes per second, however it is diced."""
    assert split_scaling(feature) == pytest.approx(0.0)


@pytest.mark.parametrize("feature", ["SYN Flag Count", "FIN Flag Count", "PSH Flag Count"])
def test_flag_counts_do_not_divide(feature: str) -> None:
    """A split session is several connections, and each one carries its own SYN."""
    assert split_scaling(feature) == pytest.approx(0.0)


def test_splitting_into_one_piece_changes_nothing() -> None:
    x = np.array([[100.0, 5.0]])
    assert np.array_equal(split_flow(x, np.array([0]), np.array([1.0, 0.0]), 1), x)


def test_splitting_divides_only_the_scaled_columns() -> None:
    x = np.array([[100.0, 5.0]])
    out = split_flow(x, np.array([0]), np.array([1.0, 0.0]), 4)
    assert out[0, 0] == pytest.approx(25.0)
    assert out[0, 1] == pytest.approx(5.0)


def test_splitting_does_not_mutate_its_input() -> None:
    x = np.array([[100.0, 5.0]])
    split_flow(x, np.array([0]), np.array([1.0, 0.0]), 4)
    assert x[0, 0] == pytest.approx(100.0)


def test_more_pieces_divide_further() -> None:
    x = np.array([[100.0]])
    four = split_flow(x, np.array([0]), np.array([1.0]), 4)[0, 0]
    sixteen = split_flow(x, np.array([0]), np.array([1.0]), 16)[0, 0]
    assert sixteen < four < 100.0


# --------------------------------------------------------------------------------------
# The record.
# --------------------------------------------------------------------------------------


def _split(pieces: int, detection: float) -> SplitRow:
    return SplitRow(
        pieces=pieces,
        detection=detection,
        clean_detection=0.20,
        alert_rate=0.058,
        realised_fpr=0.0082,
    )


def _study(splits: list[SplitRow]) -> ThreatModelStudy:
    return ThreatModelStudy(
        verdicts=[Verdict("Total Fwd Packets", FORWARD, True)],
        models=[ModelRow("shipped", "", 39, 0.14, 0.20)],
        splits=splits,
        clean_detection=0.20,
        budget=0.01,
    )


def test_an_arm_reports_what_the_attack_cost_the_detector() -> None:
    arm = ModelRow("shipped", "", 39, 0.14, 0.20)
    assert arm.kept == pytest.approx(0.7)
    assert arm.cost == pytest.approx(0.06)


def test_a_backfiring_attack_keeps_more_than_it_started_with() -> None:
    """The result the study actually found, so the arithmetic has to survive it."""
    arm = ModelRow("forward only", "", 24, 0.238, 0.207)
    assert arm.kept > 1.0
    assert arm.cost < 0.0


def test_splitting_that_never_helps_reports_one_piece_as_best() -> None:
    study = _study([_split(1, 0.207), _split(4, 0.27), _split(32, 0.30)])
    assert not study.splitting_helps()
    assert study.best_for_attacker().pieces == 1
    assert study.most_split().pieces == 32


def test_splitting_that_helps_is_reported_as_such() -> None:
    study = _study([_split(1, 0.207), _split(4, 0.10), _split(32, 0.15)])
    assert study.splitting_helps()
    assert study.best_for_attacker().pieces == 4
