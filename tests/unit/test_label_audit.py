"""Label-audit tests: the class-conditional suspect rule, the recovery arithmetic
against planted ground truth, and the flip-planting bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.evaluation.label_audit import (
    _plant_flips,
    audit_labels,
    recovery_metrics,
)


def test_audit_flags_benign_rows_scoring_like_attacks() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.6])
    findings = audit_labels(y, scores)
    assert findings.t_attack == pytest.approx(0.7)  # mean of the attack-labeled scores
    assert findings.t_benign == pytest.approx(0.4)
    assert findings.suspect_benign.tolist() == [2]  # 0.9 >= 0.7: scores like an attack
    assert findings.suspect_attack.tolist() == []  # no attack row scores <= 0.4


def test_audit_flags_attack_rows_scoring_like_benign() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.15, 0.9])
    findings = audit_labels(y, scores)
    # t_benign = 0.2; the attack row at 0.15 looks benign.
    assert findings.suspect_attack.tolist() == [2]


def test_recovery_metrics_precision_and_recall() -> None:
    flagged = np.array([1, 2, 3, 4])
    planted = np.array([2, 4, 9])
    precision, recall = recovery_metrics(flagged, planted)
    assert precision == pytest.approx(2 / 4)
    assert recall == pytest.approx(2 / 3)


def test_recovery_metrics_degenerate_sets_are_nan() -> None:
    precision, recall = recovery_metrics(np.array([], dtype=int), np.array([], dtype=int))
    assert np.isnan(precision) and np.isnan(recall)


def test_plant_flips_relabels_both_targets_and_reports_positions() -> None:
    frame = pd.DataFrame(
        {
            BINARY_TARGET: [1, 1, 1, 1, 0, 0],
            MULTICLASS_TARGET: ["DoS Hulk"] * 4 + ["BENIGN"] * 2,
            "Flow Duration": range(6),
        }
    )
    flipped, planted = _plant_flips(frame, rate=0.5, benign_label="BENIGN", seed=7)
    assert len(planted) == 2
    assert (flipped.iloc[planted][BINARY_TARGET] == 0).all()
    assert (flipped.iloc[planted][MULTICLASS_TARGET] == "BENIGN").all()
    # The original frame is untouched, and only attack rows were eligible.
    assert frame[BINARY_TARGET].sum() == 4
    assert set(planted.tolist()) <= {0, 1, 2, 3}
