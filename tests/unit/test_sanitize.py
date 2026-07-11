"""Poisoning defense: flip placement, audit-drop bookkeeping, outcome arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.evaluation.label_audit import audit_labels
from netsentry.robustness.sanitize import apply_flips, defense_outcome, flip_positions


def _labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            BINARY_TARGET: [0, 0, 0, 0, 1, 1, 1, 1],
            MULTICLASS_TARGET: ["BENIGN"] * 4 + ["DoS Hulk"] * 4,
            "Flow Duration": np.arange(8.0),
        }
    )


def test_flip_positions_only_targets_attack_rows() -> None:
    y = np.array([0, 1, 0, 1, 1, 0, 1])
    positions = flip_positions(y, rate=0.5, seed=7)
    assert len(positions) == 2  # int(4 * 0.5)
    assert all(y[p] == 1 for p in positions)


def test_flip_positions_is_seeded_and_sorted() -> None:
    y = np.array([1] * 20)
    first = flip_positions(y, rate=0.3, seed=42)
    second = flip_positions(y, rate=0.3, seed=42)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.sort(first))


def test_flip_positions_zero_rate_is_empty() -> None:
    assert len(flip_positions(np.array([1, 1, 1]), rate=0.0, seed=1)) == 0


def test_apply_flips_relabels_both_targets_and_copies() -> None:
    frame = _labeled_frame()
    flipped = apply_flips(frame, np.array([4, 5]), benign_label="BENIGN")
    assert list(flipped[BINARY_TARGET]) == [0, 0, 0, 0, 0, 0, 1, 1]
    assert list(flipped[MULTICLASS_TARGET][4:6]) == ["BENIGN", "BENIGN"]
    # The original frame is untouched (copy-safe), and features are untouched.
    assert list(frame[BINARY_TARGET]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert flipped["Flow Duration"].equals(frame["Flow Duration"])


def test_apply_flips_no_positions_returns_frame_unchanged() -> None:
    frame = _labeled_frame()
    assert apply_flips(frame, np.array([], dtype=int), "BENIGN") is frame


def test_defense_outcome_arithmetic() -> None:
    dropped = np.array([1, 2, 3, 7])
    planted = np.array([2, 3, 9])
    caught, clean_lost = defense_outcome(dropped, planted)
    assert caught == 2  # rows 2 and 3
    assert clean_lost == 2  # rows 1 and 7 were legitimate


def test_audit_drop_catches_score_separated_flips() -> None:
    # 8 clean benign rows scoring low, 2 flips (labeled benign, scoring like the
    # attacks), 5 genuine attacks scoring high. The audit's class-conditional
    # thresholds must flag exactly the flips, so the defense drops them and
    # loses no clean rows.
    y = np.array([0] * 10 + [1] * 5)
    scores = np.array([0.1] * 8 + [0.9, 0.9] + [0.9] * 5)
    findings = audit_labels(y, scores)
    dropped = np.union1d(findings.suspect_benign, findings.suspect_attack)
    caught, clean_lost = defense_outcome(dropped, planted=np.array([8, 9]))
    assert caught == 2
    assert clean_lost == 0
