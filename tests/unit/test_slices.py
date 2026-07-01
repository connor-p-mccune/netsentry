"""Per-attack-class detection slices."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.slices import per_class_detection


def test_per_class_detection_excludes_benign_and_counts_support() -> None:
    labels = np.array(["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "PortScan"])
    scores = np.array([0.1, 0.9, 0.9, 0.2, 0.95])
    slices = per_class_detection(labels, scores, threshold=0.5, benign_label="BENIGN")

    by_label = {s.label: s for s in slices}
    assert "BENIGN" not in by_label  # benign is not an attack class
    assert by_label["DoS Hulk"].support == 2
    assert by_label["DoS Hulk"].detection == 0.5  # one of two >= 0.5
    assert by_label["PortScan"].detection == 1.0


def test_detection_is_zero_when_all_below_threshold() -> None:
    labels = np.array(["Bot", "Bot"])
    scores = np.array([0.1, 0.2])
    slices = per_class_detection(labels, scores, threshold=0.5, benign_label="BENIGN")
    assert slices[0].label == "Bot"
    assert slices[0].detection == 0.0


def test_absent_class_not_reported() -> None:
    labels = np.array(["BENIGN", "PortScan"])
    scores = np.array([0.1, 0.9])
    slices = per_class_detection(labels, scores, threshold=0.5, benign_label="BENIGN")
    assert [s.label for s in slices] == ["PortScan"]
