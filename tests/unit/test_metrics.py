"""Metric correctness on hand-computed cases — a wrong metric invalidates all results."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.metrics import (
    alerts_per_day,
    binary_summary,
    confusion,
    operating_point,
    per_class_report,
    rates_at_threshold,
    tpr_at_fpr,
)

# A perfectly separable binary problem: 4 benign (low scores), 4 attack (high).
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])


def test_rates_at_threshold_known_values() -> None:
    # threshold 0.65 -> predict positive for {0.7, 0.8, 0.9}: TP=3, FN=1, FP=0, TN=4.
    rates = rates_at_threshold(Y, S, 0.65)
    assert (rates["tp"], rates["fp"], rates["fn"], rates["tn"]) == (3, 0, 1, 4)
    assert rates["tpr"] == pytest.approx(0.75)
    assert rates["fpr"] == pytest.approx(0.0)
    assert rates["precision"] == pytest.approx(1.0)


def test_tpr_at_fpr_perfect_separation() -> None:
    _threshold, tpr = tpr_at_fpr(Y, S, target_fpr=0.0)
    assert tpr == pytest.approx(1.0)  # all attacks caught at zero false positives


def test_binary_summary_perfect_separation() -> None:
    summary = binary_summary(Y, S)
    assert summary["pr_auc"] == pytest.approx(1.0)
    assert summary["roc_auc"] == pytest.approx(1.0)


def test_alerts_per_day_arithmetic() -> None:
    # 0.1% FPR over 1M flows/day = 1000 false alerts/day.
    assert alerts_per_day(0.001, 1_000_000, benign_fraction=1.0) == pytest.approx(1000.0)


def test_operating_point_uses_val_threshold_on_test() -> None:
    point = operating_point(Y, S, Y, S, target_fpr=0.0, flows_per_day=1_000_000)
    assert point["tpr"] == pytest.approx(1.0)
    assert point["fpr"] == pytest.approx(0.0)
    assert point["alerts_per_day"] == pytest.approx(0.0)


def test_per_class_report_known_confusion() -> None:
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    report = per_class_report(y_true, y_pred, labels=["a", "b"])
    # class a: precision 1/1=1.0, recall 1/2=0.5, f1=0.667
    assert report["a"]["precision"] == pytest.approx(1.0)
    assert report["a"]["recall"] == pytest.approx(0.5)
    assert report["a"]["f1"] == pytest.approx(2 / 3, abs=1e-3)
    # class b: precision 2/3, recall 2/2=1.0, f1=0.8
    assert report["b"]["precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["b"]["recall"] == pytest.approx(1.0)
    assert report["b"]["f1"] == pytest.approx(0.8, abs=1e-3)
    assert report["__macro__"]["f1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-3)


def test_confusion_orientation() -> None:
    y_true = np.array(["a", "a", "b"])
    y_pred = np.array(["a", "b", "b"])
    cm = confusion(y_true, y_pred, labels=["a", "b"])
    # rows = true, cols = pred: a->a once, a->b once, b->b once.
    assert cm.tolist() == [[1, 1], [0, 1]]
