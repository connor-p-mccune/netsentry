"""LODO fold-metric tests: the NaN semantics matter — a benign-only day has no
detection to measure and PR-AUC is undefined with one class, and reporting 0.0 there
would silently read as 'caught nothing'."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.lodo import fold_metrics


def test_fold_metrics_on_a_mixed_day() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    pr_auc, detection, fpr = fold_metrics(y, scores, threshold=0.5)
    assert detection == 0.5  # one of two attacks clears the threshold
    assert fpr == 0.5  # one of two benign flows is flagged
    # AP by hand: positives at ranks 2 and 3 -> (1/2 + 2/3) / 2.
    assert pr_auc == pytest.approx(7 / 12)


def test_fold_metrics_on_a_benign_only_day() -> None:
    y = np.zeros(4, dtype=int)
    scores = np.array([0.1, 0.2, 0.9, 0.3])
    pr_auc, detection, fpr = fold_metrics(y, scores, threshold=0.5)
    assert np.isnan(pr_auc) and np.isnan(detection)
    assert fpr == 0.25  # the false-alarm audit is still well-defined


def test_fold_metrics_with_no_benign_flows() -> None:
    y = np.ones(3, dtype=int)
    scores = np.array([0.9, 0.1, 0.8])
    pr_auc, detection, fpr = fold_metrics(y, scores, threshold=0.5)
    assert np.isnan(pr_auc) and np.isnan(fpr)
    assert detection == pytest.approx(2 / 3)
