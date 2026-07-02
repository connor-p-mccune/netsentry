"""Streaming-simulation unit tests: stream ordering by capture day, the per-batch
operating-point helper, and the nan-safe mean used in the summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET
from netsentry.monitoring.streaming import _nanmean, _operating_point, order_stream


def test_order_stream_sorts_thursday_before_friday_stably() -> None:
    df = pd.DataFrame(
        {
            schema.DAY_COLUMN: ["Friday", "Thursday", "Friday", "Thursday"],
            "tag": [0, 1, 2, 3],
        },
        index=[10, 11, 12, 13],
    )
    ordered = order_stream(df)
    # Thursdays first (stable order preserves original relative order), then Fridays.
    assert ordered[schema.DAY_COLUMN].tolist() == ["Thursday", "Thursday", "Friday", "Friday"]
    assert ordered["tag"].tolist() == [1, 3, 0, 2]


def test_order_stream_without_day_column_is_identity() -> None:
    df = pd.DataFrame({BINARY_TARGET: [0, 1, 0]})
    pd.testing.assert_frame_equal(order_stream(df), df)


def test_operating_point_matches_manual_rates() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    pr_auc, detection = _operating_point(y, scores, threshold=0.5)
    assert detection == 1.0  # both attacks score >= 0.5
    assert 0.0 <= pr_auc <= 1.0


def test_operating_point_pr_auc_is_nan_for_single_class_batch() -> None:
    y = np.array([0, 0, 0])
    scores = np.array([0.2, 0.3, 0.4])
    pr_auc, detection = _operating_point(y, scores, threshold=0.5)
    assert np.isnan(pr_auc)  # PR-AUC undefined without both classes
    assert detection == 0.0


def test_nanmean_ignores_nans_and_empty() -> None:
    assert _nanmean([0.5, float("nan"), 0.7]) == 0.6
    assert _nanmean([]) == 0.0
    assert _nanmean([float("nan")]) == 0.0
