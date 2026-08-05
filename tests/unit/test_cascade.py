"""Cascade inference: threshold selection, cascade ranking, and the latency algebra.

The cascade's correctness claims are all structural — the stage-1 cut must be derived from
the full model's alerts (not from traffic percentiles), a filtered flow must never outrank a
forwarded one, and the blended latency must be stage 1 plus the deferred share of stage 2.
Each is pinned here without training anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.serving.cascade import (
    blended_latency_ms,
    cascade_scores,
    median_latency_ms,
    stage1_threshold,
)


def test_full_retention_forwards_every_alert() -> None:
    stage1 = np.array([0.9, 0.1, 0.5, 0.4])
    alerts = np.array([True, True, False, False])
    # keep=1.0 must cut at or below the lowest-scoring alert (0.1), so both survive.
    cut = stage1_threshold(stage1, alerts, 1.0)
    assert cut <= 0.1
    assert (stage1 >= cut)[alerts].all()


def test_half_retention_cuts_at_the_alert_median() -> None:
    stage1 = np.array([0.1, 0.2, 0.8, 0.9])
    alerts = np.array([True, True, True, True])
    assert stage1_threshold(stage1, alerts, 0.5) == pytest.approx(0.5)


def test_threshold_is_derived_from_alerts_not_from_traffic() -> None:
    # 96 benign flows score high and 4 alerts score low. A traffic-percentile cut would
    # keep the benign mass and drop every alert; the alert-conditioned cut does the reverse.
    stage1 = np.concatenate([np.full(96, 0.9), np.array([0.1, 0.11, 0.12, 0.13])])
    alerts = np.concatenate([np.zeros(96, dtype=bool), np.ones(4, dtype=bool)])
    cut = stage1_threshold(stage1, alerts, 1.0)
    assert (stage1 >= cut)[alerts].all()
    assert cut <= 0.1


def test_threshold_with_no_alerts_forwards_everything() -> None:
    cut = stage1_threshold(np.array([0.3, 0.7]), np.zeros(2, dtype=bool), 0.9)
    assert cut == -np.inf


def test_filtered_flows_never_outrank_forwarded_ones() -> None:
    stage1 = np.array([0.99, 0.01, 0.5, 0.2])
    stage2 = np.array([0.10, 0.90, 0.30, 0.40])
    forwarded = np.array([True, False, True, False])
    out = cascade_scores(stage1, stage2, forwarded)
    # Flow 1's stage-2 score (0.90) is irrelevant: it never reached stage 2.
    assert out[forwarded].min() > out[~forwarded].max()


def test_forwarded_flows_keep_their_stage2_score_exactly() -> None:
    stage2 = np.array([0.1, 0.9, 0.3])
    forwarded = np.array([True, False, True])
    out = cascade_scores(np.array([0.5, 0.5, 0.5]), stage2, forwarded)
    assert out[0] == 0.1 and out[2] == 0.3


def test_filtered_flows_keep_the_cheap_models_ordering_among_themselves() -> None:
    stage1 = np.array([0.9, 0.1, 0.5])
    forwarded = np.array([True, False, False])
    out = cascade_scores(stage1, np.array([0.8, 0.0, 0.0]), forwarded)
    assert out[2] > out[1]  # 0.5 still ranks above 0.1 inside the filtered band


def test_cascade_scores_with_everything_forwarded_is_the_full_model() -> None:
    stage2 = np.array([0.2, 0.8, 0.5])
    out = cascade_scores(np.array([0.1, 0.2, 0.3]), stage2, np.ones(3, dtype=bool))
    assert np.allclose(out, stage2)


def test_blended_latency_is_stage1_plus_the_deferred_share_of_stage2() -> None:
    assert blended_latency_ms(1.0, 10.0, 0.0) == 1.0  # nothing deferred
    assert blended_latency_ms(1.0, 10.0, 1.0) == 11.0  # everything deferred, plus the filter
    assert blended_latency_ms(1.0, 10.0, 0.2) == pytest.approx(3.0)


def test_deferring_everything_is_slower_than_the_full_model_alone() -> None:
    # The honest accounting: a cascade that forwards everything pays for stage 1 twice over.
    assert blended_latency_ms(1.0, 10.0, 1.0) > 10.0


def test_median_latency_returns_zero_for_an_empty_matrix() -> None:
    assert median_latency_ms(lambda x: x, np.zeros((0, 3)), 10) == 0.0


def test_median_latency_measures_the_requested_number_of_calls() -> None:
    calls = []
    median_latency_ms(lambda row: calls.append(row.shape), np.zeros((50, 4)), 7)
    assert len(calls) == 7
    assert calls[0] == (1, 4)  # single-row, the serving unit of work
