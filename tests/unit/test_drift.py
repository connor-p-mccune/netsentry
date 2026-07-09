"""Drift detection (PSI) correctness and the rolling serving monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.monitoring.drift import (
    DriftReport,
    classify_psi,
    compute_drift_report,
    feature_drift,
    population_stability_index,
    quantile_bin_edges,
)
from netsentry.monitoring.monitor import DriftMonitor, reference_summary


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_psi_zero_for_exact_same_array() -> None:
    x = np.linspace(0.0, 100.0, 1000)
    assert population_stability_index(x, x, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_psi_near_zero_for_same_distribution() -> None:
    rng = _rng()
    ref, cur = rng.normal(size=5000), rng.normal(size=5000)
    assert population_stability_index(ref, cur, bins=10) < 0.1


def test_psi_flags_major_shift() -> None:
    rng = _rng()
    ref = rng.normal(0.0, 1.0, size=5000)
    cur = rng.normal(3.0, 1.0, size=5000)  # shifted three sigma
    psi = population_stability_index(ref, cur, bins=10)
    assert psi >= 0.25
    assert classify_psi(psi) == "major"


def test_classify_psi_thresholds() -> None:
    assert classify_psi(0.05) == "none"
    assert classify_psi(0.1) == "moderate"
    assert classify_psi(0.24) == "moderate"
    assert classify_psi(0.25) == "major"


def test_quantile_bin_edges_open_outer_and_constant_feature() -> None:
    edges = quantile_bin_edges(np.arange(100.0), bins=10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    const = quantile_bin_edges(np.full(50, 7.0), bins=10)  # must not crash
    assert const[0] == -np.inf and const[-1] == np.inf and const.size >= 2


def test_constant_reference_feature_still_registers_drift() -> None:
    """Regression (found by the property suite): a constant reference feature —
    e.g. the always-zero bulk columns — must not be drift-blind. Total migration
    off the constant reads as major drift, in either direction; staying on it
    reads as zero."""
    const = np.full(200, 0.0)
    assert population_stability_index(const, const) == 0.0
    assert population_stability_index(const, const + 5.0) >= 0.25
    assert population_stability_index(const, const - 5.0) >= 0.25


def test_two_valued_reference_feature_still_registers_drift() -> None:
    """Near-binary features (flag counts) collapse to two quantile values; a
    migration to a third value must still register."""
    ref = np.array([0.0, 1.0] * 100)
    assert population_stability_index(ref, ref) == 0.0
    assert population_stability_index(ref, np.full(200, 7.0)) >= 0.25
    # Mass shifting between the two observed values also registers.
    assert population_stability_index(ref, np.full(200, 1.0)) >= 0.25


def test_feature_drift_only_shared_columns() -> None:
    rng = _rng()
    ref = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    cur = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(3.0, 1.0, size=2000)})
    psi = feature_drift(ref, cur, ["a", "b", "missing"], bins=10)
    assert set(psi) == {"a", "b"}  # 'missing' silently skipped
    assert psi["b"] > psi["a"]  # only b shifted


def test_drift_report_aggregates_and_ranks() -> None:
    report = DriftReport(feature_psi={"a": 0.02, "b": 0.40, "c": 0.15})
    assert report.max_psi == pytest.approx(0.40)
    assert report.mean_psi == pytest.approx((0.02 + 0.40 + 0.15) / 3)
    assert report.drifted(level="moderate") == ["b", "c"]  # >= 0.1, worst first
    assert report.drifted(level="major") == ["b"]
    assert report.classify(0.40) == "major"


def test_compute_drift_report_includes_score_drift() -> None:
    rng = _rng()
    ref = pd.DataFrame({"a": rng.normal(size=2000)})
    cur = pd.DataFrame({"a": rng.normal(size=2000)})
    score_ref = rng.uniform(0.0, 0.3, size=2000)
    score_cur = rng.uniform(0.7, 1.0, size=2000)  # model scores shifted high
    report = compute_drift_report(
        ref, cur, ["a"], bins=10, score_reference=score_ref, score_current=score_cur
    )
    assert report.score_psi is not None and report.score_psi >= 0.25


def test_monitor_buffers_until_window_then_reports_and_resets() -> None:
    rng = _rng()
    ref = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    monitor = DriftMonitor.from_summary(
        reference_summary(ref, ["a", "b"], bins=10), window=400, moderate=0.1, major=0.25
    )

    # Below the window: nothing reported yet.
    partial = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    assert monitor.observe(partial) is None

    # Completing the window with a shifted 'b' yields a report flagging it.
    shifted = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(3.0, 1.0, size=400)})
    report = monitor.observe(shifted)
    assert report is not None
    assert report.feature_psi["b"] > report.feature_psi["a"]

    # The window reset: a fresh small observation reports nothing again.
    assert monitor.observe(pd.DataFrame({"a": [0.1], "b": [0.1]})) is None
