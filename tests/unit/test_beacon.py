"""Beaconing detector: regularity math, ranking, and the synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.config import Settings
from netsentry.intel.beacon import (
    coefficient_of_variation,
    detect_beacons,
    inter_arrival_times,
    regularity_score,
    render_report,
    synthesize_beacon_flows,
)


def test_perfectly_periodic_series_scores_one() -> None:
    iats = inter_arrival_times(np.arange(0, 600, 60, dtype=float))
    assert regularity_score(iats) == 1.0
    assert coefficient_of_variation(iats) == 0.0


def test_bursty_series_scores_well_below_the_flag_line() -> None:
    rng = np.random.default_rng(0)
    # Memoryless (exponential) gaps score moderate under a robust MAD/median
    # dispersion (~0.37) — the point is separation: comfortably under the 0.85 flag
    # line, while a periodic beacon sits near 1.0.
    iats = rng.exponential(300.0, size=200)
    score = regularity_score(iats)
    assert score < 0.6
    assert score < regularity_score(inter_arrival_times(np.arange(0, 6000, 60, dtype=float)))


def test_jittered_beacon_still_scores_high() -> None:
    rng = np.random.default_rng(1)
    times = np.arange(0, 6000, 60, dtype=float) + rng.normal(0, 1.5, size=100)
    assert regularity_score(inter_arrival_times(times)) > 0.85


def test_empty_and_degenerate_inputs_are_safe() -> None:
    assert regularity_score(np.array([])) == 0.0
    assert regularity_score(np.array([0.0, 0.0])) == 0.0  # zero median -> not periodic
    assert coefficient_of_variation(np.array([])) == 0.0


def test_detect_ranks_the_planted_beacon_first() -> None:
    df = synthesize_beacon_flows(seed=7)
    candidates = detect_beacons(df, timestamp_column="Timestamp", min_events=8, by_port=True)
    assert candidates, "expected at least the beacon pair"
    top = candidates[0]
    assert top.destination == "45.77.12.9"  # the planted C2
    assert top.port == 8443
    assert top.score > 0.85
    assert 55.0 < top.median_interval_s < 65.0  # ~60 s cadence recovered


def test_detect_skips_pairs_below_min_events() -> None:
    df = pd.DataFrame(
        {
            "Src IP": ["a", "a", "b"],
            "Dst IP": ["x", "x", "y"],
            "Dst Port": [80, 80, 80],
            "Timestamp": [1.0, 61.0, 5.0],
        }
    )
    # a->x has 2 events, b->y has 1 — both below min_events=8.
    assert detect_beacons(df, timestamp_column="Timestamp", min_events=8, by_port=True) == []


def test_detector_recovers_order_from_timestamps_not_rows() -> None:
    # Rows are shuffled in the synthetic frame; the detector sorts by time internally.
    df = synthesize_beacon_flows(seed=3)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = detect_beacons(df, timestamp_column="Timestamp", min_events=8, by_port=True)
    b = detect_beacons(shuffled, timestamp_column="Timestamp", min_events=8, by_port=True)
    assert a[0].score == b[0].score
    assert a[0].pair == b[0].pair


def test_report_flags_and_renders() -> None:
    df = synthesize_beacon_flows(seed=7)
    candidates = detect_beacons(df, timestamp_column="Timestamp", min_events=8, by_port=True)
    report = render_report(candidates, threshold=0.85, top_n=20, demo=True)
    assert "# NetSentry — Beaconing" in report
    assert "45.77.12.9:8443" in report
    assert "[flag]" in report
    assert "hunt lead generator, not a verdict" in report


def test_run_beacon_demo_writes_report(tmp_path) -> None:
    settings = Settings()
    settings.paths.reports_dir = tmp_path / "reports"
    out = run_report_demo(settings)
    assert out.exists()
    assert out.name == "beacon_demo.md"


def run_report_demo(settings: Settings):
    from netsentry.intel.beacon import run_beacon_report

    return run_beacon_report(settings, demo=True)
