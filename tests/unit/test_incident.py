"""Incident grouping: contiguity mechanics (fast) and the rendered report (slow)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.intel.incident import group_incidents


def test_consecutive_same_class_alerts_form_one_incident() -> None:
    classes = ["DDoS", "DDoS", "DDoS"]
    attacks = [True, True, True]
    assert group_incidents(classes, attacks, gap_tolerance=0) == [[0, 1, 2]]


def test_small_benign_gaps_are_bridged_and_large_ones_split() -> None:
    classes = ["DDoS", "BENIGN", "BENIGN", "DDoS", "BENIGN", "BENIGN", "BENIGN", "DDoS"]
    attacks = [True, False, False, True, False, False, False, True]
    # Two benign rows bridge at tolerance 2; the three-row gap splits.
    assert group_incidents(classes, attacks, gap_tolerance=2) == [[0, 3], [7]]
    # At tolerance 3 the whole run is one operation.
    assert group_incidents(classes, attacks, gap_tolerance=3) == [[0, 3, 7]]


def test_class_change_starts_a_new_incident() -> None:
    classes = ["DDoS", "DDoS", "PortScan", "PortScan"]
    attacks = [True, True, True, True]
    assert group_incidents(classes, attacks, gap_tolerance=5) == [[0, 1], [2, 3]]


def test_no_alerts_means_no_incidents() -> None:
    assert group_incidents(["BENIGN", "BENIGN"], [False, False], gap_tolerance=3) == []
    assert group_incidents([], [], gap_tolerance=3) == []


def test_non_alert_rows_never_join_an_incident() -> None:
    classes = ["DDoS", "BENIGN", "DDoS"]
    attacks = [True, False, True]
    (incident,) = group_incidents(classes, attacks, gap_tolerance=1)
    assert incident == [0, 2]  # the benign row is bridged over, not absorbed


@pytest.mark.slow
def test_incident_report_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.intel.incident import build_incident_report
    from netsentry.serving.bundle import build_serving_bundle

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    flows = clean_synth.head(80)
    in_path = tmp_path / "flows.csv"
    out_path = tmp_path / "incidents.md"
    flows.to_csv(in_path, index=False)

    stats = build_incident_report(settings, in_path, out_path, profile="fpr_1pct")
    assert stats["scored"] == 80
    assert stats["incidents"] >= 0
    report = out_path.read_text(encoding="utf-8")
    assert "# NetSentry — Incident Report" in report
    assert "80 flows scored" in report
    if stats["incidents"]:
        assert "### Incident 1:" in report
        assert "Recommended actions" in report
    else:
        assert "No incidents" in report
