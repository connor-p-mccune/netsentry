"""Spool watcher: ECS mapping, network enrichment, and exactly-once state."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netsentry.serving.watch import WatchState, _severity, scan_spool, to_ecs_alert


def _prediction(**overrides: object) -> dict[str, object]:
    base = {
        "predicted_class": "DDoS",
        "is_attack": True,
        "attack_probability": 0.97,
        "anomaly_score": 0.4,
        "is_anomaly": False,
        "recommended_action": "auto_alert",
        "top_feature": "Flow Duration",
    }
    base.update(overrides)
    return base


def test_severity_maps_probability_to_0_100() -> None:
    assert _severity(0.0) == 0
    assert _severity(1.0) == 100
    assert _severity(0.973) == 97
    assert _severity(1.5) == 100  # clamped
    assert _severity(-0.2) == 0


def test_to_ecs_alert_has_the_core_ecs_envelope() -> None:
    doc = to_ecs_alert(
        _prediction(),
        context={},
        source_file="/spool/flows_001.csv",
        timestamp="2026-07-11T00:00:00+00:00",
        model_version="0.4.0",
        threshold_profile="fpr_0.1pct",
    )
    assert doc["@timestamp"] == "2026-07-11T00:00:00+00:00"
    assert doc["ecs"] == {"version": "8.11"}
    assert doc["event"]["kind"] == "alert"
    assert "intrusion_detection" in doc["event"]["category"]
    assert doc["event"]["severity"] == 97
    assert doc["event"]["risk_score"] == 97.0
    assert doc["rule"]["name"] == "DDoS"
    assert doc["netsentry"]["threshold_profile"] == "fpr_0.1pct"
    assert doc["netsentry"]["model_version"] == "0.4.0"
    assert doc["log"]["file"]["path"] == "/spool/flows_001.csv"


def test_to_ecs_alert_maps_mitre_when_known() -> None:
    doc = to_ecs_alert(
        _prediction(predicted_class="PortScan"),
        context={},
        source_file="x.csv",
        timestamp="t",
        model_version="0.4.0",
    )
    # PortScan maps to Discovery / T1046 in the shared attack mapping.
    assert doc["threat"]["framework"] == "MITRE ATT&CK"
    assert doc["threat"]["technique"]["id"] == "T1046"


def test_to_ecs_alert_enriches_network_from_capture_metadata() -> None:
    context = {
        "Src IP": "10.0.0.5",
        "Src Port": 44321.0,
        "Dst IP": "10.0.0.9",
        "Dst Port": 80.0,
        "Protocol": 6.0,
    }
    doc = to_ecs_alert(
        _prediction(),
        context=context,
        source_file="x.csv",
        timestamp="t",
        model_version="0.4.0",
    )
    assert doc["source"] == {"ip": "10.0.0.5", "port": 44321}
    assert doc["destination"] == {"ip": "10.0.0.9", "port": 80}
    assert doc["network"] == {"iana_number": 6}


def test_to_ecs_alert_skips_absent_and_nan_metadata() -> None:
    doc = to_ecs_alert(
        _prediction(),
        context={"Src IP": "10.0.0.5", "Dst Port": np.nan},
        source_file="x.csv",
        timestamp="t",
        model_version="0.4.0",
    )
    assert doc["source"] == {"ip": "10.0.0.5"}
    assert "destination" not in doc  # the only dst field was NaN
    assert "network" not in doc


def test_watch_state_round_trips_and_detects_changes(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    f = spool / "flows.csv"
    f.write_text("Flow Duration\n1200\n", encoding="utf-8")

    state = WatchState()
    assert not state.is_seen(f)
    state.mark(f)
    assert state.is_seen(f)

    state_path = tmp_path / "state.json"
    state.save(state_path)
    reloaded = WatchState.load(state_path)
    assert reloaded.is_seen(f)

    # Appending to the file changes its size/mtime, so it is seen as new again.
    f.write_text("Flow Duration\n1200\n3400\n", encoding="utf-8")
    assert not reloaded.is_seen(f)


def test_scan_spool_returns_only_new_supported_files(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (spool / "b.parquet").write_bytes(b"not really parquet")
    (spool / "ignore.txt").write_text("nope", encoding="utf-8")
    (spool / "sub").mkdir()

    state = WatchState()
    found = {p.name for p in scan_spool(spool, state)}
    assert found == {"a.csv", "b.parquet"}  # .txt and the subdir are ignored

    state.mark(spool / "a.csv")
    found_after = {p.name for p in scan_spool(spool, state)}
    assert found_after == {"b.parquet"}  # a.csv already seen


def test_corrupt_state_file_is_ignored(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not json", encoding="utf-8")
    state = WatchState.load(state_path)  # must not raise
    assert state.processed == {}
