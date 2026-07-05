"""MITRE ATT&CK Navigator layer export: aggregation + a valid, loadable layer."""

from __future__ import annotations

import json

from netsentry.evaluation.slices import ClassSlice
from netsentry.intel.navigator import aggregate_by_technique, build_navigator_layer


def test_aggregate_support_weights_detection_across_shared_technique() -> None:
    # FTP-Patator and SSH-Patator both map to T1110 (Brute Force).
    slices = [
        ClassSlice("FTP-Patator", support=100, detection=0.8),
        ClassSlice("SSH-Patator", support=300, detection=0.4),
    ]
    coverages = {c.technique.technique_id: c for c in aggregate_by_technique(slices)}
    assert set(coverages) == {"T1110"}
    t1110 = coverages["T1110"]
    assert t1110.support == 400
    assert t1110.detection == (0.8 * 100 + 0.4 * 300) / 400  # support-weighted mean = 0.5
    assert sorted(t1110.classes) == ["FTP-Patator", "SSH-Patator"]


def test_aggregate_skips_benign_and_unmapped_labels() -> None:
    slices = [
        ClassSlice("BENIGN", support=1000, detection=0.0),
        ClassSlice("Totally Unknown", support=10, detection=0.5),
        ClassSlice("PortScan", support=50, detection=0.9),
    ]
    coverages = aggregate_by_technique(slices)
    assert [c.technique.technique_id for c in coverages] == ["T1046"]  # only PortScan mapped


def test_build_layer_is_valid_and_json_serializable() -> None:
    slices = [
        ClassSlice("DoS Hulk", support=200, detection=0.7),
        ClassSlice("PortScan", support=120, detection=0.3),
        ClassSlice("DDoS", support=180, detection=0.9),
    ]
    layer = build_navigator_layer(slices, profile_fpr=0.01, split="stratified")

    # Round-trips as JSON (the artifact must load in the Navigator).
    reparsed = json.loads(json.dumps(layer))
    assert reparsed["domain"] == "enterprise-attack"
    assert set(reparsed["versions"]) == {"attack", "navigator", "layer"}
    assert reparsed["gradient"]["minValue"] == 0 and reparsed["gradient"]["maxValue"] == 100

    techniques = {t["techniqueID"]: t for t in reparsed["techniques"]}
    assert techniques["T1499"]["tactic"] == "impact"  # DoS Hulk -> Impact tactic shortname
    assert techniques["T1046"]["tactic"] == "discovery"  # PortScan -> Discovery
    for t in reparsed["techniques"]:
        assert 0.0 <= t["score"] <= 100.0
        assert t["enabled"] is True
        assert t["comment"]  # every entry is annotated with its classes + detection


def test_build_layer_scores_match_detection() -> None:
    layer = build_navigator_layer(
        [ClassSlice("DDoS", support=100, detection=0.9)], profile_fpr=0.01
    )
    ddos = next(t for t in layer["techniques"] if t["techniqueID"] == "T1498")
    assert ddos["score"] == 90.0  # 0.9 recall -> score 90
