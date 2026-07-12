"""STIX 2.1 export: a spec-faithful, deterministic bundle from summarised detections."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from netsentry.intel.attack_mapping import technique_for
from netsentry.intel.stix import (
    Detection,
    build_bundle,
    summarise_detections,
)

_CREATED = datetime(2026, 7, 12, 9, 0, 0, tzinfo=UTC)


def _detection(**kwargs: object) -> Detection:
    base: dict[str, object] = {
        "predicted_class": "DoS Hulk",
        "span": "1-10",
        "count": 10,
        "peak_probability": 0.97,
        "technique": technique_for("DoS Hulk"),
    }
    base.update(kwargs)
    return Detection(**base)  # type: ignore[arg-type]


def _by_type(bundle: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for obj in bundle["objects"]:  # type: ignore[union-attr]
        grouped.setdefault(str(obj["type"]), []).append(obj)
    return grouped


def test_bundle_envelope_is_valid_stix_21() -> None:
    bundle = build_bundle([_detection()], identity_name="NetSentry", tlp="amber", created=_CREATED)
    assert bundle["type"] == "bundle"
    assert str(bundle["id"]).startswith("bundle--")
    # 2.1 carries spec_version on the SDOs/SROs, not the bundle.
    assert "spec_version" not in bundle
    types = _by_type(bundle)
    assert {"identity", "attack-pattern", "indicator", "sighting", "relationship"} <= set(types)
    for sdo in types["indicator"]:
        assert sdo["spec_version"] == "2.1"


def test_attack_pattern_carries_mitre_external_reference() -> None:
    bundle = build_bundle([_detection()], identity_name="NetSentry", tlp="amber", created=_CREATED)
    (pattern,) = _by_type(bundle)["attack-pattern"]
    ref = pattern["external_references"][0]  # type: ignore[index]
    assert ref["source_name"] == "mitre-attack"
    assert ref["external_id"] == "T1499"  # DoS Hulk -> Endpoint Denial of Service


def test_attack_patterns_are_deduplicated_across_incidents() -> None:
    # Two DoS Hulk incidents share T1499; PortScan adds T1046 -> two patterns, not three.
    detections = [
        _detection(span="1-5"),
        _detection(span="9-14"),
        _detection(predicted_class="PortScan", span="20-25", technique=technique_for("PortScan")),
    ]
    bundle = build_bundle(detections, identity_name="NetSentry", tlp="amber", created=_CREATED)
    patterns = _by_type(bundle)["attack-pattern"]
    assert {p["external_references"][0]["external_id"] for p in patterns} == {"T1499", "T1046"}  # type: ignore[index]
    # But one indicator + one sighting per incident.
    assert len(_by_type(bundle)["indicator"]) == 3
    assert len(_by_type(bundle)["sighting"]) == 3


def test_pattern_prefers_attacking_hosts_then_falls_back_to_port() -> None:
    with_ips = _detection(sources=["10.0.0.9", "10.0.0.10"])
    (indicator,) = _by_type(
        build_bundle([with_ips], identity_name="n", tlp="amber", created=_CREATED)
    )["indicator"]
    assert indicator["pattern"] == "[ipv4-addr:value = '10.0.0.9' OR ipv4-addr:value = '10.0.0.10']"

    port_only = _detection(dst_ports=[80])
    (indicator2,) = _by_type(
        build_bundle([port_only], identity_name="n", tlp="amber", created=_CREATED)
    )["indicator"]
    assert indicator2["pattern"] == "[network-traffic:dst_port = 80]"


def test_observed_data_and_scos_emitted_only_with_capture_identity() -> None:
    with_ips = _detection(sources=["1.2.3.4"], targets=["9.9.9.9"], dst_ports=[80], protocol=6)
    types = _by_type(build_bundle([with_ips], identity_name="n", tlp="amber", created=_CREATED))
    assert len(types["observed-data"]) == 1
    assert len(types["ipv4-addr"]) == 2
    (traffic,) = types["network-traffic"]
    assert traffic["src_ref"].startswith("ipv4-addr--")  # type: ignore[union-attr]
    assert traffic["dst_ref"].startswith("ipv4-addr--")  # type: ignore[union-attr]
    assert "tcp" in traffic["protocols"]  # type: ignore[operator]

    # The sighting points at the observed-data.
    (sighting,) = types["sighting"]
    assert sighting["observed_data_refs"] == [types["observed-data"][0]["id"]]

    # Port-only detection: no SCOs, sighting has no observed_data_refs.
    port_only = _by_type(
        build_bundle([_detection(dst_ports=[80])], identity_name="n", tlp="amber", created=_CREATED)
    )
    assert "observed-data" not in port_only
    assert "observed_data_refs" not in port_only["sighting"][0]


def test_tlp_marking_is_referenced_by_every_object() -> None:
    bundle = build_bundle([_detection()], identity_name="n", tlp="red", created=_CREATED)
    types = _by_type(bundle)
    (marking,) = types["marking-definition"]
    assert marking["name"] == "TLP:RED"
    marking_id = marking["id"]
    for obj in bundle["objects"]:  # type: ignore[union-attr]
        if obj["type"] in {"marking-definition", "ipv4-addr", "network-traffic"}:
            continue  # SCOs and the marking itself are not marked
        assert marking_id in obj["object_marking_refs"]  # type: ignore[operator]


def test_bundle_is_deterministic() -> None:
    args = dict(identity_name="NetSentry", tlp="amber", created=_CREATED)
    a = build_bundle([_detection()], **args)  # type: ignore[arg-type]
    b = build_bundle([_detection()], **args)  # type: ignore[arg-type]
    assert a == b  # identical detections -> byte-identical bundle (idempotent TAXII push)


def test_relationship_links_indicator_to_attack_pattern() -> None:
    bundle = build_bundle([_detection()], identity_name="n", tlp="amber", created=_CREATED)
    types = _by_type(bundle)
    (rel,) = types["relationship"]
    assert rel["relationship_type"] == "indicates"
    assert rel["source_ref"] == types["indicator"][0]["id"]
    assert rel["target_ref"] == types["attack-pattern"][0]["id"]


def test_summarise_extracts_talkers_ports_and_span() -> None:
    responses = [
        _FakeResponse("DoS Hulk", True, 0.9),
        _FakeResponse("DoS Hulk", True, 0.95),
    ]
    df = pd.DataFrame(
        {
            "Src IP": ["10.0.0.1", "10.0.0.1"],
            "Dst IP": ["10.0.0.2", "10.0.0.2"],
            "Destination Port": [80, 80],
            "Protocol": [6, 6],
        }
    )
    (det,) = summarise_detections(responses, df, [[0, 1]], top_k=5)  # type: ignore[arg-type]
    assert det.predicted_class == "DoS Hulk"
    assert det.span == "1-2"
    assert det.count == 2
    assert det.peak_probability == 0.95
    assert det.sources == ["10.0.0.1"]
    assert det.dst_ports == [80]
    assert det.protocol == 6


class _FakeResponse:
    def __init__(self, predicted_class: str, is_attack: bool, probability: float) -> None:
        self.predicted_class = predicted_class
        self.is_attack = is_attack
        self.attack_probability = probability


@pytest.mark.slow
def test_stix_export_end_to_end(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> None:
    import json

    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.intel.stix import build_stix_bundle
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

    in_path = tmp_path / "flows.csv"
    out_path = tmp_path / "bundle.json"
    clean_synth.head(80).to_csv(in_path, index=False)

    stats = build_stix_bundle(settings, in_path, out_path, profile="fpr_1pct")
    assert stats["scored"] == 80
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["type"] == "bundle"
    # Always at least the marking + identity; detections add the rest.
    assert {o["type"] for o in bundle["objects"]} >= {"marking-definition", "identity"}
