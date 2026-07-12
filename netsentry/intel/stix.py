"""Export NetSentry detections as a STIX 2.1 threat-intelligence bundle.

[STIX 2.1](https://oasis-open.github.io/cti-documentation/) is the OASIS standard
for exchanging cyber-threat intelligence — the format a TAXII server serves and a
threat-intel platform (MISP, OpenCTI, Anomali) ingests. Emitting STIX turns a
NetSentry run from a private verdict stream into shareable intelligence: the same
scored flows the API serves, folded into incidents (reusing the incident grouping),
become a bundle of standard STIX objects a hunting or intel team can import
directly.

The bundle is faithful STIX 2.1, not a JSON blob that merely borrows the vocabulary:

- an **identity** SDO for the producing system (the ``created_by_ref`` on every object);
- one **attack-pattern** SDO per observed ATT&CK technique, with ``external_references``
  into ``mitre-attack`` (shared with the ``mitre`` prediction field, so intel and API
  cannot drift);
- one **indicator** per incident, carrying a real **STIX pattern** over the attacking
  hosts (``ipv4-addr:value``) or the targeted service (``network-traffic:dst_port``);
- **observed-data** + the **SCOs** it references (``ipv4-addr``, ``network-traffic``)
  when the input carries capture identity, so the sighting points at concrete observables;
- a **sighting** SRO per incident (count, first/last seen) and a **relationship**
  (``indicator`` *indicates* ``attack-pattern``) so the graph is navigable;
- a **TLP marking-definition** referenced by every object (default TLP:AMBER).

Every object id is a deterministic UUIDv5 over stable content, so re-exporting the
same detections yields a byte-identical bundle (diffable, testable, idempotent to a
TAXII push). Timestamps are the one intentional per-run field.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.intel.attack_mapping import AttackTechnique, technique_for
from netsentry.intel.incident import group_incidents
from netsentry.log import get_logger
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.serving.schemas import PredictionResponse

logger = get_logger(__name__)

DEFAULT_BUNDLE_NAME = "stix_bundle.json"
_SPEC_VERSION = "2.1"

# Fixed namespace so object ids are reproducible across machines and releases.
_STIX_NAMESPACE = uuid.UUID("8b1f7c4a-2e59-5d38-9a6b-3c0d1e2f4a5b")

# STIX 2.1 statically-defined TLP marking-definition ids (from the spec appendix).
_TLP_MARKINGS: dict[str, tuple[str, str]] = {
    "white": ("TLP:WHITE", "marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9"),
    "green": ("TLP:GREEN", "marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da"),
    "amber": ("TLP:AMBER", "marking-definition--f88d31f6-486f-44da-b317-01333bde0b82"),
    "red": ("TLP:RED", "marking-definition--5e57c739-391a-4eb3-b6be-7d15ca92d5ed"),
}

# Optional capture-identity columns (present in pcap/Zeek-derived files).
_SRC_IP, _DST_IP = "Src IP", "Dst IP"
_PROTOCOL = "Protocol"
_PORT_COLUMNS = ("Destination Port", "Dst Port")


def _stix_ts(dt: datetime) -> str:
    """STIX timestamp: RFC 3339 UTC with millisecond precision and a ``Z`` suffix."""
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _sid(stix_type: str, key: str) -> str:
    """Deterministic STIX id: ``<type>--<uuid5(namespace, key)>``."""
    return f"{stix_type}--{uuid.uuid5(_STIX_NAMESPACE, f'{stix_type}:{key}')}"


@dataclass
class Detection:
    """One incident summarised into the observables a STIX bundle needs."""

    predicted_class: str
    span: str  # stable per-input key, e.g. "3-17"
    count: int
    peak_probability: float
    technique: AttackTechnique | None
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    dst_ports: list[int] = field(default_factory=list)
    protocol: int | None = None

    @property
    def key(self) -> str:
        return f"{self.predicted_class}:{self.span}"


def _port_column(df: pd.DataFrame) -> str | None:
    for column in _PORT_COLUMNS:
        if column in df.columns:
            return column
    return None


def _top_values(df: pd.DataFrame, column: str, rows: list[int], k: int) -> list[str]:
    if column not in df.columns:
        return []
    counted = Counter(str(df[column].iloc[i]) for i in rows if not pd.isna(df[column].iloc[i]))
    return [v for v, _ in counted.most_common(k)]


def summarise_detections(
    responses: list[PredictionResponse], df: pd.DataFrame, groups: list[list[int]], top_k: int
) -> list[Detection]:
    """Fold grouped alert rows into :class:`Detection` observables (pure)."""
    port_col = _port_column(df)
    detections: list[Detection] = []
    for rows in groups:
        members = [responses[i] for i in rows]
        ports: list[int] = []
        if port_col is not None:
            for value in _top_values(df, port_col, rows, top_k):
                try:
                    ports.append(int(float(value)))
                except ValueError:
                    continue
        protocol: int | None = None
        if _PROTOCOL in df.columns:
            proto_vals = _top_values(df, _PROTOCOL, rows, 1)
            if proto_vals:
                try:
                    protocol = int(float(proto_vals[0]))
                except ValueError:
                    protocol = None
        detections.append(
            Detection(
                predicted_class=members[0].predicted_class,
                span=f"{rows[0] + 1}-{rows[-1] + 1}",
                count=len(rows),
                peak_probability=max(m.attack_probability for m in members),
                technique=technique_for(members[0].predicted_class),
                sources=_top_values(df, _SRC_IP, rows, top_k),
                targets=_top_values(df, _DST_IP, rows, top_k),
                dst_ports=ports,
                protocol=protocol,
            )
        )
    return detections


def _identity_object(name: str, created: str, marking_ref: str) -> dict[str, object]:
    return {
        "type": "identity",
        "spec_version": _SPEC_VERSION,
        "id": _sid("identity", name),
        "created": created,
        "modified": created,
        "name": name,
        "identity_class": "system",
        "description": "Machine-learning network intrusion detection system.",
        "object_marking_refs": [marking_ref],
    }


def _attack_pattern_object(
    technique: AttackTechnique, created: str, created_by: str, marking_ref: str
) -> dict[str, object]:
    return {
        "type": "attack-pattern",
        "spec_version": _SPEC_VERSION,
        "id": _sid("attack-pattern", technique.technique_id),
        "created_by_ref": created_by,
        "created": created,
        "modified": created,
        "name": technique.technique_name,
        "external_references": [
            {
                "source_name": "mitre-attack",
                "external_id": technique.technique_id,
                "url": technique.url,
            }
        ],
        "object_marking_refs": [marking_ref],
    }


def _pattern_for(detection: Detection) -> str:
    """A STIX pattern over the attacking hosts, else the targeted service port."""
    if detection.sources:
        clauses = " OR ".join(f"ipv4-addr:value = '{ip}'" for ip in detection.sources)
        return f"[{clauses}]"
    if detection.dst_ports:
        clauses = " OR ".join(f"network-traffic:dst_port = {port}" for port in detection.dst_ports)
        return f"[{clauses}]"
    # No capture identity and no port: fall back to a protocol-level observable so the
    # indicator still carries a valid, if coarse, pattern.
    return "[network-traffic:protocols[*] = 'ip']"


def _indicator_object(
    detection: Detection, created: str, created_by: str, marking_ref: str
) -> dict[str, object]:
    return {
        "type": "indicator",
        "spec_version": _SPEC_VERSION,
        "id": _sid("indicator", detection.key),
        "created_by_ref": created_by,
        "created": created,
        "modified": created,
        "name": f"NetSentry: {detection.predicted_class} ({detection.count} flows)",
        "description": (
            f"{detection.count} flow(s) classified as {detection.predicted_class} "
            f"(peak calibrated probability {detection.peak_probability:.3f})."
        ),
        "indicator_types": ["malicious-activity"],
        "pattern": _pattern_for(detection),
        "pattern_type": "stix",
        "pattern_version": _SPEC_VERSION,
        "valid_from": created,
        "confidence": round(detection.peak_probability * 100),
        "object_marking_refs": [marking_ref],
    }


def _observed_data_and_scos(
    detection: Detection, created: str, created_by: str, marking_ref: str
) -> tuple[dict[str, object], list[dict[str, object]]] | None:
    """Build the observed-data SDO plus the SCOs it references, or None if no identity.

    A STIX ``network-traffic`` SCO must carry at least one of ``src_ref`` / ``dst_ref``,
    so this is emitted only when the input carried source and/or destination IPs.
    """
    src_ip = detection.sources[0] if detection.sources else None
    dst_ip = detection.targets[0] if detection.targets else None
    if src_ip is None and dst_ip is None:
        return None

    scos: list[dict[str, object]] = []
    traffic: dict[str, object] = {
        "type": "network-traffic",
        "id": _sid("network-traffic", detection.key),
        "protocols": ["ipv4", "tcp"] if detection.protocol == 6 else ["ipv4"],
    }
    if src_ip is not None:
        src: dict[str, object] = {
            "type": "ipv4-addr",
            "id": _sid("ipv4-addr", src_ip),
            "value": src_ip,
        }
        scos.append(src)
        traffic["src_ref"] = src["id"]
    if dst_ip is not None:
        dst: dict[str, object] = {
            "type": "ipv4-addr",
            "id": _sid("ipv4-addr", dst_ip),
            "value": dst_ip,
        }
        scos.append(dst)
        traffic["dst_ref"] = dst["id"]
    if detection.dst_ports:
        traffic["dst_port"] = detection.dst_ports[0]
    scos.append(traffic)

    observed = {
        "type": "observed-data",
        "spec_version": _SPEC_VERSION,
        "id": _sid("observed-data", detection.key),
        "created_by_ref": created_by,
        "created": created,
        "modified": created,
        "first_observed": created,
        "last_observed": created,
        "number_observed": detection.count,
        "object_refs": [sco["id"] for sco in scos],
        "object_marking_refs": [marking_ref],
    }
    return observed, scos


def _sighting_object(
    detection: Detection,
    indicator_id: str,
    observed_data_id: str | None,
    created: str,
    created_by: str,
    marking_ref: str,
) -> dict[str, object]:
    sighting: dict[str, object] = {
        "type": "sighting",
        "spec_version": _SPEC_VERSION,
        "id": _sid("sighting", detection.key),
        "created_by_ref": created_by,
        "created": created,
        "modified": created,
        "sighting_of_ref": indicator_id,
        "count": detection.count,
        "first_seen": created,
        "last_seen": created,
        "object_marking_refs": [marking_ref],
    }
    if observed_data_id is not None:
        sighting["observed_data_refs"] = [observed_data_id]
    return sighting


def _relationship_object(
    indicator_id: str, attack_pattern_id: str, created: str, created_by: str, marking_ref: str
) -> dict[str, object]:
    return {
        "type": "relationship",
        "spec_version": _SPEC_VERSION,
        "id": _sid("relationship", f"{indicator_id}|{attack_pattern_id}"),
        "created_by_ref": created_by,
        "created": created,
        "modified": created,
        "relationship_type": "indicates",
        "source_ref": indicator_id,
        "target_ref": attack_pattern_id,
        "object_marking_refs": [marking_ref],
    }


def build_bundle(
    detections: list[Detection],
    *,
    identity_name: str,
    tlp: str,
    created: datetime | None = None,
) -> dict[str, object]:
    """Assemble a complete, deterministic STIX 2.1 bundle from summarised detections."""
    ts = _stix_ts(created or datetime.now(UTC))
    tlp_name, marking_ref = _TLP_MARKINGS[tlp]

    marking: dict[str, object] = {
        "type": "marking-definition",
        "spec_version": _SPEC_VERSION,
        "id": marking_ref,
        "created": "2017-01-20T00:00:00.000Z",  # the spec's fixed TLP definition timestamp
        "definition_type": "tlp",
        "name": tlp_name,
        "definition": {"tlp": tlp_name.split(":")[1].lower()},
    }
    identity = _identity_object(identity_name, ts, marking_ref)
    identity_id = str(identity["id"])

    objects: list[dict[str, object]] = [marking, identity]

    # One attack-pattern per distinct observed technique (deduplicated).
    patterns: dict[str, dict[str, object]] = {}
    for det in detections:
        if det.technique is not None and det.technique.technique_id not in patterns:
            patterns[det.technique.technique_id] = _attack_pattern_object(
                det.technique, ts, identity_id, marking_ref
            )
    objects.extend(patterns.values())

    for det in detections:
        indicator = _indicator_object(det, ts, identity_id, marking_ref)
        indicator_id = str(indicator["id"])
        objects.append(indicator)

        observed_id: str | None = None
        built = _observed_data_and_scos(det, ts, identity_id, marking_ref)
        if built is not None:
            observed, scos = built
            objects.extend(scos)
            objects.append(observed)
            observed_id = str(observed["id"])

        objects.append(
            _sighting_object(det, indicator_id, observed_id, ts, identity_id, marking_ref)
        )
        if det.technique is not None:
            pattern_id = str(patterns[det.technique.technique_id]["id"])
            objects.append(
                _relationship_object(indicator_id, pattern_id, ts, identity_id, marking_ref)
            )

    # Content-addressed bundle id: identical detections -> identical bundle.
    digest_key = "|".join(sorted(str(obj["id"]) for obj in objects))
    return {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid5(_STIX_NAMESPACE, digest_key)}",
        "objects": objects,
    }


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def build_stix_bundle(
    settings: Settings,
    input_path: Path,
    output_path: Path,
    *,
    profile: str | None = None,
) -> dict[str, int]:
    """Score a flow file, group alerts into incidents, write a STIX 2.1 bundle."""
    engine = InferenceEngine(settings)
    df = _read(input_path)
    cfg = settings.incident

    flows = df.to_dict("records")
    responses: list[PredictionResponse] = []
    batch = settings.serving.max_batch_size
    for start in range(0, len(flows), batch):
        responses.extend(engine.predict(flows[start : start + batch], profile=profile))

    groups = group_incidents(
        [r.predicted_class for r in responses],
        [r.is_attack for r in responses],
        cfg.gap_tolerance,
    )
    detections = summarise_detections(responses, df, groups, cfg.top_talkers)
    bundle = build_bundle(
        detections,
        identity_name=settings.stix.identity_name,
        tlp=settings.stix.tlp,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    objects = bundle["objects"]
    stats = {
        "scored": len(responses),
        "alerts": sum(r.is_attack for r in responses),
        "detections": len(detections),
        "stix_objects": len(objects) if isinstance(objects, list) else 0,
    }
    logger.info("Wrote STIX bundle", extra={**stats, "output": str(output_path)})
    return stats
