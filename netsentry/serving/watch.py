"""Spool watcher — turn a directory of dropped flow files into a stream of alerts.

Most networks that would run a NIDS already rotate flow records into a directory
(Zeek writes ``conn.log`` on a timer; a CICFlowMeter cron drops CSVs; the
``netsentry pcap --flows-out`` path emits feature files). ``netsentry watch``
sits on that spool: it scores each new file through the same ``InferenceEngine``
the API serves and appends the attack verdicts as **ECS** (Elastic Common Schema)
JSON lines — the format Elasticsearch / OpenSearch / most SIEMs ingest directly.
It is the streaming sibling of the one-shot ``netsentry score`` / ``incident``
commands: same engine, same verdicts, shaped for a log pipeline instead of a
report.

Design choices that keep it honest and safe to leave running:

- **Exactly-once per file.** A JSON state file records each processed file's size
  and mtime; a file already seen at that (size, mtime) is skipped, so a restart or
  an overlapping tick never double-emits. Growing files (a writer still appending)
  are re-read on the next tick when their mtime advances.
- **Alerts only, by default.** A benign flow is not an event; only flows the
  model flags at the active threshold profile are emitted (``emit_all`` overrides
  for debugging). This is the alert-fatigue discipline the rest of the project
  preaches, applied to the output stream.
- **Best-effort, never fatal.** A malformed or half-written file is logged and
  skipped, not crashed on — a watcher that dies on one bad file is useless.
- **``--once``** processes the current backlog and exits (tests, cron, a manual
  drain); without it the watcher polls every ``interval`` seconds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.intel.attack_mapping import mitre_payload
from netsentry.log import get_logger
from netsentry.serving.batch import score_dataframe
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

ECS_VERSION = "8.11"
_INPUT_SUFFIXES = (".csv", ".parquet")

# Optional capture-identity columns (present in pcap/Zeek-derived files, absent in
# bare feature CSVs); mapped into ECS network fields, never into the model.
_SRC_IP, _SRC_PORT = "Src IP", "Src Port"
_DST_IP, _DST_PORT = "Dst IP", "Dst Port"
_PROTOCOL = "Protocol"


@dataclass
class WatchState:
    """Which spool files have already been scored (keyed by name -> size/mtime)."""

    processed: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> WatchState:
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(processed={str(k): list(v) for k, v in raw.get("processed", {}).items()})
        except Exception as exc:  # a corrupt state file must not wedge the watcher
            logger.warning("Ignoring unreadable watch state (%s)", exc)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"processed": self.processed}, indent=2), encoding="utf-8")

    def is_seen(self, path: Path) -> bool:
        """True iff this exact file (by size and mtime) was already processed."""
        stat = path.stat()
        prev = self.processed.get(path.name)
        return prev is not None and prev == [stat.st_size, stat.st_mtime]

    def mark(self, path: Path) -> None:
        stat = path.stat()
        self.processed[path.name] = [stat.st_size, stat.st_mtime]


def _severity(probability: float) -> int:
    """ECS event.severity as a 0-100 integer from the calibrated attack probability."""
    return round(max(0.0, min(1.0, probability)) * 100)


def _present(value: object) -> bool:
    """A column value that is neither absent nor NaN, so it belongs in the document."""
    return value is not None and not pd.isna(value)


def _network_fields(row: dict[str, object]) -> dict[str, object]:
    """Map any present capture-identity columns into ECS source/destination/network."""
    doc: dict[str, object] = {}
    src: dict[str, object] = {}
    dst: dict[str, object] = {}
    network: dict[str, object] = {}
    if _present(row.get(_SRC_IP)):
        src["ip"] = str(row[_SRC_IP])
    if _present(row.get(_SRC_PORT)):
        src["port"] = int(float(row[_SRC_PORT]))  # type: ignore[arg-type]
    if _present(row.get(_DST_IP)):
        dst["ip"] = str(row[_DST_IP])
    if _present(row.get(_DST_PORT)):
        dst["port"] = int(float(row[_DST_PORT]))  # type: ignore[arg-type]
    if _present(row.get(_PROTOCOL)):
        network["iana_number"] = int(float(row[_PROTOCOL]))  # type: ignore[arg-type]
    if src:
        doc["source"] = src
    if dst:
        doc["destination"] = dst
    if network:
        doc["network"] = network
    return doc


def to_ecs_alert(
    prediction: dict[str, object],
    context: dict[str, object],
    *,
    source_file: str,
    timestamp: str,
    model_version: str,
    threshold_profile: str | None = None,
) -> dict[str, object]:
    """Map one scored flow to an ECS alert document (pure; JSON-serialisable).

    Follows the Elastic Common Schema so the output drops straight into a SIEM:
    ``event.*`` for the alert envelope, ``rule.name`` for the predicted class,
    ``threat.*`` for the MITRE mapping, ``source``/``destination``/``network`` for
    any capture identity that rode along, and a ``netsentry`` namespace for the
    model-specific fields that have no native ECS home.
    """
    probability = float(prediction.get("attack_probability", 0.0))  # type: ignore[arg-type]
    predicted = str(prediction.get("predicted_class", "attack"))
    action = prediction.get("recommended_action") or "alert"
    doc: dict[str, object] = {
        "@timestamp": timestamp,
        "ecs": {"version": ECS_VERSION},
        "event": {
            "kind": "alert",
            "category": ["intrusion_detection", "network"],
            "type": ["info"],
            "action": str(action),
            "dataset": "netsentry.flow",
            "module": "netsentry",
            "provider": "netsentry",
            "severity": _severity(probability),
            "risk_score": round(probability * 100, 2),
        },
        "rule": {"name": predicted, "ruleset": "netsentry-nids"},
        "netsentry": {
            "attack_probability": round(probability, 6),
            "is_attack": bool(prediction.get("is_attack", False)),
            "anomaly_score": prediction.get("anomaly_score"),
            "is_anomaly": prediction.get("is_anomaly"),
            "recommended_action": prediction.get("recommended_action"),
            "threshold_profile": threshold_profile,
            "top_feature": prediction.get("top_feature"),
            "model_version": model_version,
        },
        "message": f"NetSentry flagged {predicted} (p={probability:.3f})",
        "log": {"file": {"path": source_file}},
    }
    mitre = mitre_payload(predicted)
    if mitre:
        doc["threat"] = {
            "framework": "MITRE ATT&CK",
            "technique": {"id": mitre.get("technique_id"), "name": mitre.get("technique_name")},
            "tactic": {"name": mitre.get("tactic")},
        }
    doc.update(_network_fields(context))
    return doc


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def process_file(
    engine: InferenceEngine, path: Path, *, profile: str | None, emit_all: bool
) -> list[dict[str, object]]:
    """Score one spool file and return its ECS alert documents (attacks only by default)."""
    frame = _read(path)
    used_profile = profile or engine.default_profile
    predictions = score_dataframe(
        engine, frame, profile=profile, batch_size=engine.settings.serving.max_batch_size
    )
    now = datetime.now(UTC).isoformat()
    context_cols = [_SRC_IP, _SRC_PORT, _DST_IP, _DST_PORT, _PROTOCOL]
    present = [c for c in context_cols if c in frame.columns]
    alerts: list[dict[str, object]] = []
    for i, pred in enumerate(predictions.to_dict("records")):
        if not emit_all and not pred.get("is_attack"):
            continue
        context = {c: frame.iloc[i][c] for c in present} if present else {}
        alerts.append(
            to_ecs_alert(
                pred,
                context,
                source_file=str(path),
                timestamp=now,
                model_version=engine.version,
                threshold_profile=used_profile,
            )
        )
    return alerts


def _append_alerts(alerts_path: Path, alerts: list[dict[str, object]]) -> None:
    """Append ECS documents as JSON lines (the SIEM-native, append-safe format)."""
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    with alerts_path.open("a", encoding="utf-8") as fh:
        for alert in alerts:
            fh.write(json.dumps(alert) + "\n")


def scan_spool(spool: Path, state: WatchState) -> list[Path]:
    """New or changed flow files in the spool, oldest first (stable processing order)."""
    candidates = [
        p
        for p in sorted(spool.iterdir(), key=lambda q: q.stat().st_mtime)
        if p.is_file() and p.suffix in _INPUT_SUFFIXES and not state.is_seen(p)
    ]
    return candidates


def run_watch(
    settings: Settings,
    *,
    spool: Path,
    alerts_out: Path,
    state_path: Path,
    profile: str | None = None,
    once: bool = False,
    interval: float = 5.0,
    emit_all: bool = False,
) -> dict[str, int]:
    """Watch ``spool`` for flow files, emit ECS alerts, and track processed files.

    Returns cumulative counts (files processed, alerts emitted). With ``once`` the
    current backlog is drained and the call returns; otherwise it polls every
    ``interval`` seconds until interrupted, persisting state after each file so a
    crash resumes without re-scoring.
    """
    if not spool.exists():
        raise FileNotFoundError(f"spool directory not found: {spool}")
    engine = InferenceEngine(settings)
    state = WatchState.load(state_path)
    totals = {"files": 0, "alerts": 0}

    def _drain() -> None:
        for path in scan_spool(spool, state):
            try:
                alerts = process_file(engine, path, profile=profile, emit_all=emit_all)
            except Exception as exc:  # a bad file is skipped, not fatal
                logger.warning("Skipping unscorable spool file %s (%s)", path.name, exc)
                state.mark(path)  # do not retry a structurally broken file every tick
                state.save(state_path)
                continue
            _append_alerts(alerts_out, alerts)
            state.mark(path)
            state.save(state_path)
            totals["files"] += 1
            totals["alerts"] += len(alerts)
            logger.info("Scored spool file", extra={"file": path.name, "alerts": len(alerts)})

    _drain()
    if once:
        return totals
    logger.info("Watching spool", extra={"spool": str(spool), "interval": interval})
    while True:  # pragma: no cover - the polling loop is exercised via `once`
        time.sleep(interval)
        _drain()
