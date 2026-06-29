"""vulnpipe integration: re-rank vulnerability findings by live traffic risk.

A CVE on a host whose traffic looks like an active attack is more urgent than the
same CVE on a quiet host. This adapter scores each finding's network-flow context
with the NetSentry model (attack probability + anomaly flag) and fuses that with
the finding's base severity into a single triage priority — connecting the two
projects: vulnpipe finds the holes, NetSentry says which are being leaned on.

``VulnFinding`` is a documented contract; point it at real vulnpipe output by
mapping its fields (severity or CVSS, asset) and attaching the host's flow features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from netsentry.evaluation.metrics import attack_probability
from netsentry.features.feature_sets import model_features
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

_SEVERITY_SCORE: dict[str, float] = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}


class VulnFinding(BaseModel):
    """One vulnerability finding plus the network-flow context of its host."""

    id: str
    title: str = ""
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    cvss: float | None = None  # 0-10; if set, used instead of the severity bucket
    asset: str = ""
    flow: dict[str, float] = Field(default_factory=dict)

    def severity_score(self) -> float:
        """Normalised 0-1 base severity (CVSS/10 when provided, else the bucket)."""
        if self.cvss is not None:
            return float(min(max(self.cvss, 0.0), 10.0) / 10.0)
        return _SEVERITY_SCORE[self.severity]


@dataclass
class TriagedFinding:
    """A finding scored and ranked by fused risk."""

    finding: VulnFinding
    attack_probability: float
    is_anomaly: bool
    risk: float
    priority: int

    def as_dict(self) -> dict[str, object]:
        return {
            "priority": self.priority,
            "id": self.finding.id,
            "title": self.finding.title,
            "asset": self.finding.asset,
            "severity": self.finding.severity,
            "risk": round(self.risk, 4),
            "attack_probability": round(self.attack_probability, 4),
            "is_anomaly": self.is_anomaly,
        }


def _input_columns(bundle: ModelBundle, settings: Settings) -> list[str]:
    cols = bundle.metadata.get("input_columns")
    return list(cols) if isinstance(cols, list) else model_features(settings)


def _score(
    bundle: ModelBundle, flows: list[dict[str, float]], settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    """Attack probability and an anomaly flag per flow (anomaly off if no detector)."""
    cols = _input_columns(bundle, settings)
    frame = pd.DataFrame([{c: f.get(c, np.nan) for c in cols} for f in flows], columns=cols)
    attack = attack_probability(
        bundle.predict_proba(frame), bundle.classes, settings.labels.benign_label
    )
    is_anomaly = np.zeros(len(flows), dtype=bool)
    if bundle.anomaly_detector is not None:
        scores = bundle.anomaly_detector.score(bundle.pipeline.transform(frame))
        is_anomaly = scores >= (bundle.anomaly_threshold or float("inf"))
    return np.asarray(attack), is_anomaly


def triage_findings(
    findings: list[VulnFinding], bundle: ModelBundle, settings: Settings
) -> list[TriagedFinding]:
    """Score findings and return them ranked by fused risk (highest first)."""
    if not findings:
        return []
    w = settings.triage
    total = w.severity_weight + w.model_weight + w.anomaly_weight
    attack, is_anomaly = _score(bundle, [f.flow for f in findings], settings)

    scored: list[TriagedFinding] = []
    for finding, prob, anomalous in zip(findings, attack, is_anomaly, strict=True):
        risk = (
            w.severity_weight * finding.severity_score()
            + w.model_weight * float(prob)
            + w.anomaly_weight * (1.0 if anomalous else 0.0)
        ) / total
        scored.append(TriagedFinding(finding, float(prob), bool(anomalous), risk, 0))

    scored.sort(key=lambda t: t.risk, reverse=True)
    for rank, triaged in enumerate(scored, start=1):
        triaged.priority = rank
    return scored


def load_findings(path: Path) -> list[VulnFinding]:
    """Load findings from a JSON list (or a ``{"findings": [...]}`` object)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("findings", []) if isinstance(data, dict) else data
    return [VulnFinding.model_validate(item) for item in items]


def render_triage_markdown(triaged: list[TriagedFinding]) -> str:
    """Render a prioritised triage table."""
    rows = [
        "| # | id | asset | severity | attack p | anomaly | risk |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in triaged:
        rows.append(
            f"| {t.priority} | {t.finding.id} | {t.finding.asset or '-'} | {t.finding.severity} "
            f"| {t.attack_probability:.2f} | {'yes' if t.is_anomaly else 'no'} | {t.risk:.3f} |"
        )
    table = "\n".join(rows)
    return f"""# NetSentry — vulnpipe Triage

_Findings re-ranked by fused risk = severity + NetSentry attack probability +
anomaly flag. A vulnerability on a host whose traffic looks like an active attack
is prioritised over the same severity on a quiet host._

{table}

## Wiring real vulnpipe output

Map each vulnpipe finding to a `VulnFinding` (id, `severity` or `cvss`, asset) and
attach the host's network-flow features as `flow`. The fusion weights live in
config (`triage.*`). Run `netsentry triage --findings <file.json>`.
"""
