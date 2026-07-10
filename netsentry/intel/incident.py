"""Incident reports: scored flows folded into the artifact an analyst actually reads.

`netsentry score` and `netsentry pcap` emit one row per flow; nobody responds to
a row. This turns a scored flow file into an **incident report**: consecutive
flagged flows of the same predicted class are grouped into incidents (bridging
small benign gaps, since real attack traffic interleaves with background), and
each incident is rendered with the context a responder starts from — flow count
and span, peak/mean calibrated probability, the MITRE ATT&CK tactic/technique to
pivot on, the services involved, source/destination talkers when the input
carries capture metadata (the `netsentry pcap --flows-out` columns), the
conformal action mix, and the most-cited SHAP feature as the behavioural tell.

The grouping is a *contiguity heuristic*, stated as such in the report: it
assumes flows arrive in stream order and that nearby same-class alerts belong to
one operation — the same correlation assumption the campaigns study prices with
its ``k_confirm`` column. It creates no new detection; every number is a
re-reading of per-flow verdicts the engine already produced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.log import get_logger
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.serving.schemas import PredictionResponse

logger = get_logger(__name__)

DEFAULT_REPORT_NAME = "incident_report.md"

# Optional context columns: used when the input carries them, ignored otherwise.
_SRC_COLUMN = "Src IP"
_DST_COLUMN = "Dst IP"
_PORT_COLUMN = "Destination Port"


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def group_incidents(classes: list[str], attacks: list[bool], gap_tolerance: int) -> list[list[int]]:
    """Row indices per incident: consecutive same-class alerts, small gaps bridged.

    A run closes when the predicted class changes or more than ``gap_tolerance``
    non-alert rows intervene. Pure and order-dependent by design — the input is
    assumed to be in stream (capture) order.
    """
    incidents: list[list[int]] = []
    current: list[int] = []
    current_class: str | None = None
    gap = 0
    for i, (cls, is_attack) in enumerate(zip(classes, attacks, strict=True)):
        if not is_attack:
            gap += 1
            continue
        if current and (cls != current_class or gap > gap_tolerance):
            incidents.append(current)
            current = []
        if not current:
            current_class = cls
        current.append(i)
        gap = 0
    if current:
        incidents.append(current)
    return incidents


@dataclass
class Incident:
    """One grouped operation, summarised from its member flows' verdicts."""

    predicted_class: str
    rows: list[int]
    peak_probability: float
    mean_probability: float
    n_anomalous: int
    actions: Counter[str] = field(default_factory=Counter)
    services: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    top_features: Counter[str] = field(default_factory=Counter)
    mitre: dict[str, str] | None = None

    @property
    def n_flows(self) -> int:
        return len(self.rows)

    @property
    def span(self) -> str:
        return f"{self.rows[0] + 1}-{self.rows[-1] + 1}" if self.rows else "-"


def _summarise(
    rows: list[int],
    responses: list[PredictionResponse],
    df: pd.DataFrame,
    top_talkers: int,
) -> Incident:
    members = [responses[i] for i in rows]
    probs = [m.attack_probability for m in members]
    actions: Counter[str] = Counter(m.recommended_action or "n/a" for m in members)
    features: Counter[str] = Counter(m.top_features[0].feature for m in members if m.top_features)

    def column_values(column: str) -> list[str]:
        if column not in df.columns:
            return []
        values = Counter(str(df[column].iloc[i]) for i in rows)
        return [v for v, _ in values.most_common(top_talkers)]

    services = []
    if _PORT_COLUMN in df.columns:
        from netsentry.data.services import service_of

        counted = Counter(service_of(df[_PORT_COLUMN].iloc[i]) for i in rows)
        services = [s for s, _ in counted.most_common(top_talkers)]

    return Incident(
        predicted_class=members[0].predicted_class,
        rows=rows,
        peak_probability=max(probs),
        mean_probability=sum(probs) / len(probs),
        n_anomalous=sum(1 for m in members if m.is_anomaly),
        actions=actions,
        services=services,
        sources=column_values(_SRC_COLUMN),
        targets=column_values(_DST_COLUMN),
        top_features=features,
        mitre=members[0].mitre,
    )


def build_incident_report(
    settings: Settings,
    input_path: Path,
    output_path: Path,
    *,
    profile: str | None = None,
) -> dict[str, int]:
    """Score a flow file, group the alerts into incidents, write the report."""
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
    incidents = [_summarise(rows, responses, df, cfg.top_talkers) for rows in groups]

    report = _render(settings, input_path, responses, incidents, profile or engine.default_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    stats = {
        "scored": len(responses),
        "alerts": sum(r.is_attack for r in responses),
        "incidents": len(incidents),
    }
    logger.info("Wrote incident report", extra={**stats, "output": str(output_path)})
    return stats


def _incident_section(number: int, inc: Incident) -> str:
    lines = [f"### Incident {number}: {inc.predicted_class}", ""]
    lines.append(
        f"- **Flows:** {inc.n_flows} (rows {inc.span}); peak probability "
        f"{inc.peak_probability:.3f}, mean {inc.mean_probability:.3f}"
        + (f"; {inc.n_anomalous} also flagged anomalous" if inc.n_anomalous else "")
    )
    if inc.mitre is not None:
        lines.append(
            f"- **ATT&CK:** {inc.mitre['tactic']} / [{inc.mitre['technique_id']} "
            f"{inc.mitre['technique_name']}]({inc.mitre['url']})"
        )
    if inc.services:
        lines.append(f"- **Services:** {', '.join(inc.services)}")
    if inc.sources:
        lines.append(f"- **Sources:** {', '.join(inc.sources)}")
    if inc.targets:
        lines.append(f"- **Targets:** {', '.join(inc.targets)}")
    if inc.actions:
        mix = ", ".join(f"{action}: {n}" for action, n in inc.actions.most_common())
        lines.append(f"- **Recommended actions:** {mix}")
    if inc.top_features:
        tell, _count = inc.top_features.most_common(1)[0]
        lines.append(f"- **Behavioural tell (most-cited SHAP feature):** {tell}")
    return "\n".join(lines)


def _render(
    settings: Settings,
    input_path: Path,
    responses: list[PredictionResponse],
    incidents: list[Incident],
    profile: str,
) -> str:
    n_alerts = sum(r.is_attack for r in responses)
    version = responses[0].model_version if responses else "?"
    summary_rows = [
        "| # | class | flows | rows | peak prob | ATT&CK | services |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, inc in enumerate(incidents, start=1):
        technique = inc.mitre["technique_id"] if inc.mitre else "—"
        services = ", ".join(inc.services) if inc.services else "—"
        summary_rows.append(
            f"| {i} | **{inc.predicted_class}** | {inc.n_flows} | {inc.span} "
            f"| {inc.peak_probability:.3f} | {technique} | {services} |"
        )

    if incidents:
        body = "\n\n".join(_incident_section(i, inc) for i, inc in enumerate(incidents, 1))
    else:
        body = (
            "_No incidents: no flow crossed the operating threshold at this profile. "
            "That is a statement about this traffic at this operating point, not a "
            "clean bill of health — see the per-class slices report for what this "
            "model does and does not catch._"
        )

    return f"""# NetSentry — Incident Report

_Input: `{input_path.name}` — {len(responses)} flows scored through model
version {version} at the `{profile}` threshold profile; {n_alerts} flows
alerted, grouped into **{len(incidents)} incident(s)** (same predicted class,
stream-contiguous, benign gaps ≤ {settings.incident.gap_tolerance} bridged)._

## Summary

{chr(10).join(summary_rows) if incidents else "_(empty)_"}

## Incidents

{body}

## How to read this

An *incident* here is a contiguity heuristic over per-flow verdicts — nearby
same-class alerts are assumed to be one operation, the same correlation
assumption the campaigns study prices. The grouping adds no detection: every
verdict, probability, and recommended action comes from the same engine and
operating threshold the API serves, and a silent attack stays silent no matter
how its neighbours are grouped. Probabilities are calibrated scores; "services"
come from `Destination Port` as routing metadata (never a model feature); the
ATT&CK mapping is indicative of the CIC-IDS2017 scenarios.
"""
