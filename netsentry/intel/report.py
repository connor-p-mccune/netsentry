"""Render the MITRE ATT&CK coverage report for NetSentry's detected classes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.intel.attack_mapping import coverage_summary
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "mitre.md"


def run_mitre_report(settings: Settings) -> Path:
    """Write the ATT&CK coverage report (static — derived from the class mapping)."""
    summary = coverage_summary()

    rows = ["| attack class | tactic | technique |", "|---|---|---|"]
    for label, t in sorted(summary.mapping.items(), key=lambda kv: (kv[1].tactic, kv[0])):
        rows.append(f"| {label} | {t.tactic} | [{t.technique_id} {t.technique_name}]({t.url}) |")

    tactics = ", ".join(summary.tactics)
    report = f"""# NetSentry — MITRE ATT&CK Coverage

Each detected attack class is mapped to a MITRE ATT&CK tactic and technique, so a
prediction carries something an analyst can pivot on — not just a class name. The
serving API returns this mapping in the `mitre` field of every attack prediction.

> These are **indicative** mappings for the CIC-IDS2017 capture scenarios (the
> dataset is not natively labelled with ATT&CK IDs). They encode the behaviour each
> class represents, and are the single source of truth shared by serving and this report.

**Coverage:** {summary.n_classes} attack classes across **{len(summary.tactics)} tactics**
({tactics}) and **{len(summary.techniques)} techniques**.

{chr(10).join(rows)}

## Why this matters

Detection is only the first step; response needs context. Tagging a flagged flow
with its ATT&CK technique lets a SOC correlate NetSentry alerts with EDR/SIEM
detections that speak the same language, prioritise by tactic (a Credential-Access
brute force vs an Impact DoS), and measure detection coverage against the framework
their threat model is written in.

## ATT&CK Navigator layer

`netsentry navigator` exports this coverage as a **MITRE ATT&CK Navigator layer**
(`attack_navigator_layer.json`) — a file you can load directly into the
[ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to see the
technique matrix colored by NetSentry's measured per-class detection (red = coverage
gap, green = well detected). It turns this table into the shareable, framework-native
picture a detection-engineering team actually works from.
"""
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote MITRE ATT&CK report", extra={"path": str(out_path)})
    return out_path
