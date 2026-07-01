"""Threat-intelligence enrichment for NetSentry detections.

Maps CIC-IDS2017 attack classes onto the MITRE ATT&CK framework so a detection
carries a tactic/technique an analyst can pivot on — not just a class name.
"""

from __future__ import annotations

from netsentry.intel.attack_mapping import (
    AttackTechnique,
    coverage_summary,
    mitre_payload,
    technique_for,
)

__all__ = [
    "AttackTechnique",
    "coverage_summary",
    "mitre_payload",
    "technique_for",
]
