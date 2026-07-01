"""CIC-IDS2017 attack classes -> MITRE ATT&CK techniques.

A class label ("DoS Hulk") tells an analyst little; a tactic + technique
("Impact / T1499 Endpoint Denial of Service") is something they can pivot on,
correlate with other tooling, and report up. These are **indicative** mappings for
the CIC-IDS2017 capture scenarios (the dataset is not labelled with ATT&CK IDs);
they are documented as such and live in one place so serving and the coverage
report agree.
"""

from __future__ import annotations

from dataclasses import dataclass

TACTIC_CREDENTIAL_ACCESS = "Credential Access"
TACTIC_IMPACT = "Impact"
TACTIC_INITIAL_ACCESS = "Initial Access"
TACTIC_EXECUTION = "Execution"
TACTIC_C2 = "Command and Control"
TACTIC_DISCOVERY = "Discovery"


@dataclass(frozen=True)
class AttackTechnique:
    """A MITRE ATT&CK technique a detected class maps onto."""

    tactic: str
    technique_id: str
    technique_name: str

    @property
    def url(self) -> str:
        """Canonical ATT&CK page (handles sub-techniques like ``T1499.002``)."""
        return f"https://attack.mitre.org/techniques/{self.technique_id.replace('.', '/')}/"


# Keyed by the *consolidated* model labels (what the classifier actually emits).
_MAPPING: dict[str, AttackTechnique] = {
    "FTP-Patator": AttackTechnique(TACTIC_CREDENTIAL_ACCESS, "T1110", "Brute Force"),
    "SSH-Patator": AttackTechnique(TACTIC_CREDENTIAL_ACCESS, "T1110", "Brute Force"),
    "DoS slowloris": AttackTechnique(TACTIC_IMPACT, "T1499.002", "Service Exhaustion Flood"),
    "DoS Slowhttptest": AttackTechnique(TACTIC_IMPACT, "T1499.002", "Service Exhaustion Flood"),
    "DoS Hulk": AttackTechnique(TACTIC_IMPACT, "T1499", "Endpoint Denial of Service"),
    "DoS GoldenEye": AttackTechnique(TACTIC_IMPACT, "T1499", "Endpoint Denial of Service"),
    "Heartbleed": AttackTechnique(
        TACTIC_INITIAL_ACCESS, "T1190", "Exploit Public-Facing Application"
    ),
    "Web Attack": AttackTechnique(
        TACTIC_INITIAL_ACCESS, "T1190", "Exploit Public-Facing Application"
    ),
    "Infiltration": AttackTechnique(TACTIC_EXECUTION, "T1204", "User Execution"),
    "Bot": AttackTechnique(TACTIC_C2, "T1071", "Application Layer Protocol"),
    "PortScan": AttackTechnique(TACTIC_DISCOVERY, "T1046", "Network Service Discovery"),
    "DDoS": AttackTechnique(TACTIC_IMPACT, "T1498", "Network Denial of Service"),
}

# Raw web-attack variants consolidate to "Web Attack"; map them too for robustness.
_RAW_ALIASES: dict[str, str] = {
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
}


def technique_for(label: str) -> AttackTechnique | None:
    """ATT&CK technique for a (consolidated or raw) attack label; None if benign/unknown."""
    if label in _MAPPING:
        return _MAPPING[label]
    alias = _RAW_ALIASES.get(label)
    return _MAPPING.get(alias) if alias else None


def mitre_payload(label: str) -> dict[str, str] | None:
    """Compact serialisable mapping for a prediction response (or None)."""
    t = technique_for(label)
    if t is None:
        return None
    return {
        "tactic": t.tactic,
        "technique_id": t.technique_id,
        "technique_name": t.technique_name,
        "url": t.url,
    }


@dataclass(frozen=True)
class CoverageSummary:
    """ATT&CK coverage across all mapped attack classes."""

    n_classes: int
    tactics: list[str]
    techniques: list[str]
    mapping: dict[str, AttackTechnique]


def coverage_summary() -> CoverageSummary:
    """Tactics/techniques covered across all mapped classes (for the report)."""
    return CoverageSummary(
        n_classes=len(_MAPPING),
        tactics=sorted({t.tactic for t in _MAPPING.values()}),
        techniques=sorted({t.technique_id for t in _MAPPING.values()}),
        mapping=dict(_MAPPING),
    )
