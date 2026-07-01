"""MITRE ATT&CK enrichment: class -> technique mapping and coverage."""

from __future__ import annotations

from netsentry.data import schema
from netsentry.intel.attack_mapping import (
    coverage_summary,
    mitre_payload,
    technique_for,
)


def test_benign_and_unknown_map_to_none() -> None:
    assert technique_for("BENIGN") is None
    assert technique_for("not-a-class") is None
    assert mitre_payload("BENIGN") is None


def test_known_class_mappings() -> None:
    assert technique_for("PortScan").technique_id == "T1046"  # discovery
    assert technique_for("DDoS").technique_id == "T1498"  # network DoS
    assert technique_for("FTP-Patator").technique_id == "T1110"  # brute force
    assert technique_for("Bot").tactic == "Command and Control"


def test_web_attack_raw_variants_alias_to_consolidated() -> None:
    consolidated = technique_for("Web Attack")
    assert consolidated is not None
    assert technique_for("Web Attack - XSS") == consolidated
    assert technique_for("Web Attack - Sql Injection") == consolidated


def test_technique_url_handles_subtechniques() -> None:
    t = technique_for("DoS slowloris")  # T1499.002
    assert t.technique_id == "T1499.002"
    assert t.url == "https://attack.mitre.org/techniques/T1499/002/"


def test_payload_shape() -> None:
    payload = mitre_payload("PortScan")
    assert payload is not None
    assert set(payload) == {"tactic", "technique_id", "technique_name", "url"}


def test_every_consolidated_attack_class_is_mapped() -> None:
    # Every attack class the model can emit should carry an ATT&CK technique.
    consolidation = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack",
    }
    consolidated = {consolidation.get(a, a) for a in schema.attack_labels()}
    for label in consolidated:
        assert technique_for(label) is not None, f"{label} has no ATT&CK mapping"


def test_coverage_summary_counts() -> None:
    summary = coverage_summary()
    assert summary.n_classes >= 12
    assert "Impact" in summary.tactics
    assert "T1046" in summary.techniques
