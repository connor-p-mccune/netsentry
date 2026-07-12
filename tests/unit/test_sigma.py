"""Sigma export: valid rules, comparison modifiers, ATT&CK tags, deterministic ids."""

from __future__ import annotations

import yaml

from netsentry.config import Settings
from netsentry.config.settings import RuleClause, RuleDefinition
from netsentry.intel.sigma import (
    export_sigma_rules,
    render_sigma_rule,
    sigma_rule_dict,
)

_DATE = "2026/07/12"


def _rule() -> RuleDefinition:
    return RuleDefinition(
        name="volumetric-flood",
        description="High packet- and byte-rate flood (DoS Hulk / DDoS style)",
        clauses=[
            RuleClause(feature="Flow Packets/s", op="ge", value=800.0),
            RuleClause(feature="Flow Bytes/s", op="ge", value=8000.0),
        ],
    )


def test_rule_has_required_sigma_fields() -> None:
    doc = sigma_rule_dict(_rule(), date=_DATE)
    for key in ("title", "id", "logsource", "detection", "status"):
        assert key in doc
    assert doc["logsource"] == {"category": "flow", "product": "cicflowmeter"}
    assert doc["detection"]["condition"] == "selection"


def test_comparison_clause_uses_gte_modifier() -> None:
    doc = sigma_rule_dict(_rule(), date=_DATE)
    selection = doc["detection"]["selection"]
    assert selection["Flow Packets/s|gte"] == 800
    assert selection["Flow Bytes/s|gte"] == 8000


def test_eq_clause_is_exact_match_with_integer_value() -> None:
    rule = RuleDefinition(
        name="ftp-bruteforce",
        description="FTP brute force",
        clauses=[RuleClause(feature="Destination Port", op="eq", value=21.0)],
    )
    selection = sigma_rule_dict(rule, date=_DATE)["detection"]["selection"]
    # eq -> no modifier suffix, and an integral threshold is emitted as an int.
    assert selection == {"Destination Port": 21}
    assert isinstance(selection["Destination Port"], int)


def test_le_clause_uses_lte_modifier() -> None:
    rule = RuleDefinition(
        name="port-scan-sweep",
        description="Short probe",
        clauses=[RuleClause(feature="Flow Duration", op="le", value=20000.0)],
    )
    selection = sigma_rule_dict(rule, date=_DATE)["detection"]["selection"]
    assert selection == {"Flow Duration|lte": 20000}


def test_attack_tags_reuse_the_shared_mapping() -> None:
    # volumetric-flood encodes DoS Hulk -> Impact / T1499.
    tags = sigma_rule_dict(_rule(), date=_DATE)["tags"]
    assert "attack.impact" in tags
    assert "attack.t1499" in tags


def test_ids_are_deterministic_uuid5() -> None:
    a = sigma_rule_dict(_rule(), date=_DATE)["id"]
    b = sigma_rule_dict(_rule(), date="1999/01/01")["id"]  # id ignores the date
    assert a == b  # stable across runs -> no version-control churn


def test_rendered_rule_round_trips_as_valid_yaml() -> None:
    text = render_sigma_rule(_rule(), date=_DATE)
    parsed = yaml.safe_load(text)
    assert parsed["title"] == _rule().description
    assert parsed["detection"]["condition"] == "selection"


def test_colliding_clauses_split_into_separate_selections() -> None:
    # Two clauses on the same field+modifier must not collide as one duplicate key.
    rule = RuleDefinition(
        name="range-rule",
        description="A bounded range",
        clauses=[
            RuleClause(feature="Flow Duration", op="ge", value=100.0),
            RuleClause(feature="Flow Duration", op="ge", value=200.0),
        ],
    )
    detection = sigma_rule_dict(rule, date=_DATE)["detection"]
    assert detection["condition"] == "all of selection_*"
    assert detection["selection_0"] == {"Flow Duration|gte": 100}
    assert detection["selection_1"] == {"Flow Duration|gte": 200}


def test_export_writes_one_file_per_rule_plus_readme(tmp_path) -> None:
    settings = Settings()
    out = export_sigma_rules(settings, tmp_path / "sigma")
    rule_files = sorted(out.glob("*.yml"))
    assert len(rule_files) == len(settings.rules.definitions)
    assert (out / "README.md").exists()
    # Every emitted file is valid, loadable Sigma YAML with a detection block.
    for path in rule_files:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "detection" in doc and "condition" in doc["detection"]
