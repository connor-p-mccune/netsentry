"""Export the signature ruleset as portable Sigma detection rules.

Sigma (https://sigmahq.io) is the generic, vendor-neutral signature format the
detection-engineering community writes rules in: author once in YAML, compile to
Splunk SPL / Elastic KQL / Sentinel KQL / whatever backend a shop runs. NetSentry
already carries a hand-written signature baseline (``rules.definitions``, the
incumbent the ML model is benchmarked against in ``netsentry rules``); this emits
that baseline as Sigma so it drops straight into an existing SIEM pipeline —
alongside the ECS alert stream (``netsentry watch``) and the ATT&CK Navigator
layer (``netsentry navigator``), the third artifact that lets NetSentry speak a
language a SOC already deploys.

The honest scoping, stated in the generated ``README.md`` and mirrored here:

- **Field names are NetSentry/CICFlowMeter flow-feature names** (``Flow Packets/s``,
  ``SYN Flag Count``, ...), not a normalised SIEM taxonomy. A Sigma field-mapping
  (``config`` in the sigmac/pySigma toolchain) points them at whatever flow-log
  schema a deployment ingests. The rules are correct Sigma; the taxonomy binding is
  the operator's, exactly as it is for any custom log source.
- **Numeric comparisons use the Sigma comparison modifiers** ``|gte`` / ``|lte``
  (pySigma >= 0.10); an ``eq`` clause is an exact field match. NaN never matches,
  the same semantics as the in-repo :class:`RuleEngine`.
- **ATT&CK tags are indicative** of the class each signature encodes (via the shared
  ``attack_mapping``), so the exported rules carry the same tactic/technique the
  ``mitre`` prediction field does and the two cannot drift.

Rule ``id``s are deterministic UUIDv5s over the rule name, so regenerating the pack
produces byte-identical files (no churn in version control) and a downstream SIEM
sees stable rule identities across releases.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from netsentry.intel.attack_mapping import tactic_shortname, technique_for
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import RuleDefinition

logger = get_logger(__name__)

SIGMA_DIR = "sigma"

# Fixed namespace so rule ids are reproducible across machines and releases.
_SIGMA_NAMESPACE = uuid.UUID("2f5b0e14-9c2a-5f83-b6d1-7a0e4c9d1f22")

_AUTHOR = "NetSentry"
_REFERENCE = "https://github.com/connor-p-mccune/netsentry"

# Raw clause op -> Sigma field-modifier suffix. ``eq`` is an exact match (no suffix).
_MODIFIER: dict[str, str | None] = {"ge": "gte", "le": "lte", "eq": None}

# Representative attack class each signature encodes, used only to attach indicative
# ATT&CK tags via the shared mapping (so Sigma tags and the `mitre` field agree).
_RULE_CLASS: dict[str, str] = {
    "volumetric-flood": "DoS Hulk",
    "port-scan-sweep": "PortScan",
    "slow-drip-dos": "DoS slowloris",
    "ftp-bruteforce": "FTP-Patator",
    "ssh-bruteforce": "SSH-Patator",
    "tls-heartbeat-exfil": "Heartbleed",
}


def _format_value(value: float) -> float | int:
    """Emit an integral threshold as an int (``21``) and a fractional one as-is."""
    return int(value) if float(value).is_integer() else value


def _attack_tags(rule_name: str) -> list[str]:
    """Indicative ATT&CK tags (``attack.<tactic>`` + ``attack.tNNNN``) for a rule."""
    label = _RULE_CLASS.get(rule_name)
    technique = technique_for(label) if label else None
    if technique is None:
        return []
    return [
        f"attack.{tactic_shortname(technique.tactic)}",
        f"attack.{technique.technique_id.lower()}",
    ]


def _detection_block(rule: RuleDefinition) -> dict[str, object]:
    """Build the Sigma ``detection`` map for a rule (implicit AND over its clauses).

    Each clause becomes a ``field`` or ``field|modifier`` selection key. Two clauses
    on the same field+modifier would collide as duplicate keys, so those are split
    into separate ``selection_N`` groups combined with ``all of selection*``.
    """
    selection: dict[str, object] = {}
    extra: list[dict[str, object]] = []
    for clause in rule.clauses:
        modifier = _MODIFIER[clause.op]
        key = clause.feature if modifier is None else f"{clause.feature}|{modifier}"
        value = _format_value(clause.value)
        if key in selection:
            extra.append({key: value})  # collision: give it its own selection group
        else:
            selection[key] = value

    if not extra:
        return {"selection": selection, "condition": "selection"}

    detection: dict[str, object] = {"selection_0": selection}
    for i, block in enumerate(extra, start=1):
        detection[f"selection_{i}"] = block
    detection["condition"] = "all of selection_*"
    return detection


def sigma_rule_dict(rule: RuleDefinition, *, date: str) -> dict[str, object]:
    """Assemble one Sigma rule as an ordered dict (ready for ``yaml.safe_dump``)."""
    doc: dict[str, object] = {
        "title": rule.description,
        "id": str(uuid.uuid5(_SIGMA_NAMESPACE, rule.name)),
        "status": "experimental",
        "description": (
            f"NetSentry signature '{rule.name}'. {rule.description}. Fields are "
            "CICFlowMeter/NetSentry flow-feature names; map them to your flow-log "
            "schema with a Sigma field mapping."
        ),
        "references": [_REFERENCE],
        "author": _AUTHOR,
        "date": date,
        "logsource": {"category": "flow", "product": "cicflowmeter"},
        "detection": _detection_block(rule),
        "falsepositives": ["Legitimate traffic matching the same volumetric/timing shape"],
        "level": "medium",
    }
    tags = _attack_tags(rule.name)
    if tags:
        doc["tags"] = tags
    return doc


def render_sigma_rule(rule: RuleDefinition, *, date: str) -> str:
    """Render a single rule to a Sigma YAML string (keys kept in authored order)."""
    return yaml.safe_dump(
        sigma_rule_dict(rule, date=date),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def _render_readme(rules: list[RuleDefinition], filenames: list[str]) -> str:
    """Index + honest scoping note for the generated Sigma pack."""
    rows = ["| rule file | signature | ATT&CK |", "|---|---|---|"]
    for rule, filename in zip(rules, filenames, strict=True):
        label = _RULE_CLASS.get(rule.name)
        technique = technique_for(label) if label else None
        tech_id = technique.technique_id if technique else "-"
        rows.append(f"| [`{filename}`]({filename}) | {rule.description} | {tech_id} |")
    return f"""# NetSentry — Sigma detection rules

Generated by `netsentry sigma` from the signature ruleset in
`rules.definitions` — the same hand-written baseline `netsentry rules` benchmarks
the ML model against. [Sigma](https://sigmahq.io) is the vendor-neutral detection
format; compile these with [pySigma](https://github.com/SigmaHQ/pySigma) /
`sigma convert` to Splunk, Elastic, Sentinel, or any supported backend.

{chr(10).join(rows)}

## Field mapping (read before deploying)

The `detection` fields are **NetSentry/CICFlowMeter flow-feature names**
(`Flow Packets/s`, `SYN Flag Count`, ...), not a normalised SIEM taxonomy. Point
them at whatever flow-log schema your SIEM ingests with a Sigma field mapping —
the same one-time binding any custom log source needs. Numeric thresholds use the
Sigma comparison modifiers `|gte` / `|lte`; an unset (NaN) field never matches, so
a signature that references a missing feature simply does not fire.

ATT&CK tags are **indicative** of the class each signature encodes (shared with
the `mitre` prediction field via one mapping, so they cannot drift). Rule `id`s are
deterministic UUIDv5s over the rule name, so regenerating this pack is byte-stable.
"""


def export_sigma_rules(settings: Settings, out_dir: Path | None = None) -> Path:
    """Write one Sigma ``.yml`` per configured rule plus an index README.

    Returns the output directory. Deterministic: the only per-run variation is the
    ``date`` field, taken from the wall clock at generation time.
    """
    target = out_dir or (settings.paths.reports_dir / SIGMA_DIR)
    target.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).strftime("%Y/%m/%d")

    rules = list(settings.rules.definitions)
    filenames: list[str] = []
    for rule in rules:
        filename = f"{rule.name}.yml"
        (target / filename).write_text(render_sigma_rule(rule, date=date), encoding="utf-8")
        filenames.append(filename)

    (target / "README.md").write_text(_render_readme(rules, filenames), encoding="utf-8")
    logger.info("Wrote Sigma rules", extra={"dir": str(target), "rules": len(rules)})
    return target


def run_sigma_export(settings: Settings) -> Path:
    """Analysis-suite entry point: export the Sigma pack and return its directory."""
    return export_sigma_rules(settings)
