"""The detector as a target: NetSentry's own attack surface, governed against MITRE ATLAS.

The [ATT&CK mapping](mitre.md) answers "which adversary behaviours can this system see?".
This answers the question a security reviewer asks next and a portfolio of scattered
robustness studies cannot: **which attacks against the detector itself have we accounted
for, and which have we not?** Twelve separate studies in this repo attack or defend the
model — evasion, poisoning, backdoors, membership inference, model extraction, watermarking,
differential privacy, sanitization, adversarial training, certification, supply-chain
attestation, unlearning — and scattered across twelve reports they are a collection of
interesting exercises rather than a threat model.

**MITRE ATLAS** (Adversarial Threat Landscape for Artificial-Intelligence Systems) is the
ATT&CK-shaped knowledge base for exactly this: adversary tactics and techniques aimed at ML
systems. Mapping the repo's work onto it turns twelve exercises into one governed picture,
in the vocabulary a security team already reads, and — because the matrix contains
techniques nobody here has touched — it makes the **gaps** as legible as the coverage. A
threat model that only lists what you did is marketing; the residual-risk table is the part
that is worth anything.

Two design decisions keep this from becoming a stale wall-chart:

- **Coverage claims are verified against the repository, not asserted.** Every mapping names
  the module, the report, and the CLI command that back it, and the exporter checks those
  paths exist on disk. A study that is deleted or renamed downgrades its own technique to
  `unverified` in the next run rather than silently continuing to claim coverage — the
  failure mode of every hand-maintained compliance document.
- **The catalogue is a pinned snapshot, and says so.** Technique identifiers are recorded
  with the ATLAS version they were taken from and a link to the live entry, because ATLAS
  revises its matrix and a mapping without provenance is worse than no mapping. Verify
  against the live matrix before quoting these IDs anywhere that matters.

The output is a coverage report plus an ATLAS Navigator layer, so the picture drops into the
same tooling a SOC already uses for ATT&CK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "atlas.md"
LAYER_NAME = "atlas_navigator_layer.json"

# The ATLAS matrix revision these identifiers were taken from. ATLAS is versioned and
# revised; anything quoting these IDs downstream should re-check them against the live
# matrix at https://atlas.mitre.org/matrices/ATLAS.
ATLAS_VERSION = "4.5.2 (2024)"
ATLAS_BASE_URL = "https://atlas.mitre.org/techniques"

# Coverage grades, worst to best. The ordering is used to colour the Navigator layer and to
# sort the residual-risk table, so it is defined once here.
NOT_COVERED = "not_covered"
OUT_OF_SCOPE = "out_of_scope"
MEASURED = "measured"
ATTACKED = "attacked"
MITIGATED = "mitigated"
DEFENDED = "defended"

_GRADE_ORDER = [NOT_COVERED, OUT_OF_SCOPE, MEASURED, ATTACKED, MITIGATED, DEFENDED]
_GRADE_SCORE = {
    NOT_COVERED: 0,
    OUT_OF_SCOPE: 15,
    MEASURED: 50,
    ATTACKED: 70,
    MITIGATED: 85,
    DEFENDED: 100,
}
# The distinction between MITIGATED and DEFENDED is deliberate and load-bearing: a control
# that was built but never attacked is weaker evidence than one that survived a measured
# attack, and collapsing the two is how security documents overstate themselves.
_GRADE_LABEL = {
    NOT_COVERED: "not covered",
    OUT_OF_SCOPE: "out of scope",
    MEASURED: "measured",
    ATTACKED: "attack implemented",
    MITIGATED: "control implemented (attack not simulated)",
    DEFENDED: "attack + defense, re-measured",
}


@dataclass(frozen=True)
class AtlasEntry:
    """One ATLAS technique and what this repository does about it.

    ``module``/``report``/``command`` are the evidence: repo-relative paths and the CLI verb
    that reproduces the claim. They are checked to exist, so a claim cannot outlive the code
    that justified it.
    """

    technique_id: str
    technique: str
    tactic: str
    grade: str
    summary: str
    module: str = ""
    report: str = ""
    command: str = ""

    @property
    def url(self) -> str:
        """Link to the live ATLAS entry, so every ID in the report is checkable."""
        return f"{ATLAS_BASE_URL}/{self.technique_id}"

    @property
    def score(self) -> int:
        return _GRADE_SCORE.get(self.grade, 0)


# The mapping. Ordered by ATLAS tactic, then technique. Entries with no evidence paths are
# gaps, and they are here deliberately: a threat model that lists only what you did is
# marketing.
ATLAS_MAPPING: tuple[AtlasEntry, ...] = (
    AtlasEntry(
        technique_id="AML.T0002",
        technique="Acquire Public ML Artifacts",
        tactic="Resource Development",
        grade=MITIGATED,
        summary=(
            "The published model bundle is an artifact an adversary can acquire. Provenance "
            "attestation signs it and the verify gate refuses a bundle whose bytes changed, so "
            "an acquired artifact is at least detectably not the deployed one."
        ),
        module="netsentry/governance/provenance.py",
        report="docs/reports/provenance.md",
        command="netsentry provenance && netsentry verify",
    ),
    AtlasEntry(
        technique_id="AML.T0005",
        technique="Create Proxy ML Model",
        tactic="Resource Development",
        grade=ATTACKED,
        summary=(
            "A surrogate trained on query answers, measured for fidelity and then used to "
            "mount black-box transfer evasion against the real model."
        ),
        module="netsentry/robustness/extraction.py",
        report="docs/reports/extraction.md",
        command="netsentry extraction",
    ),
    AtlasEntry(
        technique_id="AML.T0010",
        technique="ML Supply Chain Compromise",
        tactic="Initial Access",
        grade=MITIGATED,
        summary=(
            "A CycloneDX SBOM plus a hashed model manifest, enforced by an integrity gate "
            "before serving; the canary then re-checks behaviour at load and at hot reload."
        ),
        module="netsentry/governance/provenance.py",
        report="docs/reports/provenance.md",
        command="netsentry verify",
    ),
    AtlasEntry(
        technique_id="AML.T0040",
        technique="ML Model Inference API Access",
        tactic="ML Model Access",
        grade=MITIGATED,
        summary=(
            "The inference API is the adversary's entry point for every query-based attack "
            "below. API-key auth and a per-client rate limit bound the query budget that "
            "extraction and query-search evasion both depend on."
        ),
        module="netsentry/serving/app.py",
        report="docs/reports/extraction.md",
        command="netsentry serve",
    ),
    AtlasEntry(
        technique_id="AML.T0044",
        technique="Full ML Model Access",
        tactic="ML Model Access",
        grade=MEASURED,
        summary=(
            "The white-box case is treated as the worst case throughout: the evasion study's "
            "mimicry attack and the certification study both assume full knowledge of the "
            "model, so the reported robustness is a floor, not a best case."
        ),
        module="netsentry/robustness/certify.py",
        report="docs/reports/certify.md",
        command="netsentry certify",
    ),
    AtlasEntry(
        technique_id="AML.T0043",
        technique="Craft Adversarial Data",
        tactic="ML Attack Staging",
        grade=DEFENDED,
        summary=(
            "Mimicry and adaptive query-search craft evading flows; adversarial training "
            "hardens against them and the study re-measures rather than assuming the fix "
            "worked. Randomized smoothing adds a provable per-flow radius."
        ),
        module="netsentry/robustness/evasion.py",
        report="docs/reports/robustness.md",
        command="netsentry robustness && netsentry harden && netsentry certify",
    ),
    AtlasEntry(
        technique_id="AML.T0042",
        technique="Verify Attack",
        tactic="ML Attack Staging",
        grade=ATTACKED,
        summary=(
            "The query-search evasion attack verifies each candidate against the live decision "
            "before committing, which is exactly this technique and the reason a query budget "
            "is the defender's most effective lever."
        ),
        module="netsentry/robustness/evasion.py",
        report="docs/reports/robustness.md",
        command="netsentry robustness",
    ),
    AtlasEntry(
        technique_id="AML.T0015",
        technique="Evade ML Model",
        tactic="Defense Evasion",
        grade=DEFENDED,
        summary=(
            "The headline evasion result: detection under budgeted feature perturbation, "
            "before and after adversarial training, with the residual gap stated."
        ),
        module="netsentry/robustness/hardening.py",
        report="docs/reports/hardening.md",
        command="netsentry harden",
    ),
    AtlasEntry(
        technique_id="AML.T0020",
        technique="Poison Training Data",
        tactic="Persistence",
        grade=DEFENDED,
        summary=(
            "Label-flip and benign-pool contamination curves quantify the damage; "
            "audit-and-drop sanitization is applied and the recovery re-measured."
        ),
        module="netsentry/robustness/poisoning.py",
        report="docs/reports/poisoning_defense.md",
        command="netsentry poisoning && netsentry sanitize",
    ),
    AtlasEntry(
        technique_id="AML.T0018",
        technique="Backdoor ML Model",
        tactic="Persistence",
        grade=DEFENDED,
        summary=(
            "A BadNets trigger walks attacks through a model whose clean metrics stay green; "
            "spectral signatures detect the poisoned rows without knowing the trigger. The "
            "same mechanism is used constructively to watermark the model for ownership proof."
        ),
        module="netsentry/robustness/backdoor.py",
        report="docs/reports/backdoor.md",
        command="netsentry backdoor && netsentry watermark",
    ),
    AtlasEntry(
        technique_id="AML.T0024",
        technique="Exfiltration via ML Inference API",
        tactic="Exfiltration",
        grade=MITIGATED,
        summary=(
            "The parent technique for the two sub-techniques below; the rate limit and API-key "
            "controls on the prediction endpoints are the shared mitigation, since every "
            "variant is paid for in queries."
        ),
        module="netsentry/serving/app.py",
        report="docs/reports/extraction.md",
        command="netsentry serve",
    ),
    AtlasEntry(
        technique_id="AML.T0024.000",
        technique="Infer Training Data Membership",
        tactic="Exfiltration",
        grade=DEFENDED,
        summary=(
            "Shokri shadow-model and Yeom threshold attacks measure the leak, with an "
            "overfit reference model to price it; DP-SGD with a from-scratch Renyi accountant "
            "buys a formal bound and the utility-leakage frontier prices that."
        ),
        module="netsentry/robustness/membership.py",
        report="docs/reports/dp.md",
        command="netsentry privacy && netsentry dp",
    ),
    AtlasEntry(
        technique_id="AML.T0024.002",
        technique="Extract ML Model",
        tactic="Exfiltration",
        grade=DEFENDED,
        summary=(
            "Query-only model stealing: surrogate fidelity and stolen detection measured "
            "against the query budget, and the budget named as the defense that works."
        ),
        module="netsentry/robustness/extraction.py",
        report="docs/reports/extraction.md",
        command="netsentry extraction",
    ),
    AtlasEntry(
        technique_id="AML.T0031",
        technique="Erode ML Model Integrity",
        tactic="Impact",
        grade=MITIGATED,
        summary=(
            "Slow degradation is treated as a first-class failure: drift monitoring (PSI, "
            "KS+FDR, Page-Hinkley/DDM, a conformal test martingale with an anytime-valid "
            "false-alarm bound), retrain-trigger policy, and threshold refresh."
        ),
        module="netsentry/monitoring/exchangeability.py",
        report="docs/reports/exchangeability.md",
        command="netsentry driftscan && netsentry retrainpolicy",
    ),
    AtlasEntry(
        technique_id="AML.T0046",
        technique="Spamming ML System with Chaff Data",
        tactic="Impact",
        grade=MEASURED,
        summary=(
            "An adversary who floods the queue with near-threshold traffic attacks the "
            "analysts, not the model. The SOC queue simulation and alert-queue capacity study "
            "quantify what that does to the attack SLA; no mitigation is implemented."
        ),
        module="netsentry/evaluation/socsim.py",
        report="docs/reports/socsim.md",
        command="netsentry socsim",
    ),
    AtlasEntry(
        technique_id="AML.T0029",
        technique="Denial of ML Service",
        tactic="Impact",
        grade=MEASURED,
        summary=(
            "Serving cost is measured (benchmark percentiles, the SHAP share of latency, the "
            "cascade's load reduction) and the rate limiter bounds per-client volume, but no "
            "resource-exhaustion attack is implemented or defended against end to end."
        ),
        module="netsentry/serving/cascade.py",
        report="docs/reports/cascade.md",
        command="netsentry benchmark && netsentry cascade",
    ),
    AtlasEntry(
        technique_id="AML.T0034",
        technique="Cost Harvesting",
        tactic="Impact",
        grade=NOT_COVERED,
        summary=(
            "Driving inference spend up by querying an expensive endpoint. The rate limit "
            "bounds it incidentally, but no cost model, quota, or per-tenant accounting exists "
            "and nothing here measures the attack."
        ),
    ),
    AtlasEntry(
        technique_id="AML.T0025",
        technique="Exfiltration via Cyber Means",
        tactic="Exfiltration",
        grade=NOT_COVERED,
        summary=(
            "Stealing the model file off disk or out of the registry rather than through the "
            "API. This is host and IAM security, outside what a detection pipeline controls; "
            "the signed manifest makes tampering detectable but does not prevent theft."
        ),
    ),
    AtlasEntry(
        technique_id="AML.T0013",
        technique="Discover ML Model Ontology",
        tactic="Discovery",
        grade=NOT_COVERED,
        summary=(
            "The API returns class names, SHAP feature attributions, MITRE context, and "
            "conformal prediction sets — a rich description of the model's ontology, offered "
            "deliberately because explanations are a product requirement. The trade-off is "
            "named in the extraction study but not defended against."
        ),
    ),
    AtlasEntry(
        technique_id="AML.T0014",
        technique="Discover ML Artifacts",
        tactic="Discovery",
        grade=NOT_COVERED,
        summary=(
            "Locating model files, MLflow runs, and training data on a compromised host. "
            "Infrastructure hardening, not modelling; noted here so the gap is on the record."
        ),
    ),
    AtlasEntry(
        technique_id="AML.T0051",
        technique="LLM Prompt Injection",
        tactic="Initial Access",
        grade=OUT_OF_SCOPE,
        summary=(
            "NetSentry has no language model and no prompt surface anywhere in the pipeline. "
            "Recorded as out of scope rather than omitted, so the matrix reads honestly."
        ),
    ),
    AtlasEntry(
        technique_id="AML.T0011",
        technique="User Execution",
        tactic="Execution",
        grade=OUT_OF_SCOPE,
        summary=(
            "Requires a human operator to be induced into running adversary-supplied content. "
            "There is no interactive user surface in the serving path; the only human in the "
            "loop is an analyst reading alerts."
        ),
    ),
)


@dataclass
class VerifiedEntry:
    """An ATLAS entry after checking its evidence actually exists in the repository."""

    entry: AtlasEntry
    module_exists: bool
    report_exists: bool

    @property
    def verified(self) -> bool:
        """A coverage claim counts only if the code and report backing it are present."""
        if not self.entry.module and not self.entry.report:
            return self.entry.grade in {NOT_COVERED, OUT_OF_SCOPE}
        return self.module_exists and self.report_exists

    @property
    def effective_grade(self) -> str:
        """The grade after verification — an unbacked claim is not coverage."""
        return self.entry.grade if self.verified else NOT_COVERED


def verify_mapping(
    repo_root: Path, mapping: tuple[AtlasEntry, ...] = ATLAS_MAPPING
) -> list[VerifiedEntry]:
    """Check every coverage claim against the files on disk.

    This is what stops the mapping from becoming the usual stale compliance wall-chart: a
    renamed or deleted study downgrades its own technique on the next run instead of quietly
    continuing to claim coverage.
    """
    out = []
    for entry in mapping:
        out.append(
            VerifiedEntry(
                entry=entry,
                module_exists=not entry.module or (repo_root / entry.module).exists(),
                report_exists=not entry.report or (repo_root / entry.report).exists(),
            )
        )
    return out


def coverage_counts(verified: list[VerifiedEntry]) -> dict[str, int]:
    """How many techniques sit at each grade, after verification."""
    counts = dict.fromkeys(_GRADE_ORDER, 0)
    for v in verified:
        counts[v.effective_grade] = counts.get(v.effective_grade, 0) + 1
    return counts


def coverage_ratio(verified: list[VerifiedEntry]) -> float:
    """Share of *in-scope* techniques with an implemented attack or defense.

    Out-of-scope techniques are excluded from the denominator rather than counted as wins —
    a system with no language model does not get credit for being immune to prompt injection.
    """
    in_scope = [v for v in verified if v.effective_grade != OUT_OF_SCOPE]
    if not in_scope:
        return 0.0
    covered = [v for v in in_scope if v.effective_grade in {ATTACKED, MITIGATED, DEFENDED}]
    return len(covered) / len(in_scope)


def build_layer(verified: list[VerifiedEntry]) -> dict[str, object]:
    """An ATLAS Navigator layer, the same JSON shape the ATT&CK Navigator consumes."""
    techniques = []
    for v in verified:
        e = v.entry
        techniques.append(
            {
                "techniqueID": e.technique_id,
                "tactic": e.tactic.lower().replace(" ", "-"),
                "score": _GRADE_SCORE.get(v.effective_grade, 0),
                "color": "",
                "comment": f"[{_GRADE_LABEL[v.effective_grade]}] {e.summary}",
                "enabled": True,
                "metadata": [
                    {"name": "grade", "value": _GRADE_LABEL[v.effective_grade]},
                    {"name": "module", "value": e.module or "n/a"},
                    {"name": "report", "value": e.report or "n/a"},
                    {"name": "command", "value": e.command or "n/a"},
                    {"name": "evidence verified", "value": "yes" if v.verified else "no"},
                ],
                "showSubtechniques": False,
            }
        )
    return {
        "name": "NetSentry — ML Attack Surface (ATLAS)",
        "versions": {"atlas": ATLAS_VERSION, "navigator": "4.9.1", "layer": "4.5"},
        "domain": "atlas",
        "description": (
            "NetSentry's own ML attack surface mapped to MITRE ATLAS. Scores grade what the "
            "repository implements: 100 = attack and defense with a re-measurement, 75 = "
            "attack implemented, 50 = measured only, 15 = out of scope, 0 = not covered. "
            "Every claim names the module, report, and command backing it, and is verified "
            "against the repository at export time."
        ),
        "techniques": techniques,
        "gradient": {"colors": ["#e02b35", "#f7e463", "#3fa34d"], "minValue": 0, "maxValue": 100},
        "legendItems": [
            {"label": _GRADE_LABEL[g], "color": c}
            for g, c in (
                (DEFENDED, "#3fa34d"),
                (MITIGATED, "#8fbf4d"),
                (ATTACKED, "#b8cf5a"),
                (MEASURED, "#f7e463"),
                (OUT_OF_SCOPE, "#c9c9c9"),
                (NOT_COVERED, "#e02b35"),
            )
        ],
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_atlas_report(settings: Settings) -> Path:
    """Verify the ATLAS mapping against the repo and write the report + Navigator layer."""
    repo_root = Path(__file__).resolve().parents[2]
    verified = verify_mapping(repo_root)
    counts = coverage_counts(verified)
    ratio = coverage_ratio(verified)

    layer_path = settings.paths.reports_dir / LAYER_NAME
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    layer_path.write_text(json.dumps(build_layer(verified), indent=2), encoding="utf-8")

    report = _render(verified, counts, ratio)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote ATLAS coverage report",
        extra={"path": str(out_path), "covered": round(ratio, 3)},
    )
    return out_path


def _coverage_table(verified: list[VerifiedEntry]) -> str:
    rows = [
        "| ATLAS technique | tactic | status | what NetSentry does | reproduce |",
        "|---|---|---|---|---|",
    ]
    order = {g: i for i, g in enumerate(reversed(_GRADE_ORDER))}
    for v in sorted(verified, key=lambda v: (order[v.effective_grade], v.entry.technique_id)):
        e = v.entry
        mark = "" if v.verified else " _(evidence missing)_"
        command = f"`{e.command}`" if e.command else "—"
        rows.append(
            f"| [{e.technique_id}]({e.url}) {e.technique} | {e.tactic} "
            f"| **{_GRADE_LABEL[v.effective_grade]}**{mark} | {e.summary} | {command} |"
        )
    return "\n".join(rows)


def _gap_table(verified: list[VerifiedEntry]) -> str:
    gaps = [v for v in verified if v.effective_grade in {NOT_COVERED, MEASURED}]
    rows = ["| technique | tactic | status | residual risk |", "|---|---|---|---|"]
    for v in sorted(gaps, key=lambda v: (v.effective_grade != NOT_COVERED, v.entry.technique_id)):
        e = v.entry
        rows.append(
            f"| [{e.technique_id}]({e.url}) {e.technique} | {e.tactic} "
            f"| {_GRADE_LABEL[v.effective_grade]} | {e.summary} |"
        )
    return "\n".join(rows)


def _summary_read(counts: dict[str, int], ratio: float, verified: list[VerifiedEntry]) -> str:
    total = len(verified)
    in_scope = total - counts[OUT_OF_SCOPE]
    unverified = [v for v in verified if not v.verified]
    tail = (
        ""
        if not unverified
        else (
            f" **{len(unverified)}** claim(s) failed verification because the module or report "
            "they name is no longer present, and were downgraded automatically — which is the "
            "mechanism working, not a defect."
        )
    )
    return (
        f"Of {total} mapped techniques, {counts[OUT_OF_SCOPE]} are genuinely out of scope "
        f"(no language model, no interactive user surface), leaving {in_scope} in scope. "
        f"{counts[DEFENDED]} carry an implemented attack **and** a defense that was "
        f"re-measured afterwards, {counts[MITIGATED]} carry a control that was built but never "
        f"attacked here, {counts[ATTACKED]} carry an implemented attack with no defense, "
        f"{counts[MEASURED]} are measured but unmitigated, and {counts[NOT_COVERED]} are not "
        f"covered at all — **{ratio:.0%} of in-scope techniques** have working code behind "
        f"them.{tail}"
    )


def _render(verified: list[VerifiedEntry], counts: dict[str, int], ratio: float) -> str:
    return f"""# NetSentry — The Detector as a Target (MITRE ATLAS Coverage)

_Mapping pinned to ATLAS **{ATLAS_VERSION}**. Every coverage claim names the module, report,
and CLI command backing it, and is verified against the repository at export time — a study
that is deleted or renamed downgrades its own technique on the next run. A Navigator layer
is written alongside this report as `{LAYER_NAME}`._

## Why this report exists

The [ATT&CK mapping](mitre.md) answers "which adversary behaviours can this system see?".
This answers the question a security reviewer asks next: **which attacks against the
detector itself have been accounted for, and which have not?** A dozen studies in this
repository attack or defend the model, and scattered across a dozen reports they are a
collection of interesting exercises rather than a threat model. MITRE ATLAS is the
ATT&CK-shaped knowledge base for adversarial ML, so mapping onto it turns them into one
governed picture in a vocabulary a security team already reads — and, because the matrix
contains techniques nobody here has touched, it makes the **gaps** as legible as the
coverage.

## Coverage summary

| status | techniques |
|---|---|
| attack + defense, re-measured | {counts[DEFENDED]} |
| control implemented (attack not simulated) | {counts[MITIGATED]} |
| attack implemented | {counts[ATTACKED]} |
| measured, unmitigated | {counts[MEASURED]} |
| not covered | {counts[NOT_COVERED]} |
| out of scope | {counts[OUT_OF_SCOPE]} |

{_summary_read(counts, ratio, verified)}

## The matrix

{_coverage_table(verified)}

## Residual risk — the part that matters

A threat model that lists only what you did is marketing. These are the in-scope techniques
with no implemented defense, stated plainly so a reviewer does not have to infer them from
absence:

{_gap_table(verified)}

Three of these are honest scope boundaries rather than oversights: model theft off disk and
artifact discovery on a compromised host are infrastructure and IAM problems that a
detection pipeline does not control, and ontology disclosure is a **deliberate** trade — the
API returns SHAP attributions, class names, ATT&CK context and conformal prediction sets
because explanations are a product requirement here, and the
[extraction study](extraction.md) prices what that costs rather than pretending it is free.
The genuine gaps are cost harvesting (no quota or per-tenant accounting exists) and chaff
flooding, where the [SOC simulation](socsim.md) measures the damage to the analyst queue but
nothing mitigates it.

## How this stays true

The failure mode of every security mapping is that it is written once and then drifts from
the system it describes. Two mechanisms guard against that here. Each claim carries its
evidence — a module path, a report path, and the command that regenerates it — and the
exporter checks those paths exist, downgrading any claim whose code has moved. And the
technique identifiers carry the ATLAS version they were taken from plus a link to the live
entry, because ATLAS revises its matrix; treat the IDs as a pinned snapshot and re-check
them against [the live matrix](https://atlas.mitre.org/matrices/ATLAS) before quoting them
anywhere that matters.

## Scope

The mapping is curated by hand: it reflects a considered reading of which ATLAS techniques
this system's work corresponds to, not an automated derivation, and reasonable people could
grade some entries differently — particularly the line between "measured" and "defended",
which this report draws at *was the defense re-measured after being applied*. Grades describe
the repository's engineering, not an operational assurance: an implemented attack proves the
capability was exercised on this synthetic stand-in, not that a production deployment is
resistant to it. Sub-techniques are mapped only where a study addresses one specifically."""
