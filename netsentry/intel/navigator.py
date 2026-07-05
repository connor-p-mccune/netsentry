"""Export NetSentry's detection coverage as a MITRE ATT&CK Navigator layer.

The MITRE ATT&CK Navigator (https://mitre-attack.github.io/attack-navigator/)
renders a technique matrix that a security team can color, annotate, and share. A
detection tool that *speaks Navigator* drops straight into that workflow: this
writes a valid layer JSON that maps NetSentry's detectable classes onto ATT&CK
techniques and colors each by the model's measured per-class detection rate, so
"what can we see, and how well" is a picture the SOC already knows how to read.

Scores are per-class **recall** at the operating threshold, computed on the
**stratified** reference split — the split on which every class appears in the test
set, so per-class detection is well defined for the whole matrix (the temporal
headline split, by construction, only contains later-day classes). The split choice
is written into the layer description and metadata so the artifact is self-describing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.evaluation.slices import ClassSlice
from netsentry.intel.attack_mapping import AttackTechnique, tactic_shortname, technique_for
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

LAYER_NAME = "attack_navigator_layer.json"

# Navigator layer / tool versions this file targets. The Navigator is tolerant of
# minor version differences; these are recent, valid values.
_ATTACK_VERSION = "14"
_NAVIGATOR_VERSION = "4.9.1"
_LAYER_VERSION = "4.5"

# Red -> amber -> green: low detection is the coverage gap a defender wants to see.
_GRADIENT = ["#e02b35", "#f7e463", "#3fa34d"]


@dataclass
class TechniqueCoverage:
    """Aggregated detection for one ATT&CK technique across the classes mapping to it."""

    technique: AttackTechnique
    classes: list[str]
    support: int
    detection: float  # support-weighted mean recall over the technique's classes, 0..1


def aggregate_by_technique(slices: list[ClassSlice]) -> list[TechniqueCoverage]:
    """Roll per-class detection up to ATT&CK techniques (support-weighted).

    Several classes can share a technique (FTP/SSH-Patator -> T1110 Brute Force; the
    DoS tools -> T1499), so detection is combined as a support-weighted mean — the
    rate a SOC would see across all flows of that technique.
    """
    grouped: dict[str, TechniqueCoverage] = {}
    for s in slices:
        technique = technique_for(s.label)
        if technique is None:
            continue
        entry = grouped.get(technique.technique_id)
        if entry is None:
            grouped[technique.technique_id] = TechniqueCoverage(
                technique=technique, classes=[s.label], support=s.support, detection=s.detection
            )
            continue
        total = entry.support + s.support
        weighted = (
            (entry.detection * entry.support + s.detection * s.support) / total if total else 0.0
        )
        entry.classes.append(s.label)
        entry.support = total
        entry.detection = weighted
    return sorted(grouped.values(), key=lambda c: c.technique.technique_id)


def build_navigator_layer(
    slices: list[ClassSlice], *, profile_fpr: float, split: str = "stratified"
) -> dict[str, object]:
    """Assemble a MITRE ATT&CK Navigator layer dict scored by per-class detection."""
    coverages = aggregate_by_technique(slices)
    techniques: list[dict[str, object]] = []
    for cov in coverages:
        classes = ", ".join(sorted(cov.classes))
        techniques.append(
            {
                "techniqueID": cov.technique.technique_id,
                "tactic": tactic_shortname(cov.technique.tactic),
                "score": round(cov.detection * 100, 1),
                "color": "",
                "comment": (
                    f"NetSentry classes: {classes}. Detection (recall) "
                    f"{cov.detection * 100:.1f}% @ {profile_fpr * 100:g}% FPR "
                    f"({split} split, n={cov.support:,})."
                ),
                "enabled": True,
                "metadata": [
                    {"name": "classes", "value": classes},
                    {"name": "support", "value": f"{cov.support}"},
                ],
                "showSubtechniques": "." in cov.technique.technique_id,
            }
        )

    return {
        "name": "NetSentry — Detection Coverage",
        "versions": {
            "attack": _ATTACK_VERSION,
            "navigator": _NAVIGATOR_VERSION,
            "layer": _LAYER_VERSION,
        },
        "domain": "enterprise-attack",
        "description": (
            "NetSentry ML-NIDS detection coverage mapped to MITRE ATT&CK. Each "
            "technique is scored by the model's per-class recall at the "
            f"{profile_fpr * 100:g}% false-positive operating point on the {split} "
            "reference split. Indicative mappings for the CIC-IDS2017 scenarios."
        ),
        "techniques": techniques,
        "gradient": {"colors": _GRADIENT, "minValue": 0, "maxValue": 100},
        "legendItems": [
            {"label": "low detection (coverage gap)", "color": _GRADIENT[0]},
            {"label": "high detection", "color": _GRADIENT[-1]},
        ],
        "metadata": [
            {"name": "generated_by", "value": "netsentry navigator"},
            {"name": "operating_point", "value": f"fpr_{profile_fpr * 100:g}pct"},
            {"name": "scoring_split", "value": split},
        ],
        "showTacticRowBackground": True,
        "tacticRowBackground": "#205b8f",
        "selectTechniquesAcrossTactics": True,
        "sorting": 3,  # descending by score, so the biggest gaps surface
    }


def _per_class_detection(settings: Settings) -> tuple[list[ClassSlice], float]:
    """Fit the stratified binary model and score per-class recall at the operating FPR."""
    from netsentry.data.clean import MULTICLASS_TARGET
    from netsentry.data.split import load_split
    from netsentry.evaluation.metrics import positive_scores, threshold_at_fpr
    from netsentry.evaluation.slices import per_class_detection
    from netsentry.training.train_supervised import fit_supervised

    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    y_val_bin = result.y_val.astype(int)
    fpr = settings.thresholds.fpr_targets[-1]  # looser budget -> more detection
    threshold = threshold_at_fpr(y_val_bin, s_val, fpr)

    test = load_split(variant, "stratified", "test")
    labels = test[MULTICLASS_TARGET].to_numpy()
    return per_class_detection(labels, s_test, threshold, settings.labels.benign_label), fpr


def run_navigator_export(settings: Settings) -> Path:
    """Compute per-class detection and write the ATT&CK Navigator layer JSON."""
    slices, fpr = _per_class_detection(settings)
    layer = build_navigator_layer(slices, profile_fpr=fpr, split="stratified")

    out_path = settings.paths.reports_dir / LAYER_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(layer, indent=2), encoding="utf-8")
    techniques = layer["techniques"]
    n_tech = len(techniques) if isinstance(techniques, list) else 0
    logger.info("Wrote ATT&CK Navigator layer", extra={"path": str(out_path), "techniques": n_tech})
    return out_path
