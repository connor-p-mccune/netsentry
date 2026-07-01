"""Counterfactual recourse — the analyst's "what would clear this flow?".

SHAP answers *why* a flow fired; recourse answers *what-if*: the smallest set of
changes to attacker-controllable features that would drop the flow below the
decision threshold. It is the defender's read of the same feature space the
robustness study attacks — useful for triaging a hit ("it fired mostly on volume")
and for understanding false positives ("normalising two features clears it").

Greedy and model-agnostic: repeatedly move the single controllable feature that most
reduces the calibrated attack probability toward the benign centroid, until the flow
flips or a change budget is hit. Deltas are in standardized (model-space) units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.log import get_logger
from netsentry.robustness.evasion import (
    attack_scores_transformed,
    base_feature_name,
    controllable_indices,
)

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

REPORT_NAME = "recourse.md"


@dataclass
class Change:
    """One counterfactual edit: move ``feature`` by ``delta`` std units."""

    feature: str
    delta: float  # signed change in standardized units (target - current)

    @property
    def direction(self) -> str:
        return "decrease" if self.delta < 0 else "increase"


@dataclass
class Recourse:
    """A counterfactual explanation for a single flow."""

    original_score: float
    final_score: float
    threshold: float
    flipped: bool
    changes: list[Change]


def recourse_for_row(
    bundle: ModelBundle,
    x_row: np.ndarray,
    centroid: np.ndarray,
    ctrl_idx: np.ndarray,
    threshold: float,
    feature_names: list[str],
    max_steps: int,
) -> Recourse:
    """Greedy minimal recourse for one transformed flow row (shape ``(1, d)``)."""
    current = np.array(x_row, dtype=float, copy=True)
    original = float(attack_scores_transformed(bundle, current)[0])
    changes: list[Change] = []
    used: set[int] = set()

    for _ in range(max_steps):
        score = float(attack_scores_transformed(bundle, current)[0])
        if score < threshold:
            break
        best_j, best_score = -1, score
        for j in ctrl_idx:
            if int(j) in used:
                continue
            trial = current.copy()
            trial[0, j] = centroid[j]
            sj = float(attack_scores_transformed(bundle, trial)[0])
            if sj < best_score:
                best_score, best_j = sj, int(j)
        if best_j < 0:  # no controllable move reduces the score further
            break
        delta = float(centroid[best_j] - current[0, best_j])
        changes.append(Change(base_feature_name(feature_names[best_j]), delta))
        current[0, best_j] = centroid[best_j]
        used.add(best_j)

    final = float(attack_scores_transformed(bundle, current)[0])
    return Recourse(original, final, threshold, final < threshold, changes)


def explain_recourse(
    settings: Settings,
    bundle: ModelBundle,
    flow: pd.DataFrame,
    benign_ref: pd.DataFrame,
    *,
    profile: str | None = None,
) -> Recourse:
    """Counterfactual recourse for a single raw flow (one-row DataFrame)."""
    cfg = settings.robustness
    feature_names = bundle.feature_names()
    ctrl_idx = controllable_indices(feature_names, cfg.controllable_features)
    centroid = np.asarray(bundle.pipeline.transform(benign_ref)).mean(axis=0)
    x_row = np.asarray(bundle.pipeline.transform(flow))[:1]
    chosen = profile or cfg.profile
    threshold = bundle.thresholds.get(chosen, 0.5)
    return recourse_for_row(
        bundle, x_row, centroid, ctrl_idx, threshold, feature_names, cfg.recourse_max_steps
    )


def run_recourse_report(settings: Settings, n_examples: int = 5) -> Path:
    """Compute recourse for a few flagged test flows and write a worked-examples report."""
    from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
    from netsentry.data.split import load_split
    from netsentry.models.registry import latest_bundle, load_bundle
    from netsentry.serving.bundle import build_serving_bundle

    bundle_path = settings.serving.artifact_path or latest_bundle(settings)
    if bundle_path is None:
        bundle_path = build_serving_bundle(settings)
    bundle = load_bundle(Path(bundle_path))

    cfg = settings.robustness
    feature_names = bundle.feature_names()
    ctrl_idx = controllable_indices(feature_names, cfg.controllable_features)
    threshold = bundle.thresholds.get(cfg.profile, 0.5)

    test = load_split(settings, "temporal", "test")
    train = load_split(settings, "temporal", "train")
    benign_ref = train[train[MULTICLASS_TARGET] == settings.labels.benign_label]
    centroid = np.asarray(bundle.pipeline.transform(benign_ref)).mean(axis=0)

    attacks = test[test[BINARY_TARGET] == 1]
    x_attacks = np.asarray(bundle.pipeline.transform(attacks))
    scores = attack_scores_transformed(bundle, x_attacks)
    flagged = np.where(scores >= threshold)[0]
    chosen = flagged[np.argsort(scores[flagged])[::-1][:n_examples]]  # most-confident hits

    examples = []
    for rank, i in enumerate(chosen, 1):
        rec = recourse_for_row(
            bundle,
            x_attacks[i : i + 1],
            centroid,
            ctrl_idx,
            threshold,
            feature_names,
            cfg.recourse_max_steps,
        )
        examples.append((rank, rec))

    report = _render_recourse(settings, examples, threshold)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote recourse report", extra={"path": str(out_path), "examples": len(examples)})
    return out_path


def _render_recourse(
    settings: Settings, examples: list[tuple[int, Recourse]], threshold: float
) -> str:
    blocks = []
    for rank, rec in examples:
        status = "cleared" if rec.flipped else "still flagged"
        lines = [
            f"### Example {rank} — score {rec.original_score:.3f} → {rec.final_score:.3f} "
            f"({status} after {len(rec.changes)} change(s))",
            "",
        ]
        if rec.changes:
            for c in rec.changes:
                lines.append(f"- **{c.direction}** `{c.feature}` by {abs(c.delta):.2f} std")
        else:
            lines.append("- no controllable change reduces the score (robust hit)")
        blocks.append("\n".join(lines))

    n_flipped = sum(rec.flipped for _, rec in examples)
    return f"""# NetSentry — Counterfactual Recourse

_Synthetic stand-in. For each flagged flow, the smallest set of moves to
attacker-controllable features (toward the benign centroid) that drops it below the
operating threshold ({settings.robustness.profile}, threshold {threshold:.3f}).
Deltas are in standardized model-space units._

SHAP explains *why* a flow fired; this explains *what would clear it* — the
analyst's what-if, and the flip side of the [robustness study](robustness.md): the
same controllable features an attacker exploits are the ones that define recourse.
**{n_flipped}/{len(examples)}** example hits can be cleared within
{settings.robustness.recourse_max_steps} changes.

{chr(10).join(f"{chr(10)}{b}" for b in blocks)}

## Why this matters

A flagged flow with a reason *and* a recourse is triage-ready: the reason points the
analyst at the behaviour, the recourse quantifies how far from the benign manifold it
sits. A hit with **no** recourse (no small controllable change clears it) is a
high-confidence detection; one cleared by a single tweak is worth a second look.
"""
