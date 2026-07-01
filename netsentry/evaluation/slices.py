"""Per-attack-class detection slices on the honest temporal split.

The headline PR-AUC is one number over all attacks; this breaks detection down by
*which* later-day attack class is caught at the operating threshold. Because the
temporal split makes the test-day attacks largely novel to the model, this is the
concrete "which unknown attacks do we actually catch, and which slip through" view —
exactly the question the aggregate number hides, and a natural place to target the
anomaly detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "slices.md"


@dataclass
class ClassSlice:
    """Detection rate for one attack class at the operating threshold."""

    label: str
    support: int
    detection: float


def per_class_detection(
    labels: np.ndarray, scores: np.ndarray, threshold: float, benign_label: str
) -> list[ClassSlice]:
    """Detection rate (recall at ``threshold``) for each attack class."""
    labels = np.asarray(labels).astype(str)
    scores = np.asarray(scores)
    slices: list[ClassSlice] = []
    for label in sorted(set(labels)):
        if label == benign_label:
            continue
        mask = labels == label
        detection = float(np.mean(scores[mask] >= threshold)) if mask.any() else 0.0
        slices.append(ClassSlice(label, int(mask.sum()), detection))
    return slices


def run_slices_report(settings: Settings) -> Path:
    """Fit the temporal binary model and report detection per later-day attack class."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    # Detection at a fixed FPR is a ranking property, so use the raw attack score —
    # matching the evaluation report's operating points (isotonic calibration's ties
    # would distort a strict threshold without changing the ranking).
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)

    benign = settings.labels.benign_label
    y_val_bin = result.y_val.astype(int)  # binary task -> y_val is already the 0/1 target
    test = load_split(settings, "temporal", "test")
    labels = test[MULTICLASS_TARGET].to_numpy()

    profile_fpr = settings.thresholds.fpr_targets[-1]  # the looser budget (more detection)
    threshold = threshold_at_fpr(y_val_bin, s_val, profile_fpr)
    slices = per_class_detection(labels, s_test, threshold, benign)

    fig = plots.plot_barh(
        [s.label for s in slices],
        [s.detection for s in slices],
        xlabel=f"Detection rate @ {profile_fpr * 100:g}% FPR",
        title="Per-attack-class detection (temporal split)",
        out_path=settings.paths.figures_dir / "slices.png",
    )

    report = _render(slices, profile_fpr, threshold, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote slices report", extra={"path": str(out_path), "classes": len(slices)})

    with track_run(settings, "slices") as run:
        run.log_metrics({f"detection_{s.label}": s.detection for s in slices})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(slices: list[ClassSlice], fpr: float, threshold: float, fig: Path) -> str:
    present = [s for s in slices if s.support > 0]
    rows = ["| attack class | test support | detection |", "|---|---|---|"]
    for s in sorted(present, key=lambda s: s.detection, reverse=True):
        rows.append(f"| {s.label} | {s.support:,} | {s.detection * 100:.1f}% |")
    if present:
        best = max(present, key=lambda s: s.detection)
        worst = min(present, key=lambda s: s.detection)
        read = (
            f"Best caught: **{best.label}** ({best.detection * 100:.0f}%); most evasive: "
            f"**{worst.label}** ({worst.detection * 100:.0f}%)."
        )
    else:
        read = "No attack classes present in the temporal test split."
    return f"""# NetSentry — Per-Class Detection Slices

_Synthetic stand-in. Detection rate per attack class on the honest **temporal** test
split, at the {fpr * 100:g}%-FPR operating point (raw attack score threshold
{threshold:.3f}, matching the evaluation report)._

The aggregate PR-AUC hides which attacks are caught. On the temporal split the
test-day attacks are largely **novel** to the model, so this is the concrete "known
vs unknown" breakdown.

{chr(10).join(rows)}

{read} Low-detection classes are exactly where the benign-only anomaly detector
earns its keep — the supervised model cannot recall an attack type it never trained
on, which is the whole argument for pairing the two.
"""
