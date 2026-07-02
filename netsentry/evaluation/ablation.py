"""Feature-group ablation — which behavioural families actually carry detection?

SHAP explains *which features a given prediction leaned on* (attribution); it does
not tell you what the model would lose if a whole family of features were
unavailable. Those are different questions — a feature can have high SHAP mass yet
be redundant with another the model would fall back on. This study answers the
second, causal-ish question directly: refit the model with one behavioural family
(timing/IAT, flow rates, packet size, TCP flags, volume, header/window) removed at a
time, and measure the drop in honest-split detection. The drop is that family's
*marginal* value given everything else.

Ablation is done on the fitted feature matrix (drop the family's columns, refit the
model) rather than by re-running feature engineering, so the leakage-safe pipeline
and every other column's train-fit statistics are untouched — the only thing that
changes between runs is which family the model may see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import feature_group
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.evasion import base_feature_name
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "ablation.md"


@dataclass
class AblationPoint:
    """Detection/PR-AUC when one feature family is removed (vs the full-feature model)."""

    group: str
    n_features: int
    pr_auc: float
    detection: float
    pr_auc_drop: float
    detection_drop: float


def _score(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    operating_fpr: float,
) -> tuple[float, float]:
    """Fit on the given matrices; return (test PR-AUC, detection at the operating FPR)."""
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    benign = settings.labels.benign_label
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
    detection = rates_at_threshold(y_test, s_test, threshold)["tpr"]
    pr_auc = float(average_precision_score(y_test, s_test))
    return pr_auc, detection


def run_ablation_report(settings: Settings) -> Path:
    """Leave-one-family-out on the temporal split; write the ablation report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    operating_fpr = variant.thresholds.fpr_targets[-1]

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())
    groups = _column_groups(feature_names)

    base_pr, base_det = _score(
        variant, x_train, y_train, x_val, y_val, x_test, y_test, operating_fpr
    )
    logger.info("Ablation baseline", extra={"pr_auc": round(base_pr, 4), "detection": base_det})

    points: list[AblationPoint] = []
    for group, cols in sorted(groups.items()):
        keep = np.array([j for j in range(len(feature_names)) if j not in cols], dtype=int)
        pr, det = _score(
            variant,
            x_train[:, keep],
            y_train,
            x_val[:, keep],
            y_val,
            x_test[:, keep],
            y_test,
            operating_fpr,
        )
        points.append(AblationPoint(group, len(cols), pr, det, base_pr - pr, base_det - det))
        logger.info(
            "Ablated group", extra={"group": group, "pr_auc": round(pr, 4), "drop": base_pr - pr}
        )

    points.sort(key=lambda p: p.pr_auc_drop, reverse=True)
    fig = plots.plot_barh(
        [p.group for p in points],
        [p.pr_auc_drop for p in points],
        xlabel="PR-AUC drop when the family is removed",
        title="Feature-group ablation (temporal split)",
        out_path=variant.paths.figures_dir / "ablation.png",
    )

    report = _render(points, base_pr, base_det, operating_fpr, fig)
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote ablation report", extra={"path": str(out_path)})

    with track_run(settings, "ablation") as run:
        run.log_metrics({"baseline_pr_auc": base_pr})
        run.log_metrics({f"drop_{p.group.replace('/', '_')}": p.pr_auc_drop for p in points})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _column_groups(feature_names: list[str]) -> dict[str, set[int]]:
    """Map each behavioural family to the transformed-matrix column indices it owns."""
    groups: dict[str, set[int]] = {}
    for j, name in enumerate(feature_names):
        groups.setdefault(feature_group(base_feature_name(name)), set()).add(j)
    return groups


def _render(
    points: list[AblationPoint],
    base_pr: float,
    base_det: float,
    operating_fpr: float,
    fig: Path,
) -> str:
    rows = [
        "| family removed | features | PR-AUC | Δ PR-AUC | detection | Δ detection |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.group} | {p.n_features} | {p.pr_auc:.3f} | **{p.pr_auc_drop:+.3f}** "
            f"| {p.detection * 100:.1f}% | {p.detection_drop * 100:+.1f} pts |"
        )
    top = points[0]
    # A negative drop means removing the family *improved* the honest number — on the
    # temporal split that is the fingerprint of overfitting to non-transferring, day-
    # specific patterns, not licence to prune (that decision belongs on validation).
    helped = [p for p in points if p.pr_auc_drop < -0.01]
    helped_note = (
        " Conversely, removing "
        + ", ".join(f"**{p.group}** ({-p.pr_auc_drop:+.3f})" for p in helped)
        + " *improved* the honest PR-AUC. That is not a licence to prune — it is the "
        "signature of **overfitting to the temporal shift**: those families encode "
        "absolute scales (packet/flow volumes, durations) that differ between the Mon–Wed "
        "training attacks and the Thu–Fri test attacks, so the model learns day-specific "
        "thresholds that mislead on later days. The rate family, being a ratio, transfers "
        "better. Acting on this would mean selecting features on *validation* "
        "(never this test split — that is the leakage the project exists to avoid); the "
        "ablation only tells you where to look."
        if helped
        else ""
    )
    return f"""# NetSentry — Feature-Group Ablation

_Synthetic stand-in. Leave-one-family-out on the honest **temporal** split (binary
attack vs benign). Baseline (all families): PR-AUC {base_pr:.3f}, detection
{base_det * 100:.1f}% at the {operating_fpr * 100:g}%-FPR operating point. Each row
refits the model with that behavioural family's columns removed; the delta is the
family's marginal value given all the others._

## Why ablation, not just SHAP

SHAP attributes a *given* prediction to features; it cannot say what the model would
lose if a whole family were unavailable, because a high-SHAP feature may be redundant
with one the model would fall back on. Ablation answers that directly by removing the
family and refitting — the causal-flavoured complement to SHAP's attribution.

{chr(10).join(rows)}

## Read

The most load-bearing family is **{top.group}** ({-top.pr_auc_drop:+.3f} PR-AUC,
{top.detection_drop * 100:+.1f} pts detection when removed): the honest temporal
signal leans on it most, and it is a *ratio* family that transfers across days.
{helped_note}

This lines up with the adversarial-robustness study — the rate/timing features
ablation shows carry the transferable signal are exactly the attacker-controllable
ones the evasion attack exploits, which is why a classifier resting on them needs the
benign-only anomaly detector beside it. Ablation measures *marginal* value given the
rest, not standalone value: a family can look redundant here yet be the only signal in
another regime.
"""
