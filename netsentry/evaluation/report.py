"""Generate the evaluation report: operational metrics + the honesty gap.

Leads with PR-AUC, per-class metrics, and TPR@fixed-FPR (never accuracy), and
explicitly contrasts the honest **temporal** split with the optimistic
**stratified** split so the over-optimism gap is visible and explained.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from netsentry.evaluation import metrics as M
from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import FitResult, fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "evaluation.md"


def _fit(
    settings: Settings,
    strategy: Literal["temporal", "stratified"],
    task: Literal["binary", "multiclass"],
) -> FitResult:
    variant = settings.model_copy(deep=True)
    variant.split.strategy = strategy
    variant.supervised.task = task
    variant.mlflow.enabled = False  # the report owns the tracking run
    return fit_supervised(variant)


def _binary_scores(result: FitResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val = M.positive_scores(result.proba_val, result.classes)
    test = M.positive_scores(result.proba_test, result.classes)
    return result.y_val.astype(int), val, result.y_test.astype(int), test


def _operating_table(result: FitResult, settings: Settings) -> tuple[str, dict[str, float]]:
    y_val, s_val, y_test, s_test = _binary_scores(result)
    rows = [
        "| FPR budget | detection (TPR) | achieved FPR | ~false alerts/day |",
        "|---|---|---|---|",
    ]
    logged: dict[str, float] = {}
    for fpr in settings.thresholds.fpr_targets:
        point = M.operating_point(
            y_val, s_val, y_test, s_test, fpr, settings.thresholds.assumed_flows_per_day
        )
        rows.append(
            f"| {fpr * 100:.1f}% | {point['tpr'] * 100:.1f}% | "
            f"{point['fpr'] * 100:.3f}% | {point['alerts_per_day']:,.0f} |"
        )
        logged[f"tpr_at_fpr_{fpr}"] = point["tpr"]
    return "\n".join(rows), logged


def _per_class_table(result: FitResult) -> str:
    preds = result.classes[result.proba_test.argmax(axis=1)]
    labels = [str(c) for c in result.classes]
    report = M.per_class_report(result.y_test.astype(str), preds.astype(str), labels=labels)
    rows = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for label in labels:
        m = report[label]
        rows.append(
            f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | "
            f"{m['f1']:.2f} | {int(m['support'])} |"
        )
    for name, agg in (("macro avg", report["__macro__"]), ("weighted avg", report["__weighted__"])):
        rows.append(
            f"| **{name}** | {agg['precision']:.2f} | {agg['recall']:.2f} | {agg['f1']:.2f} | |"
        )
    return "\n".join(rows)


def run_evaluation(settings: Settings) -> Path:
    """Fit headline/reference/multiclass variants, render figures, write the report."""
    headline = _fit(settings, "temporal", "binary")
    reference = _fit(settings, "stratified", "binary")
    multiclass = _fit(settings, "stratified", "multiclass")

    _, _, h_y_test, h_scores = _binary_scores(headline)
    _, _, r_y_test, r_scores = _binary_scores(reference)
    headline_pr = M.binary_summary(h_y_test, h_scores)
    reference_pr = M.binary_summary(r_y_test, r_scores)
    gap = reference_pr["pr_auc"] - headline_pr["pr_auc"]

    figures_dir = settings.paths.figures_dir
    curves = {
        "temporal (honest)": (h_y_test, h_scores),
        "stratified (optimistic)": (r_y_test, r_scores),
    }
    pr_fig = plots.plot_pr_curves(curves, figures_dir / "pr_curve.png")
    roc_fig = plots.plot_roc_curves(curves, figures_dir / "roc_curve.png")
    thr_fig = plots.plot_threshold_curve(h_y_test, h_scores, figures_dir / "threshold_curve.png")
    mc_labels = [str(c) for c in multiclass.classes]
    mc_preds = multiclass.classes[multiclass.proba_test.argmax(axis=1)]
    cm = M.confusion(multiclass.y_test.astype(str), mc_preds.astype(str), mc_labels)
    cm_fig = plots.plot_confusion_matrix(cm, mc_labels, figures_dir / "confusion_matrix.png")

    operating_md, operating_metrics = _operating_table(headline, settings)
    per_class_md = _per_class_table(multiclass)

    report = _render_markdown(
        settings=settings,
        headline=headline,
        headline_pr=headline_pr,
        reference_pr=reference_pr,
        gap=gap,
        operating_md=operating_md,
        per_class_md=per_class_md,
        figures={"pr": pr_fig, "roc": roc_fig, "threshold": thr_fig, "confusion": cm_fig},
    )
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote evaluation report", extra={"path": str(out_path)})

    with track_run(settings, "evaluation") as run:
        run.log_metrics(
            {
                "headline_pr_auc": headline_pr["pr_auc"],
                "reference_pr_auc": reference_pr["pr_auc"],
                "pr_auc_gap": gap,
                **{f"headline_{k}": v for k, v in operating_metrics.items()},
            }
        )
        for fig in (pr_fig, roc_fig, thr_fig, cm_fig, out_path):
            run.log_artifact(fig)

    return out_path


def _render_markdown(
    *,
    settings: Settings,
    headline: FitResult,
    headline_pr: dict[str, float],
    reference_pr: dict[str, float],
    gap: float,
    operating_md: str,
    per_class_md: str,
    figures: dict[str, Path],
) -> str:
    maj = headline.baselines["majority"]["pr_auc"]
    fig_rel = {k: Path("..") / "figures" / v.name for k, v in figures.items()}
    return f"""# NetSentry — Evaluation Report

_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. Numbers below are on
the **synthetic** CIC-IDS2017 stand-in unless you have run on the real dataset;
the methodology and framing are identical either way._

## Headline — temporal (by-day) split, attack vs benign

The honest number: trained on earlier days, tested on later days (largely
**novel** attack types). This is what generalisation to tomorrow's traffic looks
like.

- **PR-AUC: {headline_pr['pr_auc']:.3f}** (majority-class baseline: {maj:.3f})
- ROC-AUC: {headline_pr.get('roc_auc', float('nan')):.3f}

### Operating points (threshold chosen on validation at a fixed FP budget)

{operating_md}

> A SOC reads the first row as: "at a {settings.thresholds.fpr_targets[0] * 100:.1f}%
> false-positive budget, the detector catches this fraction of attacks, at roughly
> this many false alerts/day." False positives — not misses — are what cause alert
> fatigue, so the operating point matters more than any AUC.

![PR curve]({fig_rel['pr'].as_posix()})

## The honesty gap — temporal vs stratified

| Split | Binary PR-AUC |
|---|---|
| **Temporal (honest, headline)** | **{headline_pr['pr_auc']:.3f}** |
| Stratified (optimistic reference) | {reference_pr['pr_auc']:.3f} |
| **Gap (over-optimism)** | **{gap:+.3f}** |

A naive shuffled split scores markedly higher because near-duplicate flows from
one attack burst land on both sides and all attack types are seen in training.
Reporting the temporal number — and this gap — is the whole point.

![ROC curves]({fig_rel['roc'].as_posix()})
![Threshold trade-off]({fig_rel['threshold'].as_posix()})

## Per-class — stratified multiclass ("name the attack")

Multiclass naming is evaluated on the stratified split (all classes appear in
training); on the temporal split it is degenerate because attack classes are
disjoint across the day boundary.

{per_class_md}

![Confusion matrix]({fig_rel['confusion'].as_posix()})

## Notes

- Accuracy is intentionally absent from the headline: on ~80%-benign data it is
  ~0.8 for a model that detects nothing.
- A near-perfect score here would indicate leakage, not skill — see `NOTES.md`.
"""
