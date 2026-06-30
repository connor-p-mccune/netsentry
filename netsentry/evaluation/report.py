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

from netsentry.data.split import load_split
from netsentry.evaluation import confidence as C
from netsentry.evaluation import metrics as M
from netsentry.evaluation import plots
from netsentry.evaluation.calibration import calibration_summary
from netsentry.explain.shap_explainer import ShapExplainer
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
    confidence_md, confidence_metrics = _confidence_section(settings, headline, reference)
    calibration_md, reliability_fig, calibration_metrics = _calibration_section(settings, headline)
    explain_md, shap_fig = _explain_section(settings, headline)

    figures = {"pr": pr_fig, "roc": roc_fig, "threshold": thr_fig, "confusion": cm_fig}
    report = _render_markdown(
        settings=settings,
        headline=headline,
        headline_pr=headline_pr,
        reference_pr=reference_pr,
        gap=gap,
        operating_md=operating_md,
        per_class_md=per_class_md,
        confidence_md=confidence_md,
        calibration_md=calibration_md,
        explain_md=explain_md,
        figures=figures,
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
                **calibration_metrics,
                **confidence_metrics,
            }
        )
        artifacts = [pr_fig, roc_fig, thr_fig, cm_fig, out_path]
        if reliability_fig is not None:
            artifacts.append(reliability_fig)
        if shap_fig is not None:
            artifacts.append(shap_fig)
        for fig in artifacts:
            run.log_artifact(fig)

    return out_path


def _confidence_section(
    settings: Settings, headline: FitResult, reference: FitResult
) -> tuple[str, dict[str, float]]:
    """Bootstrap CIs for PR-AUC and TPR@FPR, plus a significance test on the gap."""
    cfg = settings.evaluation
    n_boot, alpha, seed = cfg.bootstrap_samples, cfg.bootstrap_alpha, settings.seed
    level = round((1 - alpha) * 100)

    y_val, s_val, y_test, s_test = _binary_scores(headline)
    _, _, r_y_test, r_s_test = _binary_scores(reference)

    pr_h = C.bootstrap_ci(y_test, s_test, C.pr_auc, n_boot=n_boot, alpha=alpha, seed=seed)
    pr_r = C.bootstrap_ci(r_y_test, r_s_test, C.pr_auc, n_boot=n_boot, alpha=alpha, seed=seed)
    gap = C.independent_diff(
        y_test, s_test, r_y_test, r_s_test, C.pr_auc, n_boot=n_boot, alpha=alpha, seed=seed
    )

    rows = [
        f"| metric | estimate | {level}% CI |",
        "|---|---|---|",
        f"| PR-AUC — temporal (honest) | {pr_h.point:.3f} | [{pr_h.low:.3f}, {pr_h.high:.3f}] |",
        f"| PR-AUC — stratified (optimistic) | {pr_r.point:.3f} | "
        f"[{pr_r.low:.3f}, {pr_r.high:.3f}] |",
    ]
    logged = {"pr_auc_temporal_ci_low": pr_h.low, "pr_auc_gap_p_value": gap.p_value}
    for fpr in settings.thresholds.fpr_targets:
        thr = M.threshold_at_fpr(y_val, s_val, fpr)
        ci = C.bootstrap_ci(
            y_test, s_test, C.tpr_at_threshold(thr), n_boot=n_boot, alpha=alpha, seed=seed
        )
        rows.append(
            f"| detection @ {fpr * 100:g}% FPR (temporal) | {ci.point * 100:.1f}% | "
            f"[{ci.low * 100:.1f}%, {ci.high * 100:.1f}%] |"
        )
        logged[f"tpr_at_fpr_{fpr}_ci_low"] = ci.low

    maj = headline.baselines["majority"]["pr_auc"]
    p_str = f"{gap.p_value:.3f}" if gap.p_value > 0 else f"< {1 / n_boot:.3f}"
    sig = "statistically significant" if gap.low > 0 else "not statistically significant"
    beats = "excludes" if pr_h.low > maj else "includes"
    verdict = "beats" if pr_h.low > maj else "does not clearly beat"
    md = (
        f"## Statistical significance (bootstrap, {level}% CIs, {n_boot:,} resamples)\n\n"
        "A point estimate invites over-reading; the headline numbers come with "
        "percentile-bootstrap intervals so the comparison can be judged, not assumed.\n\n"
        f"{chr(10).join(rows)}\n\n"
        f"The over-optimism gap (stratified minus temporal) is **{gap.diff:+.3f}** "
        f"({level}% CI [{gap.low:+.3f}, {gap.high:+.3f}], bootstrap p = {p_str}) — the gap "
        f"is **{sig}**. The temporal PR-AUC interval {beats} the majority baseline "
        f"({maj:.3f}), so the model **{verdict}** chance at the {level}% level. This is the "
        "honest-vs-optimistic finding restated with uncertainty attached.\n"
    )
    return md, logged


def _calibration_section(
    settings: Settings, headline: FitResult
) -> tuple[str, Path | None, dict[str, float]]:
    """Calibration diagnostics (Brier/ECE/MCE) on the headline test scores.

    Reports raw vs calibrated because tree scores are not probabilities; the
    calibrator is monotonic, so this changes the *meaning* of the score, not the
    ranking (PR-AUC/TPR@FPR above are unaffected).
    """
    y_test, raw = headline.y_test.astype(int), M.positive_scores(
        headline.proba_test, headline.classes
    )
    calibrator = headline.bundle.calibrator
    if calibrator is None:
        return ("", None, {})
    calibrated = calibrator.transform(raw)
    raw_m = calibration_summary(y_test, raw, settings.monitoring.psi_bins)
    cal_m = calibration_summary(y_test, calibrated, settings.monitoring.psi_bins)

    fig = plots.plot_reliability_curve(
        {
            "raw tree score": (y_test, raw),
            f"calibrated ({calibrator.method})": (y_test, calibrated),
        },
        settings.paths.figures_dir / "reliability_curve.png",
        n_bins=settings.monitoring.psi_bins,
    )
    rows = [
        "| score | Brier ↓ | ECE ↓ | MCE ↓ |",
        "|---|---|---|---|",
        f"| raw tree output | {raw_m['brier']:.4f} | {raw_m['ece']:.4f} | {raw_m['mce']:.4f} |",
        f"| **calibrated ({calibrator.method})** | **{cal_m['brier']:.4f}** | "
        f"**{cal_m['ece']:.4f}** | **{cal_m['mce']:.4f}** |",
    ]
    md = (
        "## Probability calibration\n\n"
        "Gradient-boosted scores rank well but are **not probabilities** — a raw "
        f"score of 0.9 need not mean a 90% attack rate. We fit **{calibrator.method}** "
        "calibration on the validation split and apply it to the served probability "
        "and the decision thresholds. Test-set diagnostics (lower is better):\n\n"
        f"{chr(10).join(rows)}\n\n"
        "The map is monotonic, so it preserves the ranking of flows — the PR-AUC "
        "above is the model's discriminative power either way. Calibration changes "
        "only how that score maps to a probability, which is what makes a stated FP "
        "budget or a reported `attack_probability` trustworthy.\n\n"
        f"![Reliability diagram](../figures/{fig.name})\n"
    )
    logged = {
        "calib_brier_raw": raw_m["brier"],
        "calib_brier_calibrated": cal_m["brier"],
        "calib_ece_raw": raw_m["ece"],
        "calib_ece_calibrated": cal_m["ece"],
    }
    return md, fig, logged


def _explain_section(settings: Settings, headline: FitResult) -> tuple[str, Path | None]:
    """Global SHAP importance for the headline model (skipped gracefully on error)."""
    try:
        background_rows = load_split(settings, "temporal", "test")
        sample = background_rows.sample(min(200, len(background_rows)), random_state=settings.seed)
        background = headline.bundle.pipeline.transform(sample)
        explainer = ShapExplainer(headline.bundle)
        fig = explainer.plot_global(background, settings.paths.figures_dir / "shap_global.png")
        rows = ["| rank | feature | importance |", "|---|---|---|"]
        for i, (name, value) in enumerate(explainer.global_importance(background, top_n=10), 1):
            rows.append(f"| {i} | {name} | {value:.4f} |")
        body = "\n".join(rows)
        md = (
            "## Explainability — SHAP global importance\n\n"
            f"Attribution method: **{explainer.mode}**. The features driving attack "
            "predictions (per-prediction contributions are returned by the API):\n\n"
            f"{body}\n\n![Global feature importance](../figures/{fig.name})\n"
        )
        return md, fig
    except Exception as exc:  # explanations are valuable but must not break the report
        logger.warning("SHAP section skipped (%s)", exc)
        return "", None


def _render_markdown(
    *,
    settings: Settings,
    headline: FitResult,
    headline_pr: dict[str, float],
    reference_pr: dict[str, float],
    gap: float,
    operating_md: str,
    per_class_md: str,
    confidence_md: str,
    calibration_md: str,
    explain_md: str,
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

{confidence_md}

## Per-class — stratified multiclass ("name the attack")

Multiclass naming is evaluated on the stratified split (all classes appear in
training); on the temporal split it is degenerate because attack classes are
disjoint across the day boundary.

{per_class_md}

![Confusion matrix]({fig_rel['confusion'].as_posix()})

{calibration_md}
{explain_md}
## Notes

- Accuracy is intentionally absent from the headline: on ~80%-benign data it is
  ~0.8 for a model that detects nothing.
- A near-perfect score here would indicate leakage, not skill — see `NOTES.md`.
"""
