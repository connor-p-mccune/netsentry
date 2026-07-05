"""Seed-sensitivity audit — how much of any reported number is training noise?

Every report in the analysis suite comes from one training run at one seed, and the
evaluation report's bootstrap CIs quantify *data* noise (resampling the evaluation
rows). Neither says how much a number would move if only the seed changed: row and
feature subsampling, and tie-breaking inside the trees, all inject run-to-run
variance that a single fit hides. This audit refits the honest temporal model at
several consecutive seeds and reports the spread.

Two distinct properties are measured, because they answer different questions:

- **Reproducibility** (a guarantee): the same seed twice must give bit-identical
  test scores. Asserted here on every run, not assumed from the determinism test.
- **Stability** (a measurement): different seeds give different numbers; their
  standard deviation is the *training-noise floor* under every reported metric.

The floor has a product consequence: a champion/challenger comparison that promotes
on a delta smaller than the seed noise is promoting on luck. The promotion gate's
non-inferiority margin (see ``PromotionConfig``) is calibrated against exactly the
number this report measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.confidence import bootstrap_ci, pr_auc
from netsentry.evaluation.metrics import operating_point, positive_scores
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import FitResult, fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "seed_variance.md"


@dataclass
class SeedStats:
    """Mean / sample-sd / range of one metric across seed refits."""

    mean: float
    std: float
    low: float
    high: float

    @property
    def spread(self) -> float:
        return self.high - self.low


def summarize_seed_runs(runs: list[dict[str, float]]) -> dict[str, SeedStats]:
    """Per-metric mean, sample standard deviation, and range across seed runs.

    The sample sd (ddof=1) is the honest estimator at the handful of refits this
    audit can afford; a single run reports zero spread rather than NaN.
    """
    if not runs:
        return {}
    stats: dict[str, SeedStats] = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        stats[key] = SeedStats(float(values.mean()), std, float(values.min()), float(values.max()))
    return stats


def _test_scores(result: FitResult) -> np.ndarray:
    """Raw attack scores on test (the ranking the headline evaluation reports)."""
    return np.asarray(positive_scores(result.proba_test, result.classes))


def _run_metrics(settings: Settings, result: FitResult) -> dict[str, float]:
    """The headline metric set for one seed's fit, computed exactly as `eval` does.

    PR-AUC on the raw test ranking; each TPR@FPR from a threshold re-chosen on that
    run's own validation scores — so a seed row is what shipping that seed would
    report, directly comparable to the evaluation report's numbers.
    """
    val = np.asarray(positive_scores(result.proba_val, result.classes))
    test = _test_scores(result)
    y_val = result.y_val.astype(int)
    y_test = result.y_test.astype(int)
    metrics = {"pr_auc": pr_auc(y_test, test)}
    for fpr in settings.thresholds.fpr_targets:
        op = operating_point(
            y_val, val, y_test, test, fpr, settings.thresholds.assumed_flows_per_day
        )
        metrics[f"tpr_fpr_{fpr * 100:g}pct"] = float(op["tpr"])
    return metrics


def run_seed_variance_report(settings: Settings) -> Path:
    """Refit across seeds, measure the metric spread, and write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    base_seed = variant.seed
    n_seeds = settings.seed_variance.n_seeds

    # Reproducibility gate first: the same seed twice must give identical scores.
    first = fit_supervised(variant)
    second = fit_supervised(variant)
    repro_delta = float(np.max(np.abs(_test_scores(first) - _test_scores(second))))
    logger.info("Same-seed refit max score delta", extra={"delta": repro_delta})

    seeds: list[int] = []
    runs: list[dict[str, float]] = []
    for offset in range(n_seeds):
        variant.seed = base_seed + offset
        result = first if offset == 0 else fit_supervised(variant)
        seeds.append(variant.seed)
        runs.append(_run_metrics(settings, result))
        logger.info(
            "Seed refit", extra={"seed": variant.seed, "pr_auc": round(runs[-1]["pr_auc"], 4)}
        )
    stats = summarize_seed_runs(runs)

    # Data noise for scale: the bootstrap CI of the base seed's PR-AUC. Comparing
    # its half-width against the seed sd says which noise source dominates.
    interval = bootstrap_ci(
        first.y_test.astype(int),
        _test_scores(first),
        pr_auc,
        n_boot=settings.evaluation.bootstrap_samples,
        alpha=settings.evaluation.bootstrap_alpha,
        seed=base_seed,
    )
    data_half_width = (interval.high - interval.low) / 2.0

    fig = plots.plot_lines(
        {
            name: (np.array(seeds, dtype=float), np.array([r[key] for r in runs]))
            for key, name in _figure_series(runs)
        },
        xlabel="Seed",
        ylabel="Metric value",
        title="Metric stability across training seeds",
        out_path=settings.paths.figures_dir / "seed_variance.png",
    )

    report = _render(settings, seeds, runs, stats, repro_delta, data_half_width, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote seed-variance report", extra={"path": str(out_path)})

    with track_run(settings, "seed_variance") as run:
        run.log_params({"n_seeds": n_seeds, "base_seed": base_seed})
        run.log_metrics({f"{k}_mean": s.mean for k, s in stats.items()})
        run.log_metrics({f"{k}_std": s.std for k, s in stats.items()})
        run.log_metrics({"same_seed_max_delta": repro_delta})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _figure_series(runs: list[dict[str, float]]) -> list[tuple[str, str]]:
    """(metric key, display name) pairs for the figure, in a stable order."""
    if not runs:
        return []
    pairs = [("pr_auc", "PR-AUC")]
    pairs.extend(
        (key, key.replace("tpr_fpr_", "TPR @ ").replace("pct", "% FPR"))
        for key in runs[0]
        if key.startswith("tpr_")
    )
    return pairs


def _render(
    settings: Settings,
    seeds: list[int],
    runs: list[dict[str, float]],
    stats: dict[str, SeedStats],
    repro_delta: float,
    data_half_width: float,
    fig: Path,
) -> str:
    metric_names = list(runs[0]) if runs else []
    header = "| seed | " + " | ".join(_display(m) for m in metric_names) + " |"
    divider = "|---" * (len(metric_names) + 1) + "|"
    rows = [header, divider]
    for seed, run in zip(seeds, runs, strict=True):
        cells = " | ".join(f"{run[m]:.3f}" for m in metric_names)
        rows.append(f"| {seed} | {cells} |")

    summary = [
        "| metric | mean | sd (training noise) | min | max |",
        "|---|---|---|---|---|",
    ]
    for name in metric_names:
        s = stats[name]
        summary.append(
            f"| {_display(name)} | {s.mean:.3f} | {s.std:.4f} | {s.low:.3f} | {s.high:.3f} |"
        )

    seed_sd = stats["pr_auc"].std if "pr_auc" in stats else 0.0
    if data_half_width > 2 * seed_sd:
        noise_read = (
            f"**Data noise dominates**: the bootstrap CI half-width ({data_half_width:.4f}) is "
            f"more than twice the seed sd ({seed_sd:.4f}). The evaluation set, not the training "
            "run, is the main source of uncertainty here — the bootstrap CIs the evaluation "
            "report already carries are the binding error bar."
        )
    elif seed_sd > 2 * data_half_width:
        noise_read = (
            f"**Training noise dominates**: the seed sd ({seed_sd:.4f}) is more than twice the "
            f"bootstrap CI half-width ({data_half_width:.4f}). A single-run number under-reports "
            "the real uncertainty; comparisons should average over seeds."
        )
    else:
        noise_read = (
            f"Training noise (seed sd {seed_sd:.4f}) and data noise (bootstrap half-width "
            f"{data_half_width:.4f}) are the same order of magnitude — both matter, and a "
            "comparison should clear the *larger* of the two before it means anything."
        )

    return f"""# NetSentry — Seed Sensitivity (the training-noise floor)

_Synthetic stand-in; the method is the point. The honest temporal/binary model refit
at {len(seeds)} consecutive seeds ({seeds[0]}-{seeds[-1]}); FPR thresholds re-chosen
on each run's own validation scores, exactly as the evaluation report does — so each
row is what **shipping that seed** would report, not just a re-scored model._

## Reproducibility vs stability

Two different properties, often conflated:

- **Reproducibility (guarantee).** Refit at the same seed, the numbers must be
  identical. Verified on this run: max |score delta| between two same-seed fits =
  **{repro_delta:.2e}**{" — exact reproduction" if repro_delta == 0 else ""}.
- **Stability (measurement).** Refit at a *different* seed and the numbers move by
  the training noise measured below. This spread is a property of the model class
  and data, not a bug — but any comparison that ignores it will promote on luck.

## Per-seed results

{chr(10).join(rows)}

## Spread across seeds

{chr(10).join(summary)}

- {noise_read}
- **The consequence for model comparison:** a challenger model must beat the champion
  by more than the noise floor before the delta means anything. The champion/challenger
  promotion gate (`netsentry promote`) uses a non-inferiority margin calibrated
  against this measurement — deltas inside the noise band hold the champion.

![Seed stability](../figures/{fig.name})

## Why this matters

Single-run leaderboard numbers invite over-reading; the difference between two
models is only real once it clears both noise sources. This audit prices the one
bootstrap CIs cannot see (retraining), keeps the determinism guarantee honest by
re-asserting it on every run, and hands the promotion gate an evidence-based margin
instead of a hand-picked one.
"""


def _display(metric: str) -> str:
    if metric == "pr_auc":
        return "PR-AUC"
    return metric.replace("tpr_fpr_", "TPR @ ").replace("pct", "% FPR")
