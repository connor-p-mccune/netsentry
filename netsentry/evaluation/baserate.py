"""Base-rate stress test: what the operating points mean at a realistic prevalence.

Axelsson's base-rate fallacy (1999) is the oldest hard truth in intrusion
detection: because hostile traffic is a tiny fraction of the whole, the useful
number is not the false-positive *rate* but the **precision of the alert queue**
— and Bayes' rule makes that precision collapse as the attack prevalence falls,
no matter how good the detector's conditional rates look. A 0.1%-FPR budget
sounds strict until it meets a one-in-ten-thousand base rate.

This study takes the deployed operating points (thresholds chosen on validation
at the configured FPR budgets, conditional TPR/FPR measured on the honest
temporal test split) and sweeps the production attack prevalence across orders
of magnitude, reporting for each: alert precision, the daily alert volume and
its false share, and the two inversions that make the fallacy concrete —

- the **break-even prevalence** below which most alerts are false, and
- the **FPR the detector would need** for the alert queue to hit a target
  precision at a given prevalence (usually orders of magnitude below any budget
  a threshold can realistically be held to).

The test-mix precision the evaluation report shows (~22% attack on the
stand-in) is honest *for that mix* but not a deployment claim; this report is
the bridge between the conditional rates and what an analyst's queue would
actually contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "base_rate.md"


def bayes_precision(tpr: float, fpr: float, prior: float) -> float:
    """Alert precision at attack prevalence ``prior`` given conditional TPR/FPR.

    P(attack | alert) = pi*TPR / (pi*TPR + (1-pi)*FPR) — Bayes' rule, the heart of
    the base-rate fallacy: precision depends on the prevalence at least as much as
    on the detector.
    """
    numerator = prior * tpr
    denominator = numerator + (1.0 - prior) * fpr
    return float(numerator / denominator) if denominator > 0 else 0.0


def required_fpr(tpr: float, prior: float, precision: float) -> float:
    """The FPR needed to reach ``precision`` at prevalence ``prior`` (TPR fixed).

    Inverts Bayes' rule for the false-positive rate — the "how good would the
    detector have to be" direction of the fallacy.
    """
    if precision <= 0 or prior >= 1:
        return float("inf")
    return float(prior * tpr * (1.0 - precision) / (precision * (1.0 - prior)))


def break_even_prior(tpr: float, fpr: float) -> float:
    """The prevalence at which alert precision crosses 50% (below it, most alerts lie).

    Setting precision = 1/2 in Bayes' rule gives pi* = FPR / (TPR + FPR).
    """
    total = tpr + fpr
    return float(fpr / total) if total > 0 else 1.0


@dataclass
class OperatingPointProfile:
    """One FPR budget's measured conditional rates and their prevalence sweep."""

    target_fpr: float
    tpr: float  # conditional detection rate, measured on the honest test split
    fpr: float  # realized false-positive rate at the val-chosen threshold
    rows: list[dict[str, float]]  # one row per swept prior


def sweep_priors(
    tpr: float, fpr: float, priors: list[float], flows_per_day: int
) -> list[dict[str, float]]:
    """Alert-queue composition per prevalence: precision, volumes, false share."""
    rows: list[dict[str, float]] = []
    for prior in priors:
        precision = bayes_precision(tpr, fpr, prior)
        caught = flows_per_day * prior * tpr
        false_alerts = flows_per_day * (1.0 - prior) * fpr
        rows.append(
            {
                "prior": prior,
                "precision": precision,
                "alerts_per_day": caught + false_alerts,
                "false_alerts_per_day": false_alerts,
                "attacks_caught_per_day": caught,
            }
        )
    return rows


def run_base_rate_report(settings: Settings) -> Path:
    """Sweep the deployed operating points across production base rates; write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    # Raw scores, matching the evaluation report's operating points (thresholds are
    # a ranking property; calibration would only add isotonic ties).
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    y_val = result.y_val.astype(int)
    y_test = result.y_test.astype(int)

    cfg = settings.base_rate
    flows = settings.thresholds.assumed_flows_per_day
    profiles: list[OperatingPointProfile] = []
    for budget in sorted(settings.thresholds.fpr_targets):
        threshold = threshold_at_fpr(y_val, s_val, budget)
        rates = rates_at_threshold(y_test, s_test, threshold)
        profiles.append(
            OperatingPointProfile(
                target_fpr=budget,
                tpr=rates["tpr"],
                fpr=rates["fpr"],
                rows=sweep_priors(rates["tpr"], rates["fpr"], cfg.priors, flows),
            )
        )

    priors = np.array(cfg.priors, dtype=float)
    fig = plots.plot_lines(
        {
            f"precision @ {p.target_fpr * 100:g}% FPR budget": (
                priors,
                np.array([row["precision"] for row in p.rows]),
            )
            for p in profiles
        },
        xlabel="Production attack prevalence (fraction of flows)",
        ylabel="Alert precision  P(attack | alert)",
        title="The base-rate fallacy, measured (Axelsson 1999)",
        out_path=settings.paths.figures_dir / "base_rate.png",
        vlines={"assumed production rate": settings.cost.production_attack_rate},
        xscale="log",
    )

    report = _render(settings, profiles, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote base-rate report", extra={"path": str(out_path)})

    with track_run(settings, "base_rate") as run:
        for p in profiles:
            run.log_metrics(
                {
                    f"break_even_prior_fpr{p.target_fpr:g}": break_even_prior(p.tpr, p.fpr),
                    f"precision_at_1pct_prior_fpr{p.target_fpr:g}": bayes_precision(
                        p.tpr, p.fpr, 0.01
                    ),
                }
            )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _budget_section(p: OperatingPointProfile) -> str:
    rows = [
        "| prevalence | alerts/day | of which false | precision | attacks caught/day |",
        "|---|---|---|---|---|",
    ]
    for row in p.rows:
        rows.append(
            f"| {row['prior']:.4%} | {row['alerts_per_day']:,.0f} "
            f"| {row['false_alerts_per_day']:,.0f} | {row['precision']:.1%} "
            f"| {row['attacks_caught_per_day']:,.0f} |"
        )
    return (
        f"### {p.target_fpr * 100:g}% FPR budget "
        f"(measured TPR {p.tpr:.1%}, realized FPR {p.fpr:.3%})\n\n" + "\n".join(rows)
    )


def _render(settings: Settings, profiles: list[OperatingPointProfile], fig: Path) -> str:
    cfg = settings.base_rate
    flows = settings.thresholds.assumed_flows_per_day
    tight = profiles[0]
    be = break_even_prior(tight.tpr, tight.fpr)
    prod = settings.cost.production_attack_rate
    low_prior = min(cfg.priors)
    need = required_fpr(tight.tpr, low_prior, cfg.precision_target)
    ratio = tight.fpr / need if need > 0 else float("inf")

    if prod < be:
        verdict = (
            f"the assumed production rate ({prod:.1%}) sits **below** the break-even "
            f"prevalence ({be:.2%}), so at this operating point most of the queue is "
            "false alerts before an analyst reads a single one"
        )
    else:
        verdict = (
            f"the assumed production rate ({prod:.1%}) sits above the break-even "
            f"prevalence ({be:.2%}), so the queue is majority-true at this operating "
            "point — but the margin erodes fast as the prevalence falls"
        )

    sections = "\n\n".join(_budget_section(p) for p in profiles)

    return f"""# NetSentry — The Base-Rate Fallacy, Measured

_Synthetic stand-in. Temporal split; thresholds chosen on validation at each FPR
budget; conditional TPR/FPR measured on the honest test split; then Bayes' rule
sweeps the production attack prevalence at an assumed {flows:,} flows/day. The
conditional rates are prevalence-invariant — only the queue composition changes._

## Why this report exists

Axelsson (1999) showed that intrusion detection is dominated not by the ROC curve
but by the **base rate**: when attacks are one flow in ten thousand, even a
tiny false-positive rate buries the true alerts. The evaluation report's
precision is computed on the test mix (~22% attack on the stand-in) — honest for
that mix, but not what a production queue looks like. This report is the bridge:
the same measured operating points, re-read at deployment prevalences.

## Alert-queue composition vs prevalence

{sections}

![Precision vs prevalence](../figures/{fig.name})

## The two inversions that make it concrete

- **Break-even prevalence.** At the {tight.target_fpr * 100:g}% budget the queue is
  majority-false below a prevalence of **{be:.2%}** (pi* = FPR/(TPR+FPR)). Reading
  the table: {verdict}.
- **Required FPR.** For the queue to reach **{cfg.precision_target:.0%} precision**
  at a **{low_prior:.3%}** prevalence, the detector would need an FPR of
  **{need:.2e}** — the measured operating point is **~{ratio:,.0f}x** looser. No
  threshold choice closes a gap that size; it is a property of the base rate, not
  of the model.

## What follows from this (and already ships)

The way out is not a magically better per-flow FPR; it is changing what a queue
item *is* and how it is ordered — which is exactly what the rest of the suite
prices: the **alert-queue study** ranks flows by score so the top of the queue is
far more precise than the marginal alert; the **campaign report** aggregates
flows into operations so one investigation covers hundreds of hostile flows; and
the **cost report** makes the alert/miss trade explicit instead of hiding it in a
round-number budget. The base-rate fallacy is why those layers exist.
"""
