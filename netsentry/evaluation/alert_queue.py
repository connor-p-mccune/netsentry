"""Capacity-constrained alert triage — the detection a fixed analyst budget buys.

The cost report picks the expected-cost-minimising threshold; this answers the
complementary question a SOC lead actually asks: "my team can work K alerts a day —
ranking flows by risk, how many attacks do we catch, and how much better is that
than triaging K flows at random?"

An alert budget of ``K`` per day corresponds to the decision threshold whose alert
volume equals ``K`` at the production base rate (``alerts = flows/day * (base * TPR +
(1 - base) * FPR)``). At that threshold the detection (recall = TPR) is what the team
catches, and the **lift over random triage** — recall divided by ``K / flows`` — is
what the model is worth: how many times more attacks per unit of analyst time than
picking flows blind. Everything is evaluated at a realistic production base rate, not
the ~22% synthetic test mix, so the alert-per-day and headcount numbers are sane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_curve

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "alert_queue.md"


@dataclass
class QueuePoint:
    """Detection achievable at one daily alert budget, and its lift over random."""

    budget: int
    analysts: float
    threshold: float
    recall: float
    precision: float
    alerts_per_day: float
    random_recall: float
    lift: float


def simulate_queue(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    base_rate: float,
    flows_per_day: int,
    budgets: list[int],
    minutes_per_alert: float,
    analyst_minutes_per_day: float,
) -> list[QueuePoint]:
    """Detection vs analyst capacity, from the score ranking at a production base rate.

    For each daily alert ``budget`` the best within-budget operating point on the ROC
    curve is chosen (the lowest threshold whose alert volume still fits the budget),
    and detection, precision, and lift over random triage are reported there.
    """
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, np.asarray(scores))
    # Alert volume per day at each ROC threshold, at the production base rate.
    alert_fraction = base_rate * tpr + (1.0 - base_rate) * fpr
    alerts = alert_fraction * flows_per_day

    points: list[QueuePoint] = []
    for budget in budgets:
        within = np.where(alerts <= budget)[0]
        idx = int(within[-1]) if len(within) else 0  # ROC index 0 is the 'alert nothing' point
        recall = float(tpr[idx])
        frac = float(alert_fraction[idx])
        precision = (base_rate * recall) / frac if frac > 0 else 0.0
        random_recall = min(1.0, budget / flows_per_day)
        lift = recall / random_recall if random_recall > 0 else 0.0
        points.append(
            QueuePoint(
                budget=budget,
                analysts=budget * minutes_per_alert / analyst_minutes_per_day,
                threshold=float(thresholds[idx]),
                recall=recall,
                precision=precision,
                alerts_per_day=float(alerts[idx]),
                random_recall=random_recall,
                lift=lift,
            )
        )
    return points


def run_alert_queue_report(settings: Settings) -> Path:
    """Fit the temporal binary model and write the capacity-constrained triage report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    s_test = positive_scores(result.proba_test, result.classes)
    if result.bundle.calibrator is not None:
        s_test = result.bundle.calibrator.transform(s_test)
    y_test = result.y_test.astype(int)

    cfg = settings.alert_queue
    base_rate = settings.cost.production_attack_rate
    flows = settings.thresholds.assumed_flows_per_day
    points = simulate_queue(
        y_test,
        s_test,
        base_rate=base_rate,
        flows_per_day=flows,
        budgets=cfg.alert_budgets_per_day,
        minutes_per_alert=cfg.minutes_per_alert,
        analyst_minutes_per_day=cfg.analyst_minutes_per_day,
    )

    fig = plots.plot_lines(
        {
            "NetSentry (risk-ranked)": (
                np.array([p.budget for p in points]),
                np.array([p.recall for p in points]),
            ),
            "random triage": (
                np.array([p.budget for p in points]),
                np.array([p.random_recall for p in points]),
            ),
        },
        xlabel="Alert budget (alerts triaged / day)",
        ylabel="Attacks detected (recall)",
        title="Detection vs analyst capacity",
        out_path=settings.paths.figures_dir / "alert_queue.png",
    )

    report = _render(settings, points, base_rate, flows, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote alert-queue report", extra={"path": str(out_path)})

    with track_run(settings, "alert_queue") as run:
        run.log_metrics({f"recall_at_{p.budget}_alerts": p.recall for p in points})
        run.log_metrics({f"lift_at_{p.budget}_alerts": p.lift for p in points})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(
    settings: Settings, points: list[QueuePoint], base_rate: float, flows: int, fig: Path
) -> str:
    rows = [
        "| alerts/day | analysts | detection (recall) | precision | lift vs random |",
        "|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.budget:,} | {p.analysts:.1f} | {p.recall * 100:.1f}% | "
            f"{p.precision * 100:.1f}% | {p.lift:.0f}x |"
        )
    best_lift = max(points, key=lambda p: p.lift)
    top = points[-1]
    return f"""# NetSentry - Alert-Queue Capacity Planning

_Synthetic stand-in; the method is the point. Detection a fixed analyst budget buys,
from the temporal model's risk ranking, at a realistic **{base_rate * 100:g}%**
production attack base rate over **{flows:,}** flows/day (not the ~22% synthetic test
mix, so the volumes are sane). Budgeted at **{settings.alert_queue.minutes_per_alert:g}
min/alert**, **{settings.alert_queue.analyst_minutes_per_day:g} productive min/analyst/day**._

A SOC cannot chase every flagged flow; it works a bounded queue. Ranking flows by
risk and triaging the top **K/day** is the real deployment, so the operational
question is not "what is the AUC" but "at my staffing, how many attacks do we catch,
and how much better is that than triage at random?"

{chr(10).join(rows)}

- **Lift** is recall divided by random triage's `K / flows` hit rate: at
  {best_lift.budget:,} alerts/day the ranking catches **{best_lift.lift:.0f}x** more
  attacks than working the same number of flows blind — the model's worth stated in
  the currency a SOC lead budgets in (analyst time).
- Detection climbs with the queue and then flattens: at {top.budget:,} alerts/day
  (~{top.analysts:.1f} analysts) recall is **{top.recall * 100:.1f}%**. The knee is
  where extra staffing stops paying off — the capacity-planning number the PR-AUC
  alone can't give.
- Precision is reported at the **production** base rate, so it reflects the real
  benign-heavy queue an analyst faces, not the balanced test split.

![Detection vs analyst capacity](../figures/{fig.name})

## Why this matters

PR-AUC and TPR@FPR describe the model; this describes the *deployment*. It converts a
ranking into a staffing plan — "N analysts detect M% of attacks" — and quantifies the
model as a force multiplier on scarce analyst time, which is the lens a SOC actually
buys detection with. It complements the [cost report](cost.md): cost picks the
economically optimal threshold, this reads detection off a fixed headcount.
"""
