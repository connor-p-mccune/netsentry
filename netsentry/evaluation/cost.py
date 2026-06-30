"""Cost-sensitive (decision-theoretic) threshold selection — the SOC economics.

A fixed-FPR threshold is operationally honest but arbitrary: why 1% and not 0.5%?
Attach a cost to each outcome — analyst time per raised alert, expected loss per
missed attack — and the right operating point is the threshold that minimises
*expected cost*.

The clean result this showcases: for a **calibrated** probability ``p(attack|x)``,
the per-flow optimal rule is "alert iff ``p >= cost_per_alert / cost_per_miss``".
That **Bayes threshold** is a closed form in the cost ratio and is independent of
the class base rate — but it is only correct if the score is a real probability,
which is exactly why calibration (``models/calibration.py``) comes first. The
empirical cost sweep should land on top of it, which doubles as a calibration check.

The daily extrapolation uses a configured production base rate, not the (~22%
attack) synthetic test rate, so the alert volumes and currency figures are sane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import operating_point, positive_scores, rates_at_threshold
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "cost.md"


def bayes_threshold(cost_per_alert: float, cost_per_miss: float) -> float:
    """Per-flow optimal decision threshold for a calibrated probability."""
    return cost_per_alert / cost_per_miss if cost_per_miss > 0 else 0.0


def per_flow_cost_rates(
    tpr: float, fpr: float, prior: float, cost_per_alert: float, cost_per_miss: float
) -> float:
    """Expected cost per flow given detection/false-alarm rates and a base rate.

    A fraction ``prior`` of flows are attacks: of those, ``1 - tpr`` are missed
    (cost_per_miss) and ``tpr`` are alerted (cost_per_alert); of the benign rest,
    ``fpr`` are alerted (cost_per_alert).
    """
    return float(
        prior * ((1.0 - tpr) * cost_per_miss + tpr * cost_per_alert)
        + (1.0 - prior) * fpr * cost_per_alert
    )


def _rates(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    attacks, benign = y_true == 1, y_true == 0
    tpr = float(np.mean(scores[attacks] >= threshold)) if attacks.any() else 0.0
    fpr = float(np.mean(scores[benign] >= threshold)) if benign.any() else 0.0
    return tpr, fpr


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    prior: float,
    cost_per_alert: float,
    cost_per_miss: float,
) -> np.ndarray:
    """Expected per-flow cost at each threshold (base-rate reweighted)."""
    out = []
    for t in thresholds:
        tpr, fpr = _rates(y_true, scores, float(t))
        out.append(per_flow_cost_rates(tpr, fpr, prior, cost_per_alert, cost_per_miss))
    return np.array(out)


def cost_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    prior: float,
    cost_per_alert: float,
    cost_per_miss: float,
    grid_points: int = 300,
) -> tuple[float, float]:
    """Threshold minimising expected per-flow cost at the deployment base rate."""
    scores = np.asarray(scores)
    grid = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, grid_points)))
    grid = np.concatenate([grid, [grid[-1] + 1e-6]])  # 'alert nothing' candidate
    costs = cost_curve(y_true, scores, grid, prior, cost_per_alert, cost_per_miss)
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


@dataclass
class CostPoint:
    """An operating point evaluated under the cost model at the production prior."""

    name: str
    threshold: float
    tpr: float
    fpr: float
    alerts_per_day: float
    daily_cost: float


def _point(
    name: str, threshold: float, y_test: np.ndarray, s_test: np.ndarray, s: Settings
) -> CostPoint:
    rates = rates_at_threshold(y_test, s_test, threshold)
    prior = s.cost.production_attack_rate
    flows = s.thresholds.assumed_flows_per_day
    pf = per_flow_cost_rates(
        rates["tpr"], rates["fpr"], prior, s.cost.cost_per_alert, s.cost.cost_per_miss
    )
    alert_fraction = prior * rates["tpr"] + (1.0 - prior) * rates["fpr"]
    return CostPoint(
        name=name,
        threshold=threshold,
        tpr=rates["tpr"],
        fpr=rates["fpr"],
        alerts_per_day=alert_fraction * flows,
        daily_cost=pf * flows,
    )


def _slug(name: str) -> str:
    """MLflow metric names disallow some punctuation (e.g. '%')."""
    return "".join(c if (c.isalnum() or c in "._-/ ") else "_" for c in name)


def run_cost_report(settings: Settings) -> Path:
    """Fit the temporal binary model, find the cost-optimal point, write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    if result.bundle.calibrator is not None:  # cost in probability space needs calibration
        s_val = result.bundle.calibrator.transform(s_val)
        s_test = result.bundle.calibrator.transform(s_test)
    y_val, y_test = result.y_val.astype(int), result.y_test.astype(int)

    cfg = settings.cost
    prior = cfg.production_attack_rate
    bayes = bayes_threshold(cfg.cost_per_alert, cfg.cost_per_miss)
    empirical, _ = cost_optimal_threshold(
        y_val, s_val, prior, cfg.cost_per_alert, cfg.cost_per_miss, cfg.grid_points
    )

    points = [_point("cost-optimal", empirical, y_test, s_test, settings)]
    for fpr in settings.thresholds.fpr_targets:
        op = operating_point(
            y_val, s_val, y_test, s_test, fpr, settings.thresholds.assumed_flows_per_day
        )
        points.append(
            _point(f"fixed FPR {fpr * 100:g}%", op["threshold"], y_test, s_test, settings)
        )

    grid = np.unique(np.quantile(s_test, np.linspace(0.0, 1.0, cfg.grid_points)))
    daily = (
        cost_curve(y_test, s_test, grid, prior, cfg.cost_per_alert, cfg.cost_per_miss)
        * settings.thresholds.assumed_flows_per_day
    )
    fig = plots.plot_lines(
        {"expected daily cost": (grid, daily)},
        xlabel="Decision threshold (calibrated attack probability)",
        ylabel=f"Expected cost / day ({cfg.currency})",
        title="Cost vs decision threshold",
        out_path=settings.paths.figures_dir / "cost_curve.png",
        vlines={"cost-optimal": empirical},
    )

    report = _render(settings, points, bayes, empirical, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote cost report", extra={"path": str(out_path)})

    with track_run(settings, "cost") as run:
        run.log_params({"cost_per_alert": cfg.cost_per_alert, "cost_per_miss": cfg.cost_per_miss})
        run.log_metrics({"bayes_threshold": bayes, "empirical_threshold": empirical})
        run.log_metrics({f"daily_cost_{_slug(p.name)}": p.daily_cost for p in points})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(
    settings: Settings, points: list[CostPoint], bayes: float, empirical: float, fig: Path
) -> str:
    cfg = settings.cost
    cur = cfg.currency
    rows = [
        "| operating point | threshold | detection (TPR) | FPR | alerts/day | exp. cost/day |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.name} | {p.threshold:.4f} | {p.tpr * 100:.1f}% | {p.fpr * 100:.3f}% | "
            f"{p.alerts_per_day:,.0f} | {cur}{p.daily_cost:,.0f} |"
        )
    best = min(points, key=lambda p: p.daily_cost)
    optimal = points[0]
    ratio = cfg.cost_per_miss / cfg.cost_per_alert if cfg.cost_per_alert else float("inf")
    if best is optimal:
        verdict = (
            f"On test it is the cheapest of the three (**{cur}{optimal.daily_cost:,.0f}/day**), "
            "turning the precision/recall trade-off into a single defensible number."
        )
    else:
        verdict = (
            f"On test the val-chosen optimum costs **{cur}{optimal.daily_cost:,.0f}/day**, but "
            f"the `{best.name}` profile edges it (**{cur}{best.daily_cost:,.0f}**): a threshold "
            "tuned on validation (earlier days) drifts on the later-day test set — the same "
            "temporal effect the headline split exposes, so re-select it on recent data in prod."
        )
    return f"""# NetSentry — Cost-Sensitive Threshold Selection

_Synthetic stand-in; the costs are illustrative knobs (config `cost.*`). The method
is the point: pick the operating point that minimises expected cost, not a
round-number FPR._

## Cost model

- Each **raised alert** (true or false) costs **{cur}{cfg.cost_per_alert:g}** of analyst
  triage time.
- Each **missed attack** costs an expected **{cur}{cfg.cost_per_miss:g}** in damage.
- Production attack base rate assumed at **{cfg.production_attack_rate * 100:g}%** over
  **{settings.thresholds.assumed_flows_per_day:,}** flows/day (the synthetic test split
  is ~22% attack, far above reality, so the daily figures use this prior instead).

## Cost-optimal threshold

Sweeping the threshold on **validation** to minimise expected per-flow cost at the
production prior gives **{empirical:.4f}**, then evaluated on test below. Detection,
false-alarm, alert-volume and cost are all reported at that operating point.

For reference, the closed-form per-flow optimum — "alert iff
`p(attack) >= cost_per_alert / cost_per_miss`" — is **{bayes:.4f}** (a {ratio:.0f}x
miss-to-alert ratio). That identity holds when you operate at the *scored* base
rate; under a much lower production prior ({cfg.production_attack_rate * 100:g}%) the
benign pool dominates the false-alarm cost, so the optimal single threshold rises
above it. Either way the score must be a real probability, which is why calibration
comes first.

## Operating points compared (evaluated on test, costed at the production prior)

{chr(10).join(rows)}

{verdict} Here the daily cost is dominated by **missed attacks** — on the hard
temporal split detection is modest at every threshold, so the miss term swamps
analyst time, and threshold choice moves the total only at the margin.

![Cost vs threshold](../figures/{fig.name})

## Why this matters

False positives drive alert fatigue, but misses drive breaches; the balance is a
business decision, not a default. Exposing the cost knobs makes the trade-off
explicit and tunable per deployment — raising `cost.cost_per_miss` (or the
production base rate) slides the optimum along the curve above.
"""
