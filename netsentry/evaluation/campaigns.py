"""Campaign-level detection: the SOC reading of a flow-level detection rate.

The headline TPR@FPR counts *flows*, but nobody responds to a flow — they respond
to the first alert from an attack **campaign**. On CIC-IDS2017 each attack class
runs as one contiguous operation on one capture day, so (day, class) identifies a
campaign; a campaign is *operationally* detected when at least one of its flows
(or ``k``, to guard against a single ambiguous hit) crosses the operating
threshold, and the interesting latency is how many hostile flows slip past before
that first alert fires.

This reframing cuts both ways, and the report keeps both edges:

- A 21% flow-level detection rate can still mean every large campaign raises an
  alert within its first handful of flows — flow TPR *understates* operational
  detection for sustained attacks (floods, scans, brute force).
- It buys nothing against small campaigns: an infiltration with a handful of
  flows has almost no draws from the detector, so the classes the slices report
  shows being missed stay missed here. Campaign framing is not a fix for weak
  per-flow detection — it is the correct unit for reading what a SOC would see.
- The false-positive side does **not** improve: benign flows carry no campaign
  structure, so the alert volume the FPR budget prices is unchanged. The framing
  moves the numerator, not the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data import schema
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

REPORT_NAME = "campaigns.md"


@dataclass
class CampaignOutcome:
    """One (day, attack-class) campaign at one operating threshold."""

    label: str
    day: str
    flows: int
    alerts: int
    first_alert_flow: int | None  # 1-based position of the first alerting flow

    @property
    def flow_detection(self) -> float:
        return self.alerts / self.flows if self.flows else 0.0

    def detected(self, k: int = 1) -> bool:
        return self.alerts >= k


def campaign_outcomes(
    labels: np.ndarray,
    days: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    benign_label: str,
) -> list[CampaignOutcome]:
    """Group attack flows into (day, class) campaigns and score each at ``threshold``.

    Rows must be in stream (capture) order: ``first_alert_flow`` is the count of
    the campaign's own flows up to and including its first alert.
    """
    labels = np.asarray(labels).astype(str)
    days = np.asarray(days).astype(str)
    alerts = np.asarray(scores) >= threshold
    outcomes: list[CampaignOutcome] = []
    day_rank = {day: i for i, day in enumerate(schema.DAY_ORDER)}
    keys = sorted(
        {(d, lb) for d, lb in zip(days, labels, strict=True) if lb != benign_label},
        key=lambda key: (day_rank.get(key[0], len(day_rank)), key[1]),
    )
    for day, label in keys:
        mask = (labels == label) & (days == day)
        hits = alerts[mask]
        positions = np.where(hits)[0]
        outcomes.append(
            CampaignOutcome(
                label=label,
                day=day,
                flows=int(mask.sum()),
                alerts=int(hits.sum()),
                first_alert_flow=int(positions[0]) + 1 if positions.size else None,
            )
        )
    return outcomes


def run_campaigns_report(settings: Settings) -> Path:
    """Score the temporal test stream and report detection per attack campaign."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    # Raw scores, matching the evaluation report's operating points (calibration
    # is monotone and would only add ties at a strict threshold).
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    y_val = result.y_val.astype(int)

    test = load_split(settings, "temporal", "test")
    labels = test[MULTICLASS_TARGET].to_numpy()
    days = (
        test[schema.DAY_COLUMN].to_numpy()
        if schema.DAY_COLUMN in test.columns
        else np.full(len(test), "?")
    )

    benign = settings.labels.benign_label
    k = settings.campaigns.k_confirm
    per_budget: dict[float, list[CampaignOutcome]] = {}
    for budget in sorted(settings.thresholds.fpr_targets):
        threshold = threshold_at_fpr(y_val, s_val, budget)
        per_budget[budget] = campaign_outcomes(labels, days, s_test, threshold, benign)

    loose = max(per_budget)
    fig = plots.plot_barh(
        [f"{o.label} ({o.day})" for o in per_budget[loose]],
        [o.flow_detection for o in per_budget[loose]],
        xlabel=f"flow-level detection @ {loose * 100:g}% FPR",
        title="Campaigns: the flows behind each first alert",
        out_path=settings.paths.figures_dir / "campaigns.png",
    )

    report = _render(per_budget, k, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote campaigns report", extra={"path": str(out_path)})

    with track_run(settings, "campaigns") as run:
        for budget, outcomes in per_budget.items():
            run.log_metrics(
                {
                    f"campaigns_detected_k1_fpr{budget:g}": sum(o.detected(1) for o in outcomes),
                    f"campaigns_detected_k{k}_fpr{budget:g}": sum(o.detected(k) for o in outcomes),
                }
            )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _summary_line(outcomes: list[CampaignOutcome], budget: float, k: int) -> str:
    n = len(outcomes)
    flow_tpr = sum(o.alerts for o in outcomes) / sum(o.flows for o in outcomes) if outcomes else 0.0
    d1, dk = sum(o.detected(1) for o in outcomes), sum(o.detected(k) for o in outcomes)
    return f"| {budget:.1%} | {flow_tpr:.1%} | **{d1}/{n}** | {dk}/{n} |"


def _render(per_budget: dict[float, list[CampaignOutcome]], k: int, fig: Path) -> str:
    loose = max(per_budget)
    outcomes = per_budget[loose]

    detail = [
        "| campaign | day | flows | flow-level detection | alerts | first alert at flow # |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        first = str(o.first_alert_flow) if o.first_alert_flow is not None else "—"
        detail.append(
            f"| {o.label} | {o.day} | {o.flows:,} | {o.flow_detection:.1%} "
            f"| {o.alerts:,} | {first} |"
        )

    summary = [
        f"| FPR budget | flow-level TPR | campaigns alerted (k=1) | confirmed (k={k}) |",
        "|---|---|---|---|",
        *(_summary_line(per_budget[b], b, k) for b in sorted(per_budget)),
    ]

    flow_tpr = sum(o.alerts for o in outcomes) / sum(o.flows for o in outcomes) if outcomes else 0.0
    d1 = sum(o.detected(1) for o in outcomes)
    missed = [o for o in outcomes if not o.detected(1)]
    small_missed = [o for o in missed if o.flows < 100]
    alerted = [o for o in outcomes if o.first_alert_flow is not None]
    fastest = min(alerted, key=lambda o: o.first_alert_flow or 0) if alerted else None
    slowest = max(alerted, key=lambda o: o.first_alert_flow or 0) if alerted else None

    if not missed and fastest is not None and slowest is not None:
        fastest_clause = (
            f"{fastest.label} pages on its very first flow"
            if fastest.first_alert_flow == 1
            else f"{fastest.label} pages at flow {fastest.first_alert_flow}"
        )
        read = (
            f"At the {loose:.0%} budget a {flow_tpr:.0%} flow-level rate reads like a miss, "
            f"but **every one of the {len(outcomes)} campaigns raises an alert** — a sustained "
            "attack offers the detector many draws and one hit starts an investigation. The "
            f"honest differentiator moves to the *latency* column: {fastest_clause}, while "
            f"{slowest.label} runs **{slowest.first_alert_flow:,} hostile flows** (of "
            f'{slowest.flows:,}) before its first alert. "Detected" and "detected in time" '
            "are different claims, and only the first-alert column separates them."
        )
    else:
        read = (
            f"At the {loose:.0%} budget, {d1} of {len(outcomes)} campaigns raise at least one "
            f"alert (flow-level rate {flow_tpr:.0%}) — for sustained attacks the campaign view "
            "reads far better than the flow view, because a flood or scan offers the detector "
            "thousands of draws. The reframing buys nothing where it matters most, though: the "
            f"silent campaign(s) — {', '.join(o.label for o in missed)} — stay silent"
            + (
                f", and the small ones are small "
                f"({', '.join(f'{o.label}: {o.flows} flows' for o in small_missed)}): few "
                "flows means few draws, so campaign framing cannot rescue a class the "
                "per-flow detector barely scores."
                if small_missed
                else "."
            )
        )

    return f"""# NetSentry — Campaign-Level Detection (the SOC's unit of account)

_Synthetic stand-in. Temporal split; the binary model's raw test scores at
thresholds chosen on validation. A campaign is one (capture day, attack class)
operation — on CIC-IDS2017 each attack class runs exactly once — and it counts as
alerted when ≥ 1 of its flows crosses the threshold, confirmed at k={k} to guard
against a single ambiguous hit. Rows are in stream order, so "first alert at flow
#" is how many of the campaign's own flows had already run._

## Why this view exists

Nobody responds to a flow. The headline TPR@FPR prices detection per flow because
that is the honest, comparable statistic — but an analyst experiences *campaigns*:
the flood either pages someone or it doesn't. Both statistics are needed: the
flow-level number for comparing models and setting budgets, the campaign-level
number for saying what the deployment would actually have caught.

## Summary (both budgets)

{chr(10).join(summary)}

## Per campaign, at the {loose:.0%} budget

{chr(10).join(detail)}

![Campaigns](../figures/{fig.name})

## Read

{read}

## What this framing does not do

Benign traffic has no campaign structure, so the false-alert volume the FPR
budget prices is **unchanged** — this moves the numerator (which attacks get
seen), not the denominator (what the alert queue costs; see the alert-queue
study). It also assumes something ties a campaign's alerts together for the
analyst (source, service, time proximity); the k={k} column is the conservative
reading for when that correlation is imperfect.
"""
