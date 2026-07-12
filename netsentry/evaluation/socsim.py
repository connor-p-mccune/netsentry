"""SOC queue simulation — detection in the time domain, where triage order bites.

The alert-queue study prices detection against a daily analyst *budget*: at K
alerts a day the score ranking catches this fraction of attacks. That is steady-
state capacity planning, and it implicitly assumes a perfectly-worked queue —
every budgeted alert gets looked at. Real queues are not worked perfectly. Alerts
arrive over a shift, analysts are a finite number of servers with a per-alert
service time, and when a burst of benign false positives lands, the queue backs
up. Whether the genuine attack buried in that backlog is reviewed *before the
shift ends* depends on the triage discipline — and that is exactly the dimension
a fraction cannot express.

This module runs a non-preemptive **M/G/c queue with abandonment at the shift
boundary**, seeded and event-driven, and compares two disciplines at the deployed
operating point:

- **FIFO** — work the oldest ticket. Fair, and what an unranked queue gives you.
- **Score-priority** — work the highest-risk ticket first, so a high-scoring
  attack jumps a benign false-positive pileup.

The headline is **attack-SLA attainment**: the share of true-attack alerts an
analyst *starts working within the SLA window*. It decomposes the alert-queue
study's "detected" into "detected **and** triaged in time" — a second miss layer
the ROC cannot see, and the operational reason SOCs rank by risk rather than by
arrival. The arrival timeline is a documented model (CIC-IDS2017 carries no
usable per-flow wall-clock); the scores and labels that flow through it are the
model's real outputs on the honest temporal test split.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SocSimConfig

logger = get_logger(__name__)

REPORT_NAME = "socsim.md"


@dataclass
class QueueResult:
    """Per-ticket outcomes of one simulated shift."""

    start: np.ndarray  # service-start time per ticket
    wait: np.ndarray  # start - arrival per ticket


def simulate_queue(
    arrivals: np.ndarray,
    services: np.ndarray,
    scores: np.ndarray,
    n_servers: int,
    discipline: str,
) -> QueueResult:
    """Non-preemptive c-server queue; return per-ticket service-start and wait times.

    Event-driven and exact: servers are identical, so a free count suffices and a
    ticket that finds a free server starts at its arrival instant (wait 0). When a
    server frees, the next ticket is chosen by ``discipline`` — ``"fifo"`` takes
    the earliest arrival, ``"priority"`` the highest score (ties broken by arrival
    then insertion order). Pure and deterministic in its inputs, so the arrival/
    service randomness lives entirely in the caller and the core is hand-checkable.
    """
    n = len(arrivals)
    order = np.argsort(arrivals, kind="stable")
    start = np.full(n, np.nan)
    wait = np.full(n, np.nan)
    events: list[tuple[float, int, int]] = []  # (time, kind: 0=arrival/1=free, ticket)
    for idx in order:
        heapq.heappush(events, (float(arrivals[idx]), 0, int(idx)))
    waiting: list[tuple[float, float, int]] = []  # discipline-keyed heap of ticket ids
    free = n_servers

    def _key(idx: int) -> tuple[float, float, int]:
        if discipline == "priority":  # highest score first (negate for a min-heap)
            return (-float(scores[idx]), float(arrivals[idx]), idx)
        return (float(arrivals[idx]), 0.0, idx)  # FIFO: earliest arrival first

    while events:
        t, kind, idx = heapq.heappop(events)
        if kind == 0:  # arrival: the ticket joins the waiting room
            heapq.heappush(waiting, _key(idx))
        else:  # a server freed
            free += 1
        while free > 0 and waiting:
            _, _, ticket = heapq.heappop(waiting)
            start[ticket] = t
            wait[ticket] = t - float(arrivals[ticket])
            free -= 1
            heapq.heappush(events, (t + float(services[ticket]), 1, ticket))
    return QueueResult(start=start, wait=wait)


@dataclass
class ShiftSummary:
    """Queue health for one (headcount, discipline) shift, aggregated over runs."""

    n_servers: int
    discipline: str
    rho: float  # offered load = total service demand / server-minutes available
    attack_sla: float  # share of attack alerts started within the SLA window
    attack_backlog: float  # share of attack alerts never started before shift end
    mean_wait: float
    p95_wait: float
    utilization: float


def _summarize_run(
    result: QueueResult,
    is_attack: np.ndarray,
    services: np.ndarray,
    *,
    horizon: float,
    sla: float,
    n_servers: int,
) -> dict[str, float]:
    """Reduce one simulated shift to its queue-health scalars."""
    started_in_shift = result.start <= horizon
    reviewed_in_sla = started_in_shift & (result.wait <= sla)
    attack = is_attack.astype(bool)
    n_attack = int(attack.sum())
    attack_sla = float(reviewed_in_sla[attack].mean()) if n_attack else float("nan")
    attack_backlog = float((~started_in_shift)[attack].mean()) if n_attack else float("nan")
    worked = started_in_shift
    served_minutes = float(services[worked].sum())
    return {
        "attack_sla": attack_sla,
        "attack_backlog": attack_backlog,
        "mean_wait": float(result.wait[worked].mean()) if worked.any() else 0.0,
        "p95_wait": float(np.percentile(result.wait[worked], 95)) if worked.any() else 0.0,
        "utilization": served_minutes / (n_servers * horizon) if horizon > 0 else 0.0,
    }


def _draw_arrivals(
    scores: np.ndarray, is_attack: np.ndarray, cfg: SocSimConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Lay the alert set onto a shift timeline: FPs uniform, attacks in campaigns.

    Benign false positives arrive as a uniform (Poisson-order) process; attack
    alerts cluster into ``n_campaigns`` bursts, so the queue faces the same
    correlated arrival shape CIC-IDS2017 attacks have — the condition under which
    triage order matters. Service times are exponential about the configured mean.
    """
    n = len(scores)
    attack = is_attack.astype(bool)
    arrivals = rng.uniform(0.0, cfg.horizon_minutes, size=n)
    n_attack = int(attack.sum())
    if n_attack:
        centers = rng.uniform(0.0, cfg.horizon_minutes, size=cfg.n_campaigns)
        assigned = centers[rng.integers(0, cfg.n_campaigns, size=n_attack)]
        jitter = rng.normal(0.0, cfg.campaign_spread_minutes, size=n_attack)
        arrivals[attack] = np.clip(assigned + jitter, 0.0, cfg.horizon_minutes)
    services = rng.exponential(cfg.minutes_per_alert_mean, size=n)
    return arrivals, services


def _sample_alerts(
    scores: np.ndarray, labels: np.ndarray, threshold: float, cfg: SocSimConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The alerts a shift sees: flows over threshold, subsampled to the shift volume."""
    alert_mask = scores >= threshold
    alert_scores = scores[alert_mask]
    alert_is_attack = labels[alert_mask]
    if len(alert_scores) > cfg.arrivals_per_shift:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(alert_scores), size=cfg.arrivals_per_shift, replace=False)
        return alert_scores[pick], alert_is_attack[pick]
    return alert_scores, alert_is_attack


def run_shift(
    scores: np.ndarray,
    is_attack: np.ndarray,
    cfg: SocSimConfig,
    n_servers: int,
    discipline: str,
    seed: int,
) -> ShiftSummary:
    """Simulate ``n_runs`` seeded shifts for one (headcount, discipline); report medians."""
    per_run: list[dict[str, float]] = []
    rho = 0.0
    for run in range(cfg.n_runs):
        rng = np.random.default_rng(seed + run)
        arrivals, services = _draw_arrivals(scores, is_attack, cfg, rng)
        rho = float(services.sum()) / (n_servers * cfg.horizon_minutes)
        result = simulate_queue(arrivals, services, scores, n_servers, discipline)
        per_run.append(
            _summarize_run(
                result,
                is_attack,
                services,
                horizon=cfg.horizon_minutes,
                sla=cfg.sla_minutes,
                n_servers=n_servers,
            )
        )

    def _median(key: str) -> float:
        return float(np.nanmedian([r[key] for r in per_run]))

    return ShiftSummary(
        n_servers=n_servers,
        discipline=discipline,
        rho=rho,
        attack_sla=_median("attack_sla"),
        attack_backlog=_median("attack_backlog"),
        mean_wait=_median("mean_wait"),
        p95_wait=_median("p95_wait"),
        utilization=_median("utilization"),
    )


def run_socsim_report(settings: Settings) -> Path:
    """Simulate the analyst queue across headcounts and disciplines; write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    y_val = result.y_val.astype(int)
    y_test = result.y_test.astype(int)
    threshold = threshold_at_fpr(y_val, s_val, variant.thresholds.primary_fpr)

    cfg = settings.socsim
    scores, is_attack = _sample_alerts(s_test, y_test, threshold, cfg, settings.seed)
    n_attack = int(is_attack.sum())
    logger.info("SOC sim alert set", extra={"alerts": len(scores), "attacks": n_attack})

    summaries: dict[str, list[ShiftSummary]] = {"fifo": [], "priority": []}
    for discipline in ("fifo", "priority"):
        for c in cfg.analyst_counts:
            summaries[discipline].append(
                run_shift(scores, is_attack, cfg, c, discipline, settings.seed)
            )
            logger.info(
                "Shift simulated",
                extra={
                    "servers": c,
                    "discipline": discipline,
                    "rho": round(summaries[discipline][-1].rho, 2),
                    "attack_sla": round(summaries[discipline][-1].attack_sla, 3),
                },
            )

    fig = _plot(settings, summaries, cfg)
    report = _render(settings, summaries, cfg, len(scores), n_attack, threshold)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote SOC-sim report", extra={"path": str(out_path)})

    biggest = _biggest_gap(summaries)
    with track_run(settings, "socsim") as run:
        run.log_metrics(
            {
                "max_sla_gap_priority_minus_fifo": biggest[1],
                "sla_gap_at_servers": float(biggest[0]),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _biggest_gap(summaries: dict[str, list[ShiftSummary]]) -> tuple[int, float]:
    """(headcount, priority-minus-FIFO attack-SLA) at the widest gap across the sweep."""
    best_c, best_gap = 0, -1.0
    for fifo, prio in zip(summaries["fifo"], summaries["priority"], strict=True):
        gap = prio.attack_sla - fifo.attack_sla
        if gap > best_gap:
            best_c, best_gap = fifo.n_servers, gap
    return best_c, best_gap


def _plot(settings: Settings, summaries: dict[str, list[ShiftSummary]], cfg: SocSimConfig) -> Path:
    counts = np.array(cfg.analyst_counts, dtype=float)
    return plots.plot_lines(
        {
            "FIFO (arrival order)": (
                counts,
                np.array([s.attack_sla for s in summaries["fifo"]]),
            ),
            "score-priority (risk order)": (
                counts,
                np.array([s.attack_sla for s in summaries["priority"]]),
            ),
        },
        xlabel="Analysts on shift (servers)",
        ylabel=f"Attack-SLA attainment (started within {cfg.sla_minutes:g} min)",
        title="Triage order decides which attacks are reviewed in time",
        out_path=settings.paths.figures_dir / "socsim.png",
    )


def _table(summaries: list[ShiftSummary], cfg: SocSimConfig) -> str:
    rows = [
        "| analysts | offered load (rho) | attack-SLA | attack backlog | mean wait | p95 wait "
        "| utilization |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rows.append(
            f"| {s.n_servers} | {s.rho:.2f} | {s.attack_sla * 100:.1f}% "
            f"| {s.attack_backlog * 100:.1f}% | {s.mean_wait:.1f} min | {s.p95_wait:.1f} min "
            f"| {s.utilization * 100:.0f}% |"
        )
    return "\n".join(rows)


def _read(summaries: dict[str, list[ShiftSummary]], cfg: SocSimConfig) -> str:
    """Sign-aware reading of the FIFO-vs-priority sweep."""
    best_c, best_gap = _biggest_gap(summaries)
    fifo_by_c = {s.n_servers: s for s in summaries["fifo"]}
    prio_by_c = {s.n_servers: s for s in summaries["priority"]}
    fifo, prio = fifo_by_c[best_c], prio_by_c[best_c]
    saturated = [s.n_servers for s in summaries["fifo"] if s.rho >= 1.0]
    slack = [s.n_servers for s in summaries["fifo"] if s.rho < 0.8]

    if best_gap >= 0.05:
        headline = (
            f"**Triage order is worth up to {best_gap * 100:.0f} points of attack-SLA**, and "
            f"the sweep shows exactly where: at {best_c} analysts (offered load "
            f"rho={fifo.rho:.2f}) score-priority reviews {prio.attack_sla * 100:.0f}% of attack "
            f"alerts within the {cfg.sla_minutes:g}-minute window against FIFO's "
            f"{fifo.attack_sla * 100:.0f}%. The queue is the same length under both "
            f"disciplines — what changes is *which* tickets are at the front when the shift "
            f"ends, and a risk-ranked queue puts the real attacks there."
        )
    else:
        headline = (
            f"**On this stand-in the two disciplines stay within {best_gap * 100:.0f} points** "
            f"across the sweep (widest at {best_c} analysts). The alert volume relative to the "
            f"SLA window never forces a deep enough backlog for ordering to bite here; the "
            f"machinery is built to surface the gap the moment arrival bursts or a tighter SLA "
            f"do force it — raise `arrivals_per_shift` or drop `sla_minutes` and the FIFO line "
            f"separates."
        )

    if saturated:
        load_read = (
            f"The load column is the tell. Below rho of 1 ({', '.join(str(c) for c in slack)} "
            f"analysts) both disciplines clear the queue and SLA is easy; at and above it "
            f"({', '.join(str(c) for c in saturated)} analysts) the server-minutes cannot "
            f"absorb the demand, backlog accrues, and the discipline is all that separates a "
            f"reviewed attack from one still in the queue at clock-out. This is the knee the "
            f"alert-queue study's fraction cannot show, because a fraction assumes the queue "
            f"was worked."
        )
    else:
        load_read = (
            "Every staffing level here runs below a load of 1, so the servers keep up and the "
            "backlog stays shallow — the regime where triage order is a fairness nicety, not a "
            "detection lever. The lever appears once the load crosses 1; the analyst_counts sweep "
            "is set to bracket that crossing on real data."
        )
    return f"{headline}\n\n{load_read}"


def _render(
    settings: Settings,
    summaries: dict[str, list[ShiftSummary]],
    cfg: SocSimConfig,
    n_alerts: int,
    n_attack: int,
    threshold: float,
) -> str:
    return f"""# NetSentry — SOC Queue Simulation (detection in the time domain)

_Synthetic stand-in. The deployed model's raw scores on the honest temporal test
split, thresholded at the primary {settings.thresholds.primary_fpr * 100:g}% FPR
budget (validation-chosen); the {n_alerts:,} resulting alerts ({n_attack:,} true
attacks) are laid onto a {cfg.horizon_minutes:g}-minute shift and worked by a
non-preemptive c-server queue. Each cell is the median of {cfg.n_runs} seeded
arrival draws. The arrival timeline is a model (benign FPs uniform, attacks in
{cfg.n_campaigns} campaigns); the scores and labels are the model's real outputs._

## Why this report exists

The alert-queue study prices detection against an analyst *budget* — at K alerts
a day the ranking catches this fraction of attacks. That is capacity planning,
and it assumes every budgeted alert is worked. A real queue has **time**: alerts
arrive over the shift, analysts are finite servers, and a burst of benign false
positives can bury a genuine attack past the point anyone reviews it. Whether the
attack is seen in time depends on the triage discipline — the dimension a fraction
cannot express.

The queue is a non-preemptive M/G/c model with abandonment at the shift boundary.
The **attack-SLA attainment** — the share of true-attack alerts an analyst starts
working within the {cfg.sla_minutes:g}-minute SLA — decomposes the alert-queue
study's "detected" into "detected **and** triaged in time."

## FIFO — work the oldest ticket

{_table(summaries["fifo"], cfg)}

## Score-priority — work the highest-risk ticket

{_table(summaries["priority"], cfg)}

![Attack-SLA vs analysts](../figures/{"socsim.png"})

## Read

{_read(summaries, cfg)}

## Method & limits

- **Queue:** non-preemptive, {len(cfg.analyst_counts)} identical-server settings;
  service times exponential about {cfg.minutes_per_alert_mean:g} min; a ticket in
  service is never bumped (priority reorders only the *waiting* room).
- **Timeline is a model.** CIC-IDS2017 has no usable per-flow wall-clock, so
  arrivals are synthesized: benign false positives uniform over the shift, attack
  alerts clustered into campaigns (the correlated shape that makes ordering
  matter). The scores and labels flowing through it are the model's real outputs
  on the honest split — the queue dynamics are simulated, the detector is not.
- **Abandonment is the shift boundary**, not analyst give-up behaviour; a real
  SOC also has escalation, shift handover, and alert de-duplication this omits.
- Score-priority assumes the score *ranks risk*, which the calibration and
  importance-stability reports argue it does at the head of the distribution —
  the same premise the alert-queue lift rests on.
"""
