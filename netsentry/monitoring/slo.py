"""Detection SLOs and multiwindow burn-rate alerting: when should this thing wake someone up?

The alert rules this repo shipped first are the ones everybody writes before they have thought
about it: *page if the attack-flag rate exceeds 50% for ten minutes*. Static thresholds like
that have two failure modes and they are both bad. Set them tight and the on-call is paged for
noise that would have resolved itself; set them loose and a slow, sustained regression burns the
whole month's tolerance for false alarms without ever tripping the wire.

Error budgets fix this by changing what is being measured. Pick a **service level objective** —
"no more than 1% of scored flows raise an alert" — and the complement is a **budget**: over a
30-day window, a fixed number of false alarms is *acceptable*, and the only question worth
paging about is how fast that budget is being spent. The **burn rate** is exactly that: the
observed bad-event rate divided by the rate that would exhaust the budget precisely at the end
of the window. Burn rate 1 is on plan. Burn rate 14.4 exhausts a 30-day budget in two days.

The alerting policy is then **multiwindow, multi-burn-rate** (Google SRE Workbook, ch. 5): a
fast, high-burn pair for outages, a slow, low-burn pair for erosion, each requiring a *short*
confirmation window as well as the long measurement window so the alert resets quickly once the
problem stops. Every row's behaviour is predictable in closed form — time to detect, budget
consumed before the page fires — and this module both computes those and then checks them by
**replaying the temporal test split as a stream** through the same rolling-window logic that
Prometheus would apply, so the theory and the measurement appear side by side.

Three SLIs are defined, and the distinction between them is the part that matters for a
detector specifically:

- **alert ratio** — alerts over scored flows. This is the only one computable *live*, because
  it needs no labels. It is what the generated Prometheus rules actually evaluate.
- **false-alarm rate** — alerts over confirmed-benign flows. The one an operator actually cares
  about, and it can only be computed retrospectively, once labels exist.
- **request success ratio** — ordinary serving availability, included because a detector that
  is not answering is a detector with a 100% miss rate, and that belongs in the same budget.

The output is a committed `docker/prometheus/slo_rules.yml` that a Prometheus already scraping
this service can load unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SLOConfig

logger = get_logger(__name__)

REPORT_NAME = "slo.md"
FIGURE_NAME = "slo_burn.png"
RULES_NAME = "slo_rules.yml"

HOURS_PER_DAY = 24.0


# --------------------------------------------------------------------------------------
# Error-budget arithmetic. All of it is closed form, so all of it is testable.
# --------------------------------------------------------------------------------------


def error_budget(objective: float) -> float:
    """The share of events an objective permits to be bad: `1 - objective`."""
    if not 0.0 < objective < 1.0:
        raise ValueError("objective must lie strictly between 0 and 1")
    return 1.0 - objective


def burn_rate(observed_bad_rate: float, budget: float) -> float:
    """How many times faster than sustainable the budget is being spent.

    Burn rate 1 spends the whole budget exactly at the end of the compliance period; burn rate
    `b` spends it in `period / b`. This single number is what makes one alert threshold
    meaningful across objectives of wildly different tightness.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    return float(observed_bad_rate / budget)


def budget_burn_time(budget: float, observed_bad_rate: float, period_hours: float) -> float:
    """Hours until a sustained bad rate exhausts the whole budget (inf if it never does)."""
    if observed_bad_rate <= 0:
        return float("inf")
    return float(period_hours * budget / observed_bad_rate)


def time_to_detect(
    window_hours: float, burn_threshold: float, budget: float, actual_bad_rate: float
) -> float:
    """Hours before a long window's average crosses the burn threshold, from a clean start.

    The window is a moving average, so a rate that jumps from zero to `actual_bad_rate` takes
    time to pull the average up to `burn_threshold * budget`. That fraction of the window is the
    detection delay, and it is why a 3-day window is a bad place to look for an outage.
    """
    if actual_bad_rate <= 0:
        return float("inf")
    fraction = (burn_threshold * budget) / actual_bad_rate
    if fraction > 1.0:
        return float("inf")  # the window can never average that high
    return float(window_hours * fraction)


def budget_consumed_at_detection(
    window_hours: float,
    burn_threshold: float,
    budget: float,
    actual_bad_rate: float,
    period_hours: float,
) -> float:
    """Share of the whole period's budget already spent by the time the alert fires."""
    detect = time_to_detect(window_hours, burn_threshold, budget, actual_bad_rate)
    if not np.isfinite(detect):
        return float("nan")
    return float(detect * actual_bad_rate / (period_hours * budget))


@dataclass
class AlertPolicy:
    """One row of a multiwindow burn-rate policy."""

    severity: str
    long_window_hours: float
    short_window_hours: float
    burn_threshold: float

    @property
    def name(self) -> str:
        return f"{_fmt_hours(self.long_window_hours)}/{_fmt_hours(self.short_window_hours)}"

    def fires(self, long_rate: float, short_rate: float, budget: float) -> bool:
        """Both windows must exceed the threshold — the short one is the reset condition.

        Requiring the short window as well is what keeps a resolved incident from holding the
        page open for the length of the long window; it is the difference between an alert that
        is actionable and one the on-call learns to silence.
        """
        threshold = self.burn_threshold * budget
        return long_rate >= threshold and short_rate >= threshold


def default_policies() -> list[AlertPolicy]:
    """The SRE Workbook's recommended four-row policy for a 30-day compliance period."""
    return [
        AlertPolicy("page", 1.0, 5.0 / 60.0, 14.4),
        AlertPolicy("page", 6.0, 0.5, 6.0),
        AlertPolicy("ticket", 24.0, 2.0, 3.0),
        AlertPolicy("ticket", 72.0, 6.0, 1.0),
    ]


def rolling_rate(events: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of a 0/1 event stream over a window of `window` events.

    Prometheus evaluates `rate()` over a time window; with a roughly constant flow rate that is
    a fixed number of events, which is what makes replaying a labelled split a faithful stand-in
    for what the alerting stack would have seen.
    """
    if window < 1:
        raise ValueError("window must be at least 1 event")
    x = np.asarray(events, dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    rate: np.ndarray = (cumulative[idx] - cumulative[lo]) / (idx - lo)
    return rate


def first_firing(
    events: np.ndarray, policy: AlertPolicy, budget: float, flows_per_hour: float
) -> int | None:
    """Index of the first event at which the policy fires, or None if it never does.

    Evaluation only starts once the long window is **full**. A partially filled moving average
    is dominated by its first few samples — one alert in the first two events reads as a 50%
    rate — so scoring during warm-up manufactures pages that a real deployment, which has been
    running for longer than its own window, would never see.
    """
    long_window = max(1, round(policy.long_window_hours * flows_per_hour))
    short_window = max(1, round(policy.short_window_hours * flows_per_hour))
    if long_window > len(events):
        return None  # the replayed stream is shorter than this row's measurement window
    long_rate = rolling_rate(events, long_window)
    short_rate = rolling_rate(events, short_window)
    threshold = policy.burn_threshold * budget
    ready = np.zeros(len(events), dtype=bool)
    ready[long_window - 1 :] = True
    hit = np.flatnonzero((long_rate >= threshold) & (short_rate >= threshold) & ready)
    return int(hit[0]) if len(hit) else None


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class PolicyRow:
    """A policy's predicted behaviour and its measured behaviour on the replayed stream."""

    policy: AlertPolicy
    predicted_detect_hours: float
    predicted_budget_consumed: float
    measured_detect_hours: float | None
    measurable: bool
    false_pages: int
    sweep_detect_hours: dict[float, float]


@dataclass
class SLIDefinition:
    """One service level indicator, its objective, and what its budget buys."""

    name: str
    definition: str
    computable_live: bool
    objective: float
    observed: float

    @property
    def budget(self) -> float:
        return error_budget(self.objective)

    @property
    def observed_burn(self) -> float:
        return burn_rate(self.observed, self.budget)


@dataclass
class SLOStudy:
    """SLI definitions, policy behaviour, and the replay that checks the arithmetic."""

    slis: list[SLIDefinition]
    rows: list[PolicyRow]
    period_hours: float
    flows_per_hour: float
    n_events: int
    replay_hours: float
    regression_multiplier: float
    regression_sweep: list[float]
    stable_alert_rate: float
    regressed_alert_rate: float
    specified_budget: float
    calibrated_budget: float
    headroom: float
    rules_path: Path

    @property
    def specification_holds(self) -> bool:
        """Does the healthy system already fit inside the objective it was handed?"""
        return self.stable_alert_rate <= self.specified_budget


def _alert_stream(settings: Settings) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay the temporal test split in capture order; return (alerts, benign_mask, fpr).

    The deployed threshold is the one this project ships: the primary-FPR operating point
    chosen on validation. The alert stream is therefore exactly the sequence of decisions the
    served model would have produced on the later capture days.
    """
    from netsentry.data.split import load_split

    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    model = SupervisedClassifier(variant).fit(
        x_train,
        train[BINARY_TARGET].to_numpy().astype(int),
        eval_set=(x_val, val[BINARY_TARGET].to_numpy().astype(int)),
    )
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), model.classes_, benign)
    threshold = threshold_at_fpr(
        val[BINARY_TARGET].to_numpy().astype(int), s_val, variant.thresholds.primary_fpr
    )

    order = np.argsort(test[DAY_COLUMN].to_numpy(), kind="stable") if DAY_COLUMN in test else None
    alerts = (s_test >= threshold).astype(float)
    is_benign = (test[BINARY_TARGET].to_numpy().astype(int) == 0).astype(bool)
    if order is not None:
        alerts, is_benign = alerts[order], is_benign[order]
    realized_fpr = float(np.mean(alerts[is_benign])) if is_benign.any() else 0.0
    return alerts, is_benign, realized_fpr


def run_slo(settings: Settings) -> SLOStudy:
    """Define the SLIs, price every burn-rate policy, and check the arithmetic by replay."""
    cfg: SLOConfig = settings.slo
    alerts, _is_benign, realized_fpr = _alert_stream(settings)
    flows_per_hour = settings.thresholds.assumed_flows_per_day / HOURS_PER_DAY
    period_hours = cfg.period_days * HOURS_PER_DAY

    alert_ratio = float(np.mean(alerts))
    # An objective the healthy system already violates makes every burn-rate alert meaningless:
    # the budget is gone before any incident starts, so the slow rows fire permanently and the
    # fast ones can never be reached. Check the specification against reality, and calibrate
    # from the measured baseline when it fails.
    specified_budget = cfg.alert_ratio_objective_budget
    calibrated_budget = (
        specified_budget if alert_ratio <= specified_budget else float(alert_ratio * cfg.headroom)
    )
    slis = [
        SLIDefinition(
            name="alert ratio",
            definition="alerts / scored flows",
            computable_live=True,
            objective=1.0 - calibrated_budget,
            observed=alert_ratio,
        ),
        SLIDefinition(
            name="false-alarm rate",
            definition="alerts / confirmed-benign flows",
            computable_live=False,
            objective=1.0 - cfg.false_alarm_objective_budget,
            observed=realized_fpr,
        ),
        SLIDefinition(
            name="request success ratio",
            definition="non-error responses / requests",
            computable_live=True,
            objective=cfg.availability_objective,
            observed=cfg.assumed_error_ratio,
        ),
    ]

    # The policy analysis is run against the alert-ratio SLO, the one the rules evaluate live.
    budget = slis[0].budget
    # The replay maps the captured flows onto the capture's own duration rather than onto the
    # assumed production rate, since this stream *is* the capture; that fixes how many events a
    # one-hour Prometheus window contains. The regression is injected as a **step** halfway
    # through -- the shape of a bad deploy or a decayed model -- so the measured detection delay
    # is comparable with the closed form, which also assumes a step from a clean window.
    replay_flows_per_hour = len(alerts) / cfg.replay_hours
    step = len(alerts) // 2
    rng = np.random.default_rng(settings.seed)
    regressed = alerts.copy()
    quiet = np.flatnonzero(alerts[step:] == 0) + step
    tail = len(alerts) - step
    n_extra = min(len(quiet), round(tail * alert_ratio * (cfg.regression_multiplier - 1.0)))
    if n_extra > 0:
        regressed[rng.choice(quiet, size=n_extra, replace=False)] = 1.0
    regressed_rate = float(np.mean(regressed[step:]))

    rows = []
    for policy in default_policies():
        fired = first_firing(regressed, policy, budget, replay_flows_per_hour)
        stable_fires = first_firing(alerts, policy, budget, replay_flows_per_hour)
        # Only a firing *after* the step is a detection; one before it would be a false page.
        detected = None if fired is None or fired < step else (fired - step + 1)
        rows.append(
            PolicyRow(
                policy=policy,
                predicted_detect_hours=time_to_detect(
                    policy.long_window_hours, policy.burn_threshold, budget, regressed_rate
                ),
                predicted_budget_consumed=budget_consumed_at_detection(
                    policy.long_window_hours,
                    policy.burn_threshold,
                    budget,
                    regressed_rate,
                    period_hours,
                ),
                measured_detect_hours=(
                    None if detected is None else detected / replay_flows_per_hour
                ),
                measurable=policy.long_window_hours <= cfg.replay_hours / 2,
                false_pages=0 if stable_fires is None else 1,
                sweep_detect_hours={
                    m: time_to_detect(
                        policy.long_window_hours,
                        policy.burn_threshold,
                        budget,
                        min(1.0, alert_ratio * m),
                    )
                    for m in cfg.regression_sweep
                },
            )
        )

    rules_path = write_prometheus_rules(settings, slis[0], default_policies())
    return SLOStudy(
        slis=slis,
        rows=rows,
        period_hours=period_hours,
        flows_per_hour=flows_per_hour,
        n_events=len(alerts),
        replay_hours=cfg.replay_hours,
        regression_multiplier=cfg.regression_multiplier,
        regression_sweep=list(cfg.regression_sweep),
        stable_alert_rate=alert_ratio,
        regressed_alert_rate=regressed_rate,
        specified_budget=specified_budget,
        calibrated_budget=calibrated_budget,
        headroom=cfg.headroom,
        rules_path=rules_path,
    )


# --------------------------------------------------------------------------------------
# The deployable artefact.
# --------------------------------------------------------------------------------------


def _fmt_hours(hours: float) -> str:
    """Prometheus duration literal for a window given in hours."""
    if hours < 1.0:
        return f"{round(hours * 60)}m"
    if hours % 24 == 0 and hours >= 24:
        return f"{round(hours / 24)}d"
    return f"{round(hours)}h"


def render_prometheus_rules(sli: SLIDefinition, policies: list[AlertPolicy]) -> str:
    """Render burn-rate alert rules a Prometheus scraping this service can load unchanged."""
    budget = sli.budget
    windows = sorted(
        {p.long_window_hours for p in policies} | {p.short_window_hours for p in policies}
    )
    lines = [
        "# NetSentry SLO burn-rate rules (generated by `netsentry slo`; do not hand-edit).",
        "#",
        f"# SLO: {sli.definition} <= {budget:.3%} over a rolling compliance period.",
        "# Each alert requires BOTH a long measurement window and a short confirmation window",
        "# to exceed the burn threshold, so a resolved incident stops paging promptly.",
        "groups:",
        "  - name: netsentry-slo-recording",
        "    rules:",
    ]
    for hours in windows:
        window = _fmt_hours(hours)
        lines += [
            f"      - record: netsentry:alert_ratio:rate{window}",
            "        expr: >",
            f'          sum(rate(netsentry_predictions_total{{decision="attack"}}[{window}]))',
            f"          / clamp_min(sum(rate(netsentry_predictions_total[{window}])), 1e-9)",
        ]
    lines += ["  - name: netsentry-slo-burn", "    rules:"]
    for policy in policies:
        long_w = _fmt_hours(policy.long_window_hours)
        short_w = _fmt_hours(policy.short_window_hours)
        threshold = policy.burn_threshold * budget
        exhausts = f"{policy.long_window_hours / policy.burn_threshold:.1f}"
        lines += [
            f"      - alert: AlertBudgetBurn{policy.burn_threshold:g}x".replace(".", "_"),
            "        expr: >",
            f"          netsentry:alert_ratio:rate{long_w} > {threshold:.6g}",
            f"          and netsentry:alert_ratio:rate{short_w} > {threshold:.6g}",
            "        labels:",
            f"          severity: {policy.severity}",
            f"          long_window: {long_w}",
            f"          short_window: {short_w}",
            "        annotations:",
            f'          summary: "Alert budget burning {policy.burn_threshold:g}x too fast '
            f'({long_w}/{short_w})"',
            "          description: >",
            f"            The alert ratio over {long_w} (confirmed over {short_w}) exceeds",
            f"            {threshold:.3%}, which is {policy.burn_threshold:g}x the sustainable",
            f"            rate for a {budget:.3%} budget. At this burn the full compliance",
            f"            period's budget is gone in {exhausts} hours of window-equivalents.",
        ]
    return "\n".join(lines) + "\n"


def write_prometheus_rules(
    settings: Settings, sli: SLIDefinition, policies: list[AlertPolicy]
) -> Path:
    """Write the generated rules next to the hand-written ones the compose stack loads."""
    out = Path(settings.slo.rules_dir) / RULES_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_prometheus_rules(sli, policies), encoding="utf-8")
    logger.info("Wrote SLO burn-rate rules", extra={"path": str(out)})
    return out


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def run_slo_report(settings: Settings) -> Path:
    """Run the SLO study, write the Prometheus rules, and write the report + figure."""
    study = run_slo(settings)

    finite = [r for r in study.rows if np.isfinite(r.predicted_detect_hours)]
    fig = plots.plot_barh(
        labels=[r.policy.name for r in finite],
        values=[r.predicted_detect_hours for r in finite],
        xlabel="hours to page at the measured regression burn rate",
        title="Multiwindow burn-rate policy: how fast does each row notice?",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xmax=max((r.predicted_detect_hours for r in finite), default=1.0) * 1.1,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote SLO report", extra={"path": str(out_path)})

    with track_run(settings, "slo") as run:
        run.log_metrics(
            {
                "alert_ratio": study.stable_alert_rate,
                "regressed_alert_ratio": study.regressed_alert_rate,
                **{
                    f"burn_{s.name.replace(' ', '_')}": s.observed_burn
                    for s in study.slis
                    if s.budget > 0
                },
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _sli_table(study: SLOStudy) -> str:
    rows = [
        "| SLI | definition | live? | objective | budget | observed | burn rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in study.slis:
        live = "yes" if s.computable_live else "retrospective"
        rows.append(
            f"| {s.name} | {s.definition} | {live} | {s.objective:.3%} | {s.budget:.3%} "
            f"| {s.observed:.3%} | {s.observed_burn:.2f}x |"
        )
    return "\n".join(rows)


def _policy_table(study: SLOStudy) -> str:
    rows = [
        "| windows (long/short) | burn | severity | predicted time to page | budget spent at page "
        "| measured on replay | false pages when healthy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        predicted = (
            "never"
            if not np.isfinite(r.predicted_detect_hours)
            else f"{r.predicted_detect_hours:.2f} h"
        )
        consumed = (
            "—"
            if not np.isfinite(r.predicted_budget_consumed)
            else f"{r.predicted_budget_consumed:.1%}"
        )
        if not r.measurable:
            measured = "_out of replay reach_"
        elif r.measured_detect_hours is None:
            measured = "never"
        else:
            measured = f"{r.measured_detect_hours:.2f} h"
        rows.append(
            f"| {r.policy.name} | {r.policy.burn_threshold:g}x | {r.policy.severity} | {predicted} "
            f"| {consumed} | {measured} | {r.false_pages} |"
        )
    return "\n".join(rows)


def _sweep_table(study: SLOStudy) -> str:
    header = "| windows (long/short) | burn | severity | " + " | ".join(
        f"{m:g}x regression" for m in study.regression_sweep
    )
    rows = [header + " |", "|---|---|---|" + "---|" * len(study.regression_sweep)]
    for r in study.rows:
        cells = []
        for m in study.regression_sweep:
            hours = r.sweep_detect_hours.get(m, float("inf"))
            cells.append("never" if not np.isfinite(hours) else f"{hours:.2f} h")
        rows.append(
            f"| {r.policy.name} | {r.policy.burn_threshold:g}x | {r.policy.severity} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(rows)


def _calibration_read(study: SLOStudy) -> str:
    if study.specification_holds:
        return (
            f"The specified objective survives contact with the system: a healthy stream alerts "
            f"on {study.stable_alert_rate:.3%} of flows, inside the {study.specified_budget:.3%} "
            "budget, so the budget has headroom to be spent by an actual incident and the "
            "policy below is evaluated against it unchanged."
        )
    return (
        f"**The specified objective fails before any incident happens, and that is the first "
        f"finding.** A budget of {study.specified_budget:.3%} was written down; the healthy "
        f"deployed model alerts on {study.stable_alert_rate:.3%} of flows at its own operating "
        f"point, a {study.stable_alert_rate / study.specified_budget:.2f}x burn *with nothing "
        "wrong*. Every downstream consequence follows mechanically: the slow rows page "
        "permanently, the fast rows can never be reached, and the on-call learns within a week "
        "that this alert means nothing. It is the most common way an SLO programme dies, and it "
        "is an arithmetic error rather than a judgement call — the objective was chosen from "
        "what someone wanted rather than from what the system does. The budget is therefore "
        f"**calibrated** from the measured baseline with {study.headroom:g}x headroom, to "
        f"{study.calibrated_budget:.3%}, and everything below — including the generated rules — "
        "derives from that. The honest reading of the calibrated number is that it is a "
        "*starting* objective: it says the system may get twice as noisy as it is today before "
        "anyone is woken up, and tightening it is a decision about analyst capacity, which the "
        "[alert-queue study](alert_queue.md) prices."
    )


def _sli_read(study: SLOStudy) -> str:
    live, retro = study.slis[0], study.slis[1]
    ratio = retro.observed / max(live.observed, 1e-12)
    return (
        f"The live SLI runs at {live.observed:.3%} against a {live.budget:.3%} budget — a "
        f"{live.observed_burn:.2f}x burn, meaning the month's tolerance for alerts is being "
        f"spent {live.observed_burn:.2f} times faster than it is replenished. The retrospective "
        f"one, which needs labels, sits at {retro.observed:.3%}. The gap between them is not an "
        "error: the alert ratio counts *every* alert, including the true ones, so it "
        f"overstates the false-alarm rate by a factor of {1 / max(ratio, 1e-9):.1f} at this "
        "prevalence. That is the correct direction for a live proxy to be wrong in — it is "
        "conservative — but it means the live SLO tightens automatically during a real incident, "
        "which is exactly when nobody wants to be paged about alert volume. The honest design is "
        "to keep both: page on the live one, review the retrospective one, and never claim the "
        "first is measuring the second."
    )


def _policy_read(study: SLOStudy) -> str:
    finite = [r for r in study.rows if np.isfinite(r.predicted_detect_hours)]
    if not finite:
        return (
            "No policy row can detect this regression: the injected burn rate never lifts a "
            "long-window average past its threshold, which is the honest signal that the "
            "objective is set looser than the failure being simulated."
        )
    fastest = min(finite, key=lambda r: r.predicted_detect_hours)
    slowest = max(finite, key=lambda r: r.predicted_detect_hours)
    contrast = (
        ""
        if slowest is fastest
        else (
            f"The {slowest.policy.name} row would take "
            f"{slowest.predicted_detect_hours:.1f} hours and "
            f"{slowest.predicted_budget_consumed:.1%} of the budget, which is why the slow rows "
            "are tickets rather than pages: their job is to catch erosion too gentle for the "
            "fast rows to see, not to be the primary detector. "
        )
    )
    reachable = [r for r in finite if r.measurable]
    agreeing = [
        r
        for r in reachable
        if r.measured_detect_hours is not None
        and abs(r.measured_detect_hours - r.predicted_detect_hours)
        <= 0.5 * max(r.predicted_detect_hours, 1e-9)
    ]
    false_pages = sum(r.false_pages for r in study.rows)
    return (
        f"A regression that lifts the alert ratio {study.regression_multiplier:g}x — "
        f"{study.stable_alert_rate:.3%} to {study.regressed_alert_rate:.3%} — is caught by the "
        f"{fastest.policy.name} row in {fastest.predicted_detect_hours:.2f} hours, with "
        f"{fastest.predicted_budget_consumed:.1%} of the period's budget already spent. "
        + contrast
        + "Within the replay's reach the measurement agrees with the closed form on "
        f"{len(agreeing)} of {len(reachable)} rows, and the healthy stream produces {false_pages} "
        "false pages across the whole policy — the property that a static threshold cannot "
        "deliver, because it has no notion of how much tolerance has already been spent."
    )


def _render(study: SLOStudy, fig: Path) -> str:
    return f"""# NetSentry — Detection SLOs and Burn-Rate Alerting

_Synthetic stand-in. Replay of {study.n_events:,} temporal-split test flows in capture order at
the deployed operating point, at an assumed {study.flows_per_hour:,.0f} flows/hour and a
{study.period_hours / HOURS_PER_DAY:.0f}-day compliance period. Generated rules:
`{study.rules_path.as_posix()}`._

## Why this report exists

The alerting rules this repo shipped first are static thresholds — *page if the attack-flag rate
exceeds 50% for ten minutes*. They have two failure modes and both are bad: tight enough to
catch a real regression means paging on noise, and loose enough to stay quiet means a slow,
sustained degradation can spend the entire month's tolerance for false alarms without ever
tripping the wire. Neither setting encodes the thing an operator actually cares about, which is
not the instantaneous rate but **how much of the acceptable-badness budget is left**.

An SLO makes that explicit. The objective sets a budget; the **burn rate** — observed bad-event
rate divided by the rate that would exhaust the budget exactly at the end of the period — turns
every objective, however tight, onto one comparable scale. Burn rate 1 is on plan; burn rate
14.4 empties a 30-day budget in two days.

## Does the objective survive contact with the system?

{_calibration_read(study)}

## The three indicators, and which of them can actually be measured live

{_sli_table(study)}

{_sli_read(study)}

## The burn-rate policy

Four rows, each pairing a long measurement window with a short confirmation window (Google SRE
Workbook, ch. 5). Requiring both is what makes the alert reset promptly once the problem stops,
rather than holding the page open for the length of the long window. The predicted columns are
closed form; the measured column replays the stream through the same rolling-window logic
Prometheus would apply, with the alert ratio stepped up {study.regression_multiplier:g}x halfway
through — the shape of a bad deploy — and evaluation held back until each row's long window is
full, since a half-filled moving average manufactures pages a running deployment would never
see. The replayed capture is {study.replay_hours:.0f} hours of traffic, so any row whose long
window exceeds half of that is priced by the closed form only and marked accordingly.

{_policy_table(study)}

{_policy_read(study)}

![Time to page per policy row](../figures/{fig.name})

## How each row behaves across regression sizes

The point of a multiwindow policy is that different rows own different failure shapes. Priced
across a sweep of alert-ratio lifts, that division of labour is visible rather than asserted:
the fast rows stay silent until something breaks abruptly, and the slow rows are the only thing
standing between a gentle erosion and a fully spent budget.

{_sweep_table(study)}

## The artefact

`{study.rules_path.as_posix()}` is generated, not hand-written, so the thresholds in the
alerting stack cannot drift away from the objective they were derived from. It contains one
recording rule per distinct window (so the burn expressions stay readable and the windows are
computed once) and one alert per policy row, evaluated against `netsentry_predictions_total` —
a metric the service already exports. A Prometheus already scraping this service loads it
unchanged.

## Scope

The compliance period is treated as a rolling window rather than a calendar month, which is the
simpler and slightly stricter reading. The replay converts Prometheus's *time* windows into
*event* windows through a fixed assumed flow rate; a real deployment's traffic is diurnal, so
the true window sizes breathe and the fast rows will be relatively slower overnight — an
argument for expressing the SLO on a ratio, as this one does, rather than on a count. The
false-alarm SLI is retrospective by construction, which is the same labelling constraint the
[metamorphic study](metamorphic.md) works around and the
[base-rate report](base_rate.md) prices: at production prevalence the alert ratio and the
false-alarm rate converge, and the gap shown here is a property of this split's unusually high
attack share. Budget policy — what to *do* when the budget is gone — is deliberately out of
scope; the [retrain-trigger study](retrain_policy.md) covers the model-side response and the
[promotion decision](promotion.md) covers the deploy-side one."""
