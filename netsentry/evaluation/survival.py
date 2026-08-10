"""Time to detection, with the attacks we never caught still in the denominator.

The [campaign study](campaigns.md) asks how many hostile flows slip past before an attack
raises its first alert. It reports that latency over the campaigns that were *detected*,
because those are the only ones that have a latency — and that is exactly the shape of
survivorship bias. The campaigns that never alerted are not missing data to be dropped; they
are the worst outcomes in the sample, and averaging without them describes a detector that
does not exist.

Epidemiology has spent a century on this problem under the name **right-censoring**: a
subject whose event has not happened by the end of follow-up still tells you something — it
tells you the event took *longer* than the observation window. The Kaplan-Meier estimator
(1958) uses that information without inventing an event time for it. At each moment where a
detection actually occurs, it computes the conditional probability of surviving that moment
given survival up to it, and multiplies:

    S(t) = product over event times t_i <= t of ( 1 - d_i / n_i )

where `d_i` detections happen among the `n_i` still-undetected campaigns at risk. A censored
campaign contributes to every `n_i` up to the point it leaves, then quietly stops counting.
Nothing is imputed and nothing is thrown away.

Three things follow, and this report measures all three. The **median time to detection** is
read off the curve rather than off the detected subset, and it may honestly not exist — if
the curve never falls to one half, the correct answer is "more than half of attacks are never
detected", which is a far more useful sentence than a mean over the lucky ones. The
**restricted mean survival time** gives a summary that still works in that case: the area
under the curve up to a fixed horizon, in flows. And the **log-rank test** compares two
operating points properly, using every campaign including the censored ones, so "the looser
budget catches attacks sooner" becomes a claim with a p-value instead of an impression.

The censoring here is administrative — every episode is followed for a fixed number of its
own flows and then stops — which is the well-behaved case the estimator assumes: whether a
campaign is censored is independent of how detectable it was. That is worth stating because
it is the assumption that makes the whole method valid, and it is one this design gets by
construction rather than by hope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SurvivalConfig

logger = get_logger(__name__)

REPORT_NAME = "survival.md"
CURVE_FIGURE = "survival_curves.png"
CLASS_FIGURE = "survival_by_class.png"


# --------------------------------------------------------------------------------------
# The estimator (pure; unit-tested against hand-computed examples)
# --------------------------------------------------------------------------------------
@dataclass
class SurvivalCurve:
    """A Kaplan-Meier curve: survival past each observed detection time, with a CI."""

    times: np.ndarray
    survival: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_subjects: int
    n_events: int

    @property
    def n_censored(self) -> int:
        """Campaigns whose follow-up ended before any alert — the survivorship-bias set."""
        return self.n_subjects - self.n_events


def kaplan_meier(times: np.ndarray, events: np.ndarray, *, z: float = 1.96) -> SurvivalCurve:
    """Kaplan-Meier survival with Greenwood standard errors on a log-log scale.

    ``times`` is when each campaign left observation and ``events`` marks whether it left by
    being detected (1) or by running out of flows (0). Censored campaigns count in the
    at-risk set right up to the moment they leave, which is the whole point: a campaign that
    went 40 flows undetected is evidence about the first 40 flows even though it never
    supplies an event.

    The interval is built on ``log(-log S)`` rather than on ``S`` directly, so it cannot
    escape ``[0, 1]`` — the naive symmetric interval routinely does at the tails, where
    survival analysis is usually most interesting.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events).astype(int)
    n = len(t)
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    unique = np.unique(t[e == 1])

    survival = 1.0
    cumulative_var = 0.0
    rows: list[tuple[float, float, int, int, float]] = []
    for time in unique:
        at_risk = int(np.sum(t >= time))
        d = int(np.sum((t == time) & (e == 1)))
        if at_risk <= 0:
            continue
        survival *= 1.0 - d / at_risk
        if at_risk > d:  # Greenwood's term is undefined when everyone is detected at once
            cumulative_var += d / (at_risk * (at_risk - d))
        rows.append((float(time), survival, at_risk, d, cumulative_var))

    if not rows:
        empty = np.zeros(0)
        return SurvivalCurve(empty, empty, empty, empty, empty, empty, n, 0)

    times_out = np.array([r[0] for r in rows])
    surv = np.array([r[1] for r in rows])
    at_risk_out = np.array([r[2] for r in rows], dtype=float)
    events_out = np.array([r[3] for r in rows], dtype=float)
    var = np.array([r[4] for r in rows])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_log = np.log(-np.log(np.clip(surv, 1e-12, 1 - 1e-12)))
        se = np.sqrt(var) / np.abs(np.log(np.clip(surv, 1e-12, 1 - 1e-12)))
        lower = np.exp(-np.exp(log_log + z * se))
        upper = np.exp(-np.exp(log_log - z * se))
    lower = np.where(np.isfinite(lower), lower, 0.0)
    upper = np.where(np.isfinite(upper), upper, 1.0)
    return SurvivalCurve(
        times=times_out,
        survival=surv,
        at_risk=at_risk_out,
        events=events_out,
        lower=np.minimum(lower, surv),
        upper=np.maximum(upper, surv),
        n_subjects=n,
        n_events=int(e.sum()),
    )


def median_survival(curve: SurvivalCurve) -> float:
    """First time the curve reaches one half, or infinity if it never does.

    Returning infinity is the honest answer, not a failure: a median that does not exist
    means more than half the campaigns were never detected within their follow-up, and
    substituting the mean of the detected ones would invent a number the data refuses.
    """
    below = np.flatnonzero(curve.survival <= 0.5)
    if below.size == 0:
        return math.inf
    return float(curve.times[below[0]])


def restricted_mean(curve: SurvivalCurve, horizon: float) -> float:
    """Area under the survival curve up to ``horizon`` — mean flows survived, in flows.

    The summary to use when the median does not exist. It is the average time-to-detection a
    campaign would have if every campaign were followed exactly ``horizon`` flows and no
    longer, which is both well defined and directly interpretable as "hostile flows that get
    through per campaign, on average, up to that horizon".
    """
    if curve.times.size == 0:
        return float(horizon)
    area = 0.0
    previous_t = 0.0
    previous_s = 1.0
    for time, surv in zip(curve.times, curve.survival, strict=True):
        if time >= horizon:
            break
        area += (time - previous_t) * previous_s
        previous_t, previous_s = float(time), float(surv)
    area += (horizon - previous_t) * previous_s
    return float(area)


def logrank_test(
    times_a: np.ndarray, events_a: np.ndarray, times_b: np.ndarray, events_b: np.ndarray
) -> tuple[float, float]:
    """Log-rank comparison of two survival curves; returns ``(chi-square, p-value)``.

    At every time a detection happens anywhere, compare how many detections group A actually
    contributed against how many it would have contributed if the two groups were identical,
    weighted by who was still at risk. Summing those differences and standardising gives a
    statistic that is chi-square with one degree of freedom under the null — and censored
    campaigns are counted properly throughout, which is what a naive comparison of detected
    subsets cannot do. The p-value is the exact one-degree-of-freedom tail, ``erfc(sqrt(x/2))``,
    so no SciPy is required to state it.
    """
    ta, ea = np.asarray(times_a, dtype=float), np.asarray(events_a).astype(int)
    tb, eb = np.asarray(times_b, dtype=float), np.asarray(events_b).astype(int)
    all_times = np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]]))
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for time in all_times:
        n_a = float(np.sum(ta >= time))
        n_b = float(np.sum(tb >= time))
        n = n_a + n_b
        d_a = float(np.sum((ta == time) & (ea == 1)))
        d = d_a + float(np.sum((tb == time) & (eb == 1)))
        if n <= 1 or d == 0:
            continue
        observed += d_a
        expected += d * n_a / n
        variance += d * (n_a / n) * (1 - n_a / n) * (n - d) / (n - 1)
    if variance <= 0.0:
        return 0.0, 1.0
    chi2 = (observed - expected) ** 2 / variance
    return float(chi2), float(math.erfc(math.sqrt(chi2 / 2.0)))


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class Episode:
    """One attack burst: how far it got before an alert, and whether an alert ever came."""

    attack_class: str
    day: str
    time: int
    detected: bool


@dataclass
class ArmSurvival:
    """Survival of attack episodes at one operating point."""

    budget: float
    threshold: float
    curve: SurvivalCurve
    median: float
    rmst: float
    naive_mean: float
    detected_share: float


@dataclass
class ClassSurvival:
    """Survival of one attack class at the primary operating point."""

    attack_class: str
    n_episodes: int
    detected_share: float
    median: float
    rmst: float


@dataclass
class SurvivalStudy:
    """Everything the report renders."""

    episode_flows: int
    n_episodes: int
    arms: list[ArmSurvival]
    classes: list[ClassSurvival]
    logrank_chi2: float
    logrank_p: float
    compared: tuple[float, float]


def build_episodes(
    labels: np.ndarray,
    days: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    benign_label: str,
    episode_flows: int,
) -> list[Episode]:
    """Chop each attack class's daily stream into fixed-length bursts and time the first alert.

    A burst rather than a whole campaign because a whole (day, class) operation gives a
    handful of subjects and survival analysis on a handful of subjects is theatre. Fixed
    length also makes the censoring **administrative** — follow-up ends because the window
    ended, never because of anything about the attack — which is precisely the independence
    the Kaplan-Meier estimator needs and the reason this design is preferred to, say,
    stopping when the attack stops.
    """
    labels = np.asarray(labels).astype(str)
    days = np.asarray(days).astype(str)
    scores = np.asarray(scores, dtype=float)
    out: list[Episode] = []
    for day in dict.fromkeys(days.tolist()):
        for cls in sorted(set(labels[days == day].tolist())):
            if cls == benign_label:
                continue
            idx = np.flatnonzero((days == day) & (labels == cls))
            for start in range(0, len(idx), episode_flows):
                window = idx[start : start + episode_flows]
                if len(window) < 2:
                    continue
                alerts = np.flatnonzero(scores[window] >= threshold)
                if alerts.size:
                    out.append(Episode(cls, day, int(alerts[0]) + 1, True))
                else:
                    out.append(Episode(cls, day, len(window), False))
    return out


def _arm(episodes: list[Episode], budget: float, threshold: float, horizon: float) -> ArmSurvival:
    """Summarise one operating point's episodes as a curve plus its headline numbers."""
    times = np.array([e.time for e in episodes], dtype=float)
    events = np.array([e.detected for e in episodes], dtype=int)
    curve = kaplan_meier(times, events)
    detected = times[events == 1]
    return ArmSurvival(
        budget=budget,
        threshold=threshold,
        curve=curve,
        median=median_survival(curve),
        rmst=restricted_mean(curve, horizon),
        naive_mean=float(detected.mean()) if detected.size else math.inf,
        detected_share=float(events.mean()) if len(events) else 0.0,
    )


def run_survival(settings: Settings) -> SurvivalStudy:
    """Estimate time-to-detection with the undetected campaigns kept in the denominator."""
    cfg: SurvivalConfig = settings.survival
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    from netsentry.data.split import load_split

    result = fit_supervised(variant)
    benign = variant.labels.benign_label
    s_val = attack_probability(np.asarray(result.proba_val), result.classes, benign)
    s_test = attack_probability(np.asarray(result.proba_test), result.classes, benign)
    y_val = np.asarray(result.y_val).astype(int)

    test = load_split(variant, "temporal", "test")
    labels = test[MULTICLASS_TARGET].astype(str).to_numpy()
    days = (
        test[DAY_COLUMN].astype(str).to_numpy()
        if DAY_COLUMN in test.columns
        else np.zeros(len(test), dtype=str)
    )

    horizon = float(cfg.episode_flows)
    budgets = list(dict.fromkeys([*settings.thresholds.fpr_targets]))
    arms: list[ArmSurvival] = []
    episodes_by_budget: dict[float, list[Episode]] = {}
    for budget in budgets:
        threshold = threshold_at_fpr(y_val, s_val, budget)
        episodes = build_episodes(labels, days, s_test, threshold, benign, cfg.episode_flows)
        episodes_by_budget[budget] = episodes
        arms.append(_arm(episodes, budget, threshold, horizon))
        logger.info(
            "Survival arm complete",
            extra={
                "budget": budget,
                "episodes": len(episodes),
                "detected": arms[-1].detected_share,
            },
        )

    primary = settings.thresholds.primary_fpr
    primary_episodes = episodes_by_budget.get(primary, next(iter(episodes_by_budget.values())))
    classes: list[ClassSurvival] = []
    for cls in sorted({e.attack_class for e in primary_episodes}):
        subset = [e for e in primary_episodes if e.attack_class == cls]
        if len(subset) < cfg.min_episodes:
            continue
        curve = kaplan_meier(
            np.array([e.time for e in subset], dtype=float),
            np.array([e.detected for e in subset], dtype=int),
        )
        classes.append(
            ClassSurvival(
                attack_class=cls,
                n_episodes=len(subset),
                detected_share=float(np.mean([e.detected for e in subset])),
                median=median_survival(curve),
                rmst=restricted_mean(curve, horizon),
            )
        )

    chi2, p_value = 0.0, 1.0
    compared = (budgets[0], budgets[0])
    if len(budgets) >= 2:
        a, b = budgets[0], budgets[-1]
        ea, eb = episodes_by_budget[a], episodes_by_budget[b]
        chi2, p_value = logrank_test(
            np.array([e.time for e in ea], dtype=float),
            np.array([e.detected for e in ea], dtype=int),
            np.array([e.time for e in eb], dtype=float),
            np.array([e.detected for e in eb], dtype=int),
        )
        compared = (a, b)

    return SurvivalStudy(
        episode_flows=cfg.episode_flows,
        n_episodes=len(primary_episodes),
        arms=arms,
        classes=classes,
        logrank_chi2=chi2,
        logrank_p=p_value,
        compared=compared,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _step(curve: SurvivalCurve, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Curve as a step function starting at (0, 1), for plotting."""
    if curve.times.size == 0:
        return np.array([0.0, horizon]), np.array([1.0, 1.0])
    xs = np.concatenate([[0.0], np.repeat(curve.times, 2), [horizon]])
    ys = np.concatenate([[1.0, 1.0], np.repeat(curve.survival, 2)])
    return xs[: len(ys)], ys


def run_survival_report(settings: Settings) -> Path:
    """Run the survival study and write the report + figures."""
    study = run_survival(settings)
    horizon = float(study.episode_flows)

    curve_fig = plots.plot_lines(
        {f"{a.budget:.1%} FPR budget": _step(a.curve, horizon) for a in study.arms},
        xlabel="hostile flows into the burst",
        ylabel="share of bursts still undetected",
        title="How long an attack runs before anyone notices",
        out_path=settings.paths.figures_dir / CURVE_FIGURE,
    )
    class_fig = plots.plot_barh(
        [c.attack_class for c in study.classes],
        [c.rmst for c in study.classes],
        xlabel=f"restricted mean flows undetected (horizon {study.episode_flows})",
        title="Which attacks run longest before detection",
        out_path=settings.paths.figures_dir / CLASS_FIGURE,
        xmax=horizon,
    )

    report = _render(study, curve_fig, class_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote survival report", extra={"path": str(out_path)})

    with track_run(settings, "survival") as run:
        run.log_params({"episode_flows": study.episode_flows, "n_episodes": study.n_episodes})
        metrics = {"logrank_chi2": study.logrank_chi2, "logrank_p": study.logrank_p}
        for arm in study.arms:
            key = f"{arm.budget:.4f}".replace(".", "_")
            metrics[f"rmst_{key}"] = arm.rmst
            metrics[f"detected_share_{key}"] = arm.detected_share
        run.log_metrics(metrics)
        run.log_artifact(curve_fig)
        run.log_artifact(class_fig)
        run.log_artifact(out_path)
    return out_path


def _fmt(value: float, horizon: int) -> str:
    """Render a survival time, saying so plainly when it does not exist."""
    return f"> {horizon} (never reached)" if not math.isfinite(value) else f"{value:.0f}"


def _arm_table(study: SurvivalStudy) -> str:
    rows = [
        "| FP budget | threshold | bursts detected | naive mean over detected | "
        "Kaplan-Meier median | restricted mean |",
        "|---|---|---|---|---|---|",
    ]
    for a in study.arms:
        rows.append(
            f"| {a.budget:.1%} | {a.threshold:.5f} | {a.detected_share:.1%} "
            f"| {a.naive_mean:.1f} flows | **{_fmt(a.median, study.episode_flows)}** "
            f"| {a.rmst:.1f} flows |"
        )
    return "\n".join(rows)


def _class_table(study: SurvivalStudy) -> str:
    rows = [
        "| attack class | bursts | detected | Kaplan-Meier median | restricted mean |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(study.classes, key=lambda x: -x.rmst):
        rows.append(
            f"| {c.attack_class} | {c.n_episodes} | {c.detected_share:.1%} "
            f"| {_fmt(c.median, study.episode_flows)} | {c.rmst:.1f} flows |"
        )
    return "\n".join(rows)


def _bias_read(study: SurvivalStudy) -> str:
    if not study.arms:
        return ""
    arm = study.arms[0]
    understated = arm.rmst - arm.naive_mean
    median_text = (
        f"the Kaplan-Meier median does not exist — the curve never falls to one half, because "
        f"only {arm.detected_share:.1%} of bursts are ever detected at all"
        if not math.isfinite(arm.median)
        else f"the Kaplan-Meier median is {arm.median:.0f} flows"
    )
    return (
        f"At the {arm.budget:.1%} budget the naive figure — mean first-alert position over the "
        f"bursts that alerted — is **{arm.naive_mean:.1f} flows**, and it is a number about a "
        f"detector nobody deployed. It conditions on success. Meanwhile {median_text}, and the "
        f"restricted mean over a {study.episode_flows}-flow horizon is **{arm.rmst:.1f} flows**, "
        f"{understated:+.1f} against the naive one. The gap is not noise, it is the "
        f"{1 - arm.detected_share:.0%} of bursts the naive average silently deleted — and it "
        "deleted precisely the worst ones. Reporting mean time-to-detection over detected "
        "incidents is the single most common way a detection metric flatters itself, and the "
        "correction has been standard in survival analysis since 1958."
    )


def _logrank_read(study: SurvivalStudy) -> str:
    if len(study.arms) < 2 or study.compared[0] == study.compared[1]:
        return ""
    lo, hi = study.compared
    a = next(x for x in study.arms if x.budget == lo)
    b = next(x for x in study.arms if x.budget == hi)
    verdict = (
        f"significant (chi-square {study.logrank_chi2:.1f}, p = {study.logrank_p:.2g})"
        if study.logrank_p < 0.05
        else f"**not** significant (chi-square {study.logrank_chi2:.1f}, p = {study.logrank_p:.2g})"
    )
    conclusion = (
        "so the looser budget genuinely catches attacks earlier, and the extra false positives "
        "buy detection speed rather than merely detection volume."
        if study.logrank_p < 0.05
        else (
            "so on this evidence the extra alert volume does not measurably shorten the time to "
            "catch a burst — it changes which bursts are caught more than how fast."
        )
    )
    return (
        f"Comparing the {lo:.1%} and {hi:.1%} operating points by log-rank — every burst "
        f"counted, censored ones included — the difference is {verdict}. Detection rises from "
        f"{a.detected_share:.1%} to {b.detected_share:.1%} of bursts and the restricted mean "
        f"moves from {a.rmst:.1f} to {b.rmst:.1f} flows, {conclusion}"
    )


def _class_read(study: SurvivalStudy) -> str:
    """Is time-to-detection a continuum, or is visibility a property of the class?"""
    if len(study.classes) < 2:
        return ""
    shares = np.array([c.detected_share for c in study.classes])
    bimodal = bool(np.all((shares < 0.1) | (shares > 0.9)))
    seen = [c for c in study.classes if c.detected_share > 0.9]
    unseen = [c for c in study.classes if c.detected_share < 0.1]
    if not bimodal or not seen or not unseen:
        return (
            "Restricted mean varies across classes, so the aggregate curve is a genuine mixture "
            "of latencies rather than of outcomes, and 'detect attacks sooner' is a meaningful "
            "goal to tune towards."
        )
    fast = min(seen, key=lambda c: c.rmst)
    seen_names = ", ".join(c.attack_class for c in seen)
    seen_clause = (
        f"{seen_names} is detected in essentially every burst, at a median of "
        f"{_fmt(fast.median, study.episode_flows)} flows"
        if len(seen) == 1
        else (
            f"{seen_names} are detected in essentially every burst — {fast.attack_class} at a "
            f"median of {_fmt(fast.median, study.episode_flows)} flows"
        )
    )
    return (
        "**Time to detection is not a continuum here — it is a property of the attack class.** "
        + seen_clause
        + "; "
        + ", ".join(c.attack_class for c in unseen)
        + " are detected in none of them. Nothing sits in between. That reframes the aggregate "
        f"curve above: its restricted mean of {study.arms[0].rmst:.1f} flows is a **mixture "
        "artefact**, not a typical wait — no burst anywhere in this data actually takes that "
        "long to be caught, because bursts are either caught almost immediately or never. The "
        "operational consequence is direct. There is no latency to tune: shaving flows off the "
        "time-to-alert would buy nothing, because the classes that are seen are already seen at "
        "once. The entire quantity is governed by *which classes are visible at all*, which "
        "makes this a coverage problem and hands it to the [slices](slices.md) and "
        "[novelty](novelty.md) studies rather than to the threshold."
    )


def _render(study: SurvivalStudy, curve_fig: Path, class_fig: Path) -> str:
    return f"""# NetSentry — Time to Detection, With the Misses Still Counted

_Synthetic stand-in. Honest temporal/binary split. Attack flows are chopped into
{study.episode_flows}-flow bursts within each (day, class) stream, giving {study.n_episodes}
bursts; a burst's time is the position of its first alerting flow, and a burst that never
alerts is right-censored at the end of its window._

## Why this report exists

The [campaign study](campaigns.md) reports how many hostile flows slip past before an attack
raises its first alert, over the campaigns that raised one. Those are the only campaigns with a
latency to average — and that is the exact shape of survivorship bias. The campaigns that never
alerted are not missing data; they are the worst outcomes in the sample.

Right-censoring is the standard name for this and the Kaplan-Meier estimator (1958) is the
standard fix. A burst that ran {study.episode_flows} flows without detection is not discarded:
it is evidence that detection takes longer than {study.episode_flows} flows, and it counts in
the at-risk denominator for every moment it was observed.

```
S(t) = product over detection times t_i <= t of ( 1 - d_i / n_i )
```

Nothing is imputed, nothing is dropped, and the curve is what an operator actually wants: the
probability that an attack is still running unnoticed after `t` of its flows.

## The bias, measured

{_arm_table(study)}

{_bias_read(study)}

![survival curves by operating point](../figures/{curve_fig.name})

## Does a looser budget catch attacks sooner?

{_logrank_read(study)}

## Which attacks run longest

{_class_table(study)}

{_class_read(study)}

![restricted mean by attack class](../figures/{class_fig.name})

## Scope

Censoring here is **administrative**: every burst is followed for a fixed number of its own
flows and then stops, so whether a burst is censored is independent of how detectable it was.
That independence is the assumption Kaplan-Meier needs, and this design gets it by construction
rather than by argument — a design that stopped following a burst when the attack stopped would
violate it, because attacks that end quickly are not a random sample of attacks. Time is
measured in the burst's own hostile flows rather than in seconds, which is the right unit for a
flow-level detector and the only one available without per-flow timestamps; a wall-clock version
would additionally capture the fact that some attacks emit flows far faster than others. Bursts
within a (day, class) stream are not independent — they come from the same operation and share
whatever made it easy or hard to see — so the confidence intervals are narrower than they should
be, in the usual direction for clustered data. The horizon for the restricted mean is the burst
length itself, so it is a bounded summary of a bounded window and not an estimate of the true
mean time to detection, which is undefined whenever some attacks are never detected at all."""
