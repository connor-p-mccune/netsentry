"""Holding alert volume at the analyst budget: the threshold as a control loop, and its attack.

A SOC has a fixed number of analysts. The detector has a fixed threshold. Those two facts stop
being compatible the moment traffic changes, and every threshold in this project so far is
**open-loop**: pick it on a validation set at a target rate, ship it, and hope the rate the
operator experiences resembles the one that was measured. The refresh study watched that hope
decay; the Neyman-Pearson study found the deployed rule violating its own false-positive budget
half the time. Neither closes the loop.

This module does. Alert volume is a measured output, the threshold is an actuator, the analyst
budget is a setpoint -- a feedback control problem, with a century of theory attached that says
more than "raise it when there are too many alerts":

- **The actuator has to be parameterised sensibly.** Controlling the raw threshold, or even its
  quantile, is badly conditioned: near the operating point a thousandth of a quantile separates
  ten alerts from a hundred, so a gain tuned in one regime is wrong in the next. The manipulated
  variable here is ``log10`` of the alert-rate parameter, in which the plant is close to a unit
  gain -- one decade of actuator for one decade of volume -- and a gain means the same thing
  everywhere.
- **The integral term is for a disturbance that keeps moving.** With the actuator and the error
  in the same units a *static* disturbance has an exact fixed point that proportional control
  alone reaches -- the unit tests pin both halves of that -- but real traffic drifts, and a
  proportional loop lags a moving target permanently while the integrator accumulates the
  persistent part of the error and catches up.
- **Loop gain decides stability**, and the sweep locates that boundary empirically rather than
  asserting a safe value.
- **Delayed feedback is feedback about a state that no longer exists.** Alert volume is always
  observed late (aggregation windows, dashboards, shift handover), and the delay sweep prices it.

And then the question a control-theory textbook does not ask, because its plants are not
adversarial: the loop's input is attacker-influenced. An attacker who can raise alert volume can
make the controller raise its own threshold, and then walk through the gap. That **control-loop
attack** is executed here against the same policies, measured against the correct counterfactual
(the same policy, the same flows, no flood), and then mitigated -- measure, fix, re-measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ControlConfig

logger = get_logger(__name__)

REPORT_NAME = "control.md"
TRACE_FIGURE_NAME = "control_trace.png"
GAIN_FIGURE_NAME = "control_gain.png"

STATIC = "static threshold"
PROPORTIONAL = "proportional (P)"
PI = "proportional-integral (PI)"
PI_GUARDED = "PI + surge guard"
TRACKER = "score-space tracker"

_MIN_LOG_RATE = -5.0  # one alert in 100,000 flows: the floor the actuator may reach
_MAX_LOG_RATE = float(np.log10(0.5))


class Policy(Protocol):
    """Anything that turns an observed alert volume into the next threshold."""

    def update(self, setpoint: float, measured: float, reference: np.ndarray) -> float:
        """Return the score threshold to apply to the next batch."""


@dataclass
class StaticThreshold:
    """The open-loop incumbent: calibrate once on validation, ship, never touch."""

    log_rate: float

    def update(self, setpoint: float, measured: float, reference: np.ndarray) -> float:
        """Ignore the measurement entirely -- that is the whole point of the baseline."""
        return float(np.quantile(reference, 1.0 - 10.0**self.log_rate))


@dataclass
class LogRatePI:
    """PI control on the log alert rate: the well-conditioned way to steer a tail threshold.

    The error is ``log10(measured / setpoint)`` -- a *ratio* error, because "twice the budget"
    means the same thing at ten alerts and at a thousand while "ninety alerts too many" does not.
    In these units the plant is nearly unit gain, so ``kp = 1`` is roughly deadbeat and the tuning
    question becomes how much correction to apply per batch rather than what scale the numbers
    happen to be on.

    ``max_step`` rate-limits the actuator (thresholds that jump are thresholds nobody trusts, and
    it is half the defence against a driven loop), ``anti_windup`` stops the integrator
    accumulating while the actuator is saturated, and ``freeze_above`` stops it learning from an
    excursion so large it is more likely an incident than a setpoint error.
    """

    kp: float
    ki: float = 0.0
    log_rate: float = -2.0
    max_step: float = 0.15
    anti_windup: bool = True
    freeze_above: float = float("inf")  # in decades of error
    integral: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(self, setpoint: float, measured: float, reference: np.ndarray) -> float:
        """One control step.

        The measurement is floored at half an alert rather than offset by it: an offset would
        bias the loop's fixed point away from the setpoint by that amount, which at a
        ten-alert budget is a permanent 5% error nobody would ever find.
        """
        error = float(np.log10(max(measured, 0.5) / max(setpoint, 1e-9)))
        surging = abs(error) > self.freeze_above
        if not surging:
            self.integral += error
        proposed = self.log_rate - self.kp * error - self.ki * self.integral
        clipped = float(np.clip(proposed, _MIN_LOG_RATE, _MAX_LOG_RATE))
        if self.anti_windup and abs(clipped - proposed) > 1e-12 and not surging:
            self.integral -= error  # saturated: do not accumulate what cannot be applied
        step = float(np.clip(clipped - self.log_rate, -self.max_step, self.max_step))
        self.log_rate = float(np.clip(self.log_rate + step, _MIN_LOG_RATE, _MAX_LOG_RATE))
        self.history.append(self.log_rate)
        return float(np.quantile(reference, 1.0 - 10.0**self.log_rate))


@dataclass
class ScoreSpaceTracker:
    """Robbins-Monro stochastic approximation of the threshold itself (Tierney 1983).

    The other way to parameterise the same problem: forget quantiles and move the threshold
    directly, ``theta <- theta + gamma * scale * (realised - target) / target``. It has one step
    size instead of two gains and converges for a stationary stream -- which is exactly the
    assumption a detector under drift does not get, so it belongs in the comparison as the
    gain-free option rather than as the recommended one.
    """

    target_rate: float
    step: float
    scale: float
    threshold: float
    history: list[float] = field(default_factory=list)

    def update(self, setpoint: float, measured: float, reference: np.ndarray) -> float:
        """Nudge the threshold toward the rate that would have hit the budget."""
        realised = measured / max(setpoint, 1e-9) * self.target_rate
        self.threshold += self.step * self.scale * (realised - self.target_rate) / self.target_rate
        self.threshold = float(np.clip(self.threshold, float(np.min(reference)), 1.0))
        self.history.append(self.threshold)
        return self.threshold


# --------------------------------------------------------------------------------------
# Control-theory diagnostics
# --------------------------------------------------------------------------------------


def overshoot(volumes: np.ndarray, setpoint: float) -> float:
    """Peak excursion above the setpoint, as a fraction of it."""
    if len(volumes) == 0 or setpoint <= 0:
        return 0.0
    return float(max(0.0, (np.max(volumes) - setpoint) / setpoint))


def settling_time(volumes: np.ndarray, setpoint: float, tolerance: float) -> int:
    """Batches until the output stays inside the tolerance band *for good*.

    "For good" matters: a loop that touches the band and leaves again has not settled, and
    reporting first entry instead of last exit is how an oscillating controller gets described
    as a fast one.
    """
    if len(volumes) == 0 or setpoint <= 0:
        return 0
    inside = np.abs(volumes - setpoint) <= tolerance * setpoint
    outside = np.where(~inside)[0]
    return int(outside[-1] + 1) if len(outside) else 0


def steady_state_error(volumes: np.ndarray, setpoint: float, tail: int = 10) -> float:
    """Mean relative error over the last few batches -- what the loop converges to, if anything."""
    if len(volumes) == 0 or setpoint <= 0:
        return 0.0
    window = volumes[-min(tail, len(volumes)) :]
    return float((np.mean(window) - setpoint) / setpoint)


def oscillation(actuator: np.ndarray) -> float:
    """Mean absolute actuator movement per batch: control effort, and the smell of instability."""
    if len(actuator) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(actuator))))


# --------------------------------------------------------------------------------------
# The plant: batches of scored traffic, a threshold, a volume out
# --------------------------------------------------------------------------------------


@dataclass
class LoopTrace:
    """One controller's run: what it did, what it achieved, and what it cost in detection."""

    name: str
    volumes: np.ndarray
    thresholds: np.ndarray
    setpoint: float
    tolerance: float
    attacks_caught: int
    attacks_total: int
    alerts: int
    true_alerts: int
    protected_caught: int = 0
    protected_total: int = 0

    @property
    def volume_mae(self) -> float:
        return float(np.mean(np.abs(self.volumes - self.setpoint)))

    @property
    def recall(self) -> float:
        return self.attacks_caught / max(self.attacks_total, 1)

    @property
    def precision(self) -> float:
        return self.true_alerts / max(self.alerts, 1)

    @property
    def protected_recall(self) -> float:
        """Detection of the flows an attacker would be trying to sneak past the loop."""
        return self.protected_caught / max(self.protected_total, 1)

    @property
    def overshoot(self) -> float:
        return overshoot(self.volumes, self.setpoint)

    @property
    def settling(self) -> int:
        return settling_time(self.volumes, self.setpoint, self.tolerance)

    @property
    def settled(self) -> bool:
        return self.settling < len(self.volumes)

    @property
    def steady_state(self) -> float:
        return steady_state_error(self.volumes, self.setpoint)

    @property
    def effort(self) -> float:
        """Actuator movement per batch, in decades of threshold -- comparable across policies."""
        with np.errstate(divide="ignore"):
            return oscillation(np.log10(np.maximum(self.thresholds, 1e-12)))


def simulate_loop(
    scores: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    policy: Policy,
    bounds: list[tuple[int, int]],
    *,
    setpoint: float,
    tolerance: float,
    initial_threshold: float,
    delay: int = 0,
    protected: np.ndarray | None = None,
    name: str = "policy",
) -> LoopTrace:
    """Run one closed loop over the stream, batch by batch.

    The threshold applied to a batch is always the one computed from *previous* batches: a loop
    that thresholded on the batch it is judging would be reading the answer, and its alert count
    would be constant by construction rather than a controlled variable.

    ``delay`` holds the measurement back that many batches, the way a real deployment sees its
    own alert volume only after aggregation.
    """
    threshold = initial_threshold
    volumes: list[float] = []
    thresholds: list[float] = []
    pending: list[float] = []
    caught = alerts = true_alerts = protected_caught = 0
    for start, stop in bounds:
        flagged = scores[start:stop] >= threshold
        volume = float(np.sum(flagged))
        volumes.append(volume)
        thresholds.append(threshold)
        hits = int(np.sum(flagged & (labels[start:stop] == 1)))
        caught += hits
        true_alerts += hits
        alerts += int(volume)
        if protected is not None:
            protected_caught += int(np.sum(flagged & protected[start:stop]))

        pending.append(volume)
        if len(pending) > delay:
            threshold = policy.update(setpoint, pending.pop(0), reference)

    return LoopTrace(
        name=name,
        volumes=np.array(volumes, dtype=float),
        thresholds=np.array(thresholds, dtype=float),
        setpoint=setpoint,
        tolerance=tolerance,
        attacks_caught=caught,
        attacks_total=int(np.sum(labels == 1)),
        alerts=alerts,
        true_alerts=true_alerts,
        protected_caught=protected_caught,
        protected_total=int(np.sum(protected)) if protected is not None else 0,
    )


def _bounds(n: int, batch_rows: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_rows, n)) for start in range(0, n, batch_rows)]


def _policies(
    settings: Settings, log_rate: float, reference: np.ndarray
) -> list[tuple[str, Policy]]:
    """The policies under test, all starting from the same calibrated operating point."""
    cfg: ControlConfig = settings.control
    scale = float(np.subtract(*np.percentile(reference, [75, 25])))
    start_threshold = float(np.quantile(reference, 1.0 - 10.0**log_rate))
    return [
        (STATIC, StaticThreshold(log_rate=log_rate)),
        (PROPORTIONAL, LogRatePI(kp=cfg.kp, ki=0.0, log_rate=log_rate, max_step=cfg.max_step)),
        (PI, LogRatePI(kp=cfg.kp, ki=cfg.ki, log_rate=log_rate, max_step=cfg.max_step)),
        (
            TRACKER,
            ScoreSpaceTracker(
                target_rate=cfg.target_alert_rate,
                step=cfg.tracker_step,
                scale=max(scale, 1e-6),
                threshold=start_threshold,
            ),
        ),
    ]


# --------------------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------------------


@dataclass
class GainPoint:
    """One loop gain: how well it tracks, and how hard it works to do it."""

    kp: float
    volume_mae: float
    effort: float
    overshoot: float
    settled: bool


@dataclass
class DelayPoint:
    """One measurement delay: feedback about a state that has already passed."""

    delay: int
    volume_mae: float
    effort: float
    overshoot: float


@dataclass
class AttackOutcome:
    """The control-loop attack, against its own counterfactual: the same policy, no flood."""

    name: str
    recall_without_flood: float
    recall_under_flood: float
    tightest_rate: float
    baseline_rate: float
    recovery_batches: int

    @property
    def suppression(self) -> float:
        """Detection the attacker removed by generating alerts."""
        return self.recall_without_flood - self.recall_under_flood


@dataclass
class ControlStudy:
    """Everything the report renders."""

    traces: list[LoopTrace]
    gains: list[GainPoint]
    delays: list[DelayPoint]
    attack: list[AttackOutcome]
    setpoint: float
    batch_rows: int
    n_batches: int
    target_rate: float
    tolerance: float
    decoys_per_batch: int
    attack_batches: int
    stream_prevalence: float


def run_control_study(settings: Settings) -> ControlStudy:
    """Compare open-loop and closed-loop threshold policies, then stress the loop."""
    cfg: ControlConfig = settings.control
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")

    pipeline = build_pipeline(variant)
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(val), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test), dtype=float)
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.classes_)
    val_scores = positive_scores(np.asarray(model.predict_proba(x_val)), classes)
    test_scores = positive_scores(np.asarray(model.predict_proba(x_test)), classes)

    log_rate = float(np.log10(cfg.target_alert_rate))
    setpoint = cfg.target_alert_rate * cfg.batch_rows
    initial_threshold = float(np.quantile(val_scores, 1.0 - cfg.target_alert_rate))
    bounds = _bounds(len(test_scores), cfg.batch_rows)

    traces = [
        simulate_loop(
            test_scores,
            y_test,
            val_scores,
            policy,
            bounds,
            setpoint=setpoint,
            tolerance=cfg.settling_tolerance,
            initial_threshold=initial_threshold,
            name=name,
        )
        for name, policy in _policies(variant, log_rate, val_scores)
    ]

    gains: list[GainPoint] = []
    for kp in cfg.gain_sweep:
        trace = simulate_loop(
            test_scores,
            y_test,
            val_scores,
            LogRatePI(kp=kp, ki=0.0, log_rate=log_rate, max_step=cfg.max_step),
            bounds,
            setpoint=setpoint,
            tolerance=cfg.settling_tolerance,
            initial_threshold=initial_threshold,
            name=f"P (kp={kp:g})",
        )
        gains.append(GainPoint(kp, trace.volume_mae, trace.effort, trace.overshoot, trace.settled))

    delays: list[DelayPoint] = []
    for delay in cfg.delay_sweep:
        trace = simulate_loop(
            test_scores,
            y_test,
            val_scores,
            LogRatePI(kp=cfg.kp, ki=cfg.ki, log_rate=log_rate, max_step=cfg.max_step),
            bounds,
            setpoint=setpoint,
            tolerance=cfg.settling_tolerance,
            initial_threshold=initial_threshold,
            delay=delay,
            name=f"PI (delay={delay})",
        )
        delays.append(DelayPoint(delay, trace.volume_mae, trace.effort, trace.overshoot))

    attack = _run_attack(
        variant, test_scores, y_test, val_scores, setpoint, log_rate, initial_threshold, rng
    )

    return ControlStudy(
        traces=traces,
        gains=gains,
        delays=delays,
        attack=attack,
        setpoint=setpoint,
        batch_rows=cfg.batch_rows,
        n_batches=len(bounds),
        target_rate=cfg.target_alert_rate,
        tolerance=cfg.settling_tolerance,
        decoys_per_batch=cfg.decoys_per_batch,
        attack_batches=cfg.attack_batches,
        stream_prevalence=float(np.mean(y_test)),
    )


def build_flood(
    settings: Settings,
    scores: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Insert a decoy flood into the stream and mark the flows it is covering for.

    The decoys are *loud*: flows drawn from the top of the score distribution, which any
    threshold alerts on. They cost the attacker nothing -- noisy scanning from throwaway hosts --
    and their only function is to consume the analyst budget. The protected flows are the genuine
    attacks arriving in the same batches, and their detection is what gets measured. Batch
    boundaries are returned explicitly because flooded batches are larger, and a collector
    delivers them as they arrive rather than re-slicing the day.
    """
    cfg: ControlConfig = settings.control
    loud = scores[scores >= np.quantile(scores, cfg.decoy_quantile)]
    out_scores: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    out_protected: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for index, start in enumerate(range(0, len(scores), cfg.batch_rows)):
        stop = min(start + cfg.batch_rows, len(scores))
        batch_scores = scores[start:stop]
        batch_labels = labels[start:stop]
        protected = (batch_labels == 1) & _in_window(index, cfg)
        if _in_window(index, cfg):
            decoys = rng.choice(loud, size=cfg.decoys_per_batch, replace=True)
            batch_scores = np.concatenate([batch_scores, decoys])
            batch_labels = np.concatenate([batch_labels, np.ones(len(decoys), dtype=int)])
            protected = np.concatenate([protected, np.zeros(len(decoys), dtype=bool)])
        out_scores.append(batch_scores)
        out_labels.append(batch_labels)
        out_protected.append(protected)
        bounds.append((cursor, cursor + len(batch_scores)))
        cursor += len(batch_scores)
    return (
        np.concatenate(out_scores),
        np.concatenate(out_labels),
        np.concatenate(out_protected),
        bounds,
    )


def _in_window(index: int, cfg: ControlConfig) -> bool:
    return cfg.attack_start_batch <= index < cfg.attack_start_batch + cfg.attack_batches


def realised_rates(thresholds: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each threshold expressed as the alert rate it would produce on the reference traffic.

    Scores saturate near 1, so thresholds are unreadable at three decimals and misleading at
    six. The rate they imply is the quantity an operator actually reasons about -- "one flow in
    fifty" against "one in three thousand" -- and it is the scale the actuator works in anyway.
    """
    return np.array([float(np.mean(reference >= t)) for t in thresholds], dtype=float)


def _recovery_batches(rates: np.ndarray, baseline: float, start: int, tolerance: float) -> int:
    """Batches after the flood before the alert rate returns within tolerance of its baseline."""
    after = rates[start:]
    within = np.where(np.abs(after - baseline) <= tolerance * max(abs(baseline), 1e-12))[0]
    return int(within[0]) if len(within) else len(after)


def _run_attack(
    settings: Settings,
    scores: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    setpoint: float,
    log_rate: float,
    initial_threshold: float,
    rng: np.random.Generator,
) -> list[AttackOutcome]:
    """Flood the loop with decoys, and compare each policy against itself without the flood."""
    cfg: ControlConfig = settings.control
    flood_scores, flood_labels, protected, flood_bounds = build_flood(settings, scores, labels, rng)
    clean_bounds = _bounds(len(scores), cfg.batch_rows)
    clean_protected = np.zeros(len(scores), dtype=bool)
    for index, (start, stop) in enumerate(clean_bounds):
        if _in_window(index, cfg):
            clean_protected[start:stop] = labels[start:stop] == 1

    def make() -> list[tuple[str, Policy]]:
        return [
            (STATIC, StaticThreshold(log_rate=log_rate)),
            (PI, LogRatePI(kp=cfg.kp, ki=cfg.ki, log_rate=log_rate, max_step=cfg.max_step)),
            (
                PI_GUARDED,
                LogRatePI(
                    kp=cfg.kp,
                    ki=cfg.ki,
                    log_rate=log_rate,
                    max_step=cfg.guarded_max_step,
                    freeze_above=cfg.freeze_above,
                ),
            ),
        ]

    outcomes: list[AttackOutcome] = []
    for (name, flooded_policy), (_, clean_policy) in zip(make(), make(), strict=True):
        under_flood = simulate_loop(
            flood_scores,
            flood_labels,
            reference,
            flooded_policy,
            flood_bounds,
            setpoint=setpoint,
            tolerance=cfg.settling_tolerance,
            initial_threshold=initial_threshold,
            protected=protected,
            name=name,
        )
        without = simulate_loop(
            scores,
            labels,
            reference,
            clean_policy,
            clean_bounds,
            setpoint=setpoint,
            tolerance=cfg.settling_tolerance,
            initial_threshold=initial_threshold,
            protected=clean_protected,
            name=name,
        )
        end_of_flood = cfg.attack_start_batch + cfg.attack_batches
        rates = realised_rates(under_flood.thresholds, reference)
        baseline_rate = float(np.mean(reference >= initial_threshold))
        outcomes.append(
            AttackOutcome(
                name=name,
                recall_without_flood=without.protected_recall,
                recall_under_flood=under_flood.protected_recall,
                tightest_rate=float(np.min(rates)),
                baseline_rate=baseline_rate,
                recovery_batches=_recovery_batches(
                    rates, baseline_rate, end_of_flood, cfg.recovery_tolerance
                ),
            )
        )
    return outcomes


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_control_report(settings: Settings) -> Path:
    """Run the closed-loop study and write the report + figures."""
    study = run_control_study(settings)
    batches = np.arange(1, study.n_batches + 1, dtype=float)
    series = {trace.name: (batches, trace.volumes) for trace in study.traces}
    series["analyst budget"] = (batches, np.full(study.n_batches, study.setpoint))
    trace_fig = plots.plot_lines(
        series,
        xlabel=f"batch (of {study.batch_rows:,} flows)",
        ylabel="alerts raised",
        title="Tracking the analyst budget under drift",
        out_path=settings.paths.figures_dir / TRACE_FIGURE_NAME,
    )
    gains = np.array([point.kp for point in study.gains], dtype=float)
    gain_fig = plots.plot_lines(
        {
            "tracking error (alerts/batch)": (
                gains,
                np.array([p.volume_mae for p in study.gains]),
            ),
            "control effort (x100)": (gains, np.array([p.effort * 100 for p in study.gains])),
        },
        xlabel="proportional gain kp",
        ylabel="alerts per batch  /  actuator movement x100",
        title="Loop gain: tracking against stability",
        out_path=settings.paths.figures_dir / GAIN_FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, trace_fig, gain_fig), encoding="utf-8")
    logger.info("Wrote control report", extra={"path": str(out_path)})

    with track_run(settings, "control") as run:
        run.log_params({"setpoint": study.setpoint, "batch_rows": study.batch_rows})
        for trace in study.traces:
            key = "".join(ch if ch.isalnum() else "_" for ch in trace.name)
            run.log_metrics(
                {
                    f"volume_mae_{key}": trace.volume_mae,
                    f"recall_{key}": trace.recall,
                    f"steady_state_{key}": trace.steady_state,
                }
            )
        for outcome in study.attack:
            key = "".join(ch if ch.isalnum() else "_" for ch in outcome.name)
            run.log_metrics({f"flood_suppression_{key}": outcome.suppression})
        run.log_artifact(trace_fig)
        run.log_artifact(gain_fig)
        run.log_artifact(out_path)
    return out_path


def _trace(study: ControlStudy, name: str) -> LoopTrace:
    return next(t for t in study.traces if t.name == name)


def _tracking_table(study: ControlStudy) -> str:
    rows = [
        "| policy | mean volume error | overshoot | settles | steady-state error | "
        "control effort | recall | precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for trace in study.traces:
        settles = f"batch {trace.settling}" if trace.settled else "never"
        rows.append(
            f"| **{trace.name}** | {trace.volume_mae:.1f} | {trace.overshoot:.0%} | {settles} | "
            f"{trace.steady_state:+.0%} | {trace.effort:.3f} | {trace.recall:.1%} | "
            f"{trace.precision:.1%} |"
        )
    return "\n".join(rows)


def _gain_table(study: ControlStudy) -> str:
    rows = [
        "| gain kp | tracking error (alerts/batch) | control effort | overshoot | settles |",
        "|---|---|---|---|---|",
    ]
    for point in study.gains:
        rows.append(
            f"| {point.kp:g} | {point.volume_mae:.1f} | {point.effort:.3f} | "
            f"{point.overshoot:.0%} | {'yes' if point.settled else 'never'} |"
        )
    return "\n".join(rows)


def _delay_table(study: ControlStudy) -> str:
    rows = [
        "| measurement delay (batches) | tracking error | control effort | overshoot |",
        "|---|---|---|---|",
    ]
    for point in study.delays:
        rows.append(
            f"| {point.delay} | {point.volume_mae:.1f} | {point.effort:.3f} | "
            f"{point.overshoot:.0%} |"
        )
    return "\n".join(rows)


def _attack_table(study: ControlStudy) -> str:
    rows = [
        "| policy | detection of the covered attacks, no flood | under flood | suppression | "
        "tightest alert rate reached | batches to recover |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in study.attack:
        rows.append(
            f"| {outcome.name} | {outcome.recall_without_flood:.1%} | "
            f"{outcome.recall_under_flood:.1%} | **{outcome.suppression:+.1%}** | "
            f"{outcome.tightest_rate:.3%} (from {outcome.baseline_rate:.2%}) | "
            f"{outcome.recovery_batches} |"
        )
    return "\n".join(rows)


def _tracking_read(study: ControlStudy) -> str:
    static = _trace(study, STATIC)
    proportional = _trace(study, PROPORTIONAL)
    pi = _trace(study, PI)
    tracker = _trace(study, TRACKER)
    best = min(study.traces, key=lambda t: t.volume_mae)
    direction = "under" if static.steady_state < 0 else "over"
    open_loop = (
        f"**The open-loop threshold does not deliver the budget it was calibrated for.** "
        f"Calibrated on validation to alert on {study.target_rate:.1%} of flows -- "
        f"{study.setpoint:.0f} per batch -- it lands {static.steady_state:+.0%} off on the later "
        f"days, {direction}-alerting at a mean error of {static.volume_mae:.1f} alerts per batch. "
        "Nothing is broken; the score distribution simply moved, and a threshold fixed in score "
        "space is a promise about a distribution that no longer exists. This is the same failure "
        "the refresh study measured and the certified-budget study bounded, seen from the "
        "queue's side."
    )
    integral = (
        "The integral term earns its place against a *moving* disturbance: proportional control "
        "corrects only what it can currently see, so it lags a target that keeps moving and "
        f"settles at {proportional.steady_state:+.0%}, while the integrator accumulates the "
        f"persistent part of the error and closes it to {pi.steady_state:+.0%}."
        if abs(pi.steady_state) < abs(proportional.steady_state)
        else "**The integral term hurts here, and that is a finding rather than a mis-tuning.** "
        f"PI ends at {pi.steady_state:+.0%} steady-state error against proportional control's "
        f"{proportional.steady_state:+.0%}, and works harder to do it ({pi.effort:.3f} against "
        f"{proportional.effort:.3f} decades of actuator movement per batch). An integrator is "
        "the right instrument for a *persistent* error and the wrong one for a noisy one: this "
        "stream's disturbance is largely batch-to-batch variation, and integrating noise is how "
        "a loop ends up chasing it. The unit tests show the same controller doing exactly what "
        "the theory promises against a genuine drift, which is the useful way to hold both "
        "facts at once -- the mechanism works, and this plant is not the one it is for."
    )
    detection = (
        f"Closing the loop also *raises* detection here — {best.recall:.1%} against the static "
        f"threshold's {static.recall:.1%} — which is not a modelling win but an accounting one: "
        "the open-loop threshold was leaving most of the analyst budget unspent, and the "
        "controller spends it."
        if best.recall > static.recall
        else f"Closing the loop costs detection: {best.recall:.1%} against the static "
        f"threshold's {static.recall:.1%}, which is the price of holding the queue at capacity "
        "when the traffic wants to exceed it."
    )
    return (
        f"{open_loop}\n\n**{best.name}** tracks it best, at {best.volume_mae:.1f} alerts of mean "
        f"error against a {study.setpoint:.0f}-alert setpoint. {integral} The gain-free "
        f"score-space tracker lands at {tracker.volume_mae:.1f} — worth knowing, because it is "
        "the option that does not require anyone to own a control loop's parameters.\n\n"
        f"{detection} What none of these policies can do is change what is *in* the budget: "
        f"the stream is {study.stream_prevalence:.0%} attacks and the budget is "
        f"{study.target_rate:.0%} of flows, so recall here is capacity-bound rather than "
        "model-bound. Volume control is a "
        "volume guarantee. A loop that holds the queue steady while attacks move above the "
        "threshold is doing exactly what it was asked and nothing that was wanted."
    )


def _gain_read(study: ControlStudy) -> str:
    best = min(study.gains, key=lambda p: p.volume_mae)
    worst = max(study.gains, key=lambda p: p.effort)
    unsettled = [p for p in study.gains if not p.settled]
    stability = (
        f"Past kp = {min(p.kp for p in unsettled):g} the loop stops settling inside the "
        f"{study.tolerance:.0%} band at all."
        if unsettled
        else "Every gain in the sweep settles, so the instability boundary is above the range "
        "explored here."
    )
    return (
        f"Tracking error is minimised at **kp = {best.kp:g}** ({best.volume_mae:.1f} alerts per "
        f"batch). {stability} At kp = {worst.kp:g} the actuator moves {worst.effort:.3f} decades "
        f"per batch on average — {worst.effort / max(best.effort, 1e-9):.1f}x the well-tuned "
        "loop's movement — which is the signature of a controller chasing its own corrections "
        "rather than the disturbance. In a detector this reads as an operating point that moves "
        "materially every few minutes: no analyst team accepts it, and nothing calibrated "
        "downstream of the threshold (conformal sets, cost-optimal points, alert SLOs) can track "
        "it."
    )


def _delay_read(study: ControlStudy) -> str:
    zero = study.delays[0]
    worst = max(study.delays, key=lambda p: p.volume_mae)
    if worst.delay == 0:
        return "Delay does not degrade tracking on this stream."
    return (
        f"Feedback is worth what it is timely. Delayed {worst.delay} batches "
        f"({worst.delay * study.batch_rows:,} flows), tracking error rises from "
        f"{zero.volume_mae:.1f} to {worst.volume_mae:.1f} alerts per batch: the loop is "
        "correcting a state that has already passed. Every real deployment has this lag — alerts "
        "are aggregated, dashboards refresh on a schedule, a queue depth is confirmed after the "
        "shift — so the loop's sampling interval has to be slower than its own observation lag, "
        "which caps how fast any volume controller can respond regardless of gain."
    )


def _attack_read(study: ControlStudy) -> str:
    by_name = {outcome.name: outcome for outcome in study.attack}
    static = by_name[STATIC]
    pi = by_name[PI]
    guarded = by_name[PI_GUARDED]
    recovered = guarded.recall_under_flood - pi.recall_under_flood
    if pi.suppression > 0.01:
        headline = (
            f"**The loop can be driven.** The same PI controller that detects "
            f"{pi.recall_without_flood:.1%} of those attacks without the flood detects "
            f"{pi.recall_under_flood:.1%} with it — a **{pi.suppression:.1%} suppression the "
            "attacker bought by generating alerts**, which is the opposite of what generating "
            f"alerts is supposed to do to them. The static threshold moves "
            f"{static.suppression:+.1%}, because it cannot be driven: it is not listening."
        )
    else:
        headline = (
            f"On this stream the flood opens no usable gap: the PI loop detects "
            f"{pi.recall_under_flood:.1%} of the covered attacks against "
            f"{pi.recall_without_flood:.1%} without the flood ({pi.suppression:+.1%}). The "
            "mechanism is real and the magnitude here is not, and the reason is worth stating "
            "rather than hiding: the actuator's rate limit already bounds the operating point "
            f"to {pi.tightest_rate:.3%} of flows at its tightest against a "
            f"{pi.baseline_rate:.2%} baseline, so ten batches of flooding cannot move it far "
            "enough to matter. The rate "
            "limit was put there for stability; it turns out to be a security control."
        )
    mitigation = (
        f"The guard recovers {recovered:+.1%} on top of that."
        if recovered > 0.005
        else "The guard's contribution is containment rather than detection: it tightens only to "
        f"{guarded.tightest_rate:.3%} of flows against the unguarded loop's "
        f"{pi.tightest_rate:.3%}, and returns to its operating point in "
        f"{guarded.recovery_batches} batches against {pi.recovery_batches}."
    )
    return (
        f"{headline} {mitigation}\n\nThis is the part the control-theory textbook does not cover, "
        "because its plants are not adversarial. Every feedback loop in a security system turns "
        "its own input into an attack surface: an adaptive threshold can be pushed, an adaptive "
        "baseline can be poisoned (which the poisoning study measured on the training set — this "
        "is its deployment-time cousin), and a rate limiter can be used to silence what it "
        "protects. The mitigations are not clever — freeze the integrator during a surge, bound "
        "the actuator's movement per interval, and treat a large volume excursion as an incident "
        "rather than as a setpoint error — but they have to be *designed in*, because the "
        "version without them is the version a textbook hands you."
    )


def _render(study: ControlStudy, trace_fig: Path, gain_fig: Path) -> str:
    return f"""# NetSentry — Closed-Loop Threshold Control (and the Attack on It)

_{study.n_batches} batches of {study.batch_rows:,} later-day flows. Setpoint
{study.setpoint:.0f} alerts per batch ({study.target_rate:.0%} of traffic — the analyst budget).
The actuator is `log10` of the alert rate; the open-loop baseline is the same rate, calibrated
once on validation and never touched. Settling band {study.tolerance:.0%}._

## Why this report exists

A SOC has a fixed number of analysts. The detector has a fixed threshold. Those two facts stop
being compatible the moment traffic changes, and every threshold in this project so far has been
**open-loop**: chosen on a validation set at a target rate, shipped, and left. The
threshold-refresh study watched that decay; the Neyman-Pearson study found the deployed rule
violating its own false-positive budget 51% of the time. Neither closes the loop.

This one does. Alert volume is a measured output, the threshold is an actuator, the analyst
budget is a setpoint — a feedback control problem, with a century of theory attached that says
more than "raise it when there are too many alerts" does.

**The actuator is `log10` of the alert rate, not the threshold and not its quantile.** That
choice is most of the engineering: near the operating point a thousandth of a quantile separates
ten alerts from a hundred, so a gain tuned in one regime is wrong in the next, and a loop tuned
on Tuesday oscillates on Friday. In log-rate units the plant is close to a unit gain — one decade
of actuator buys one decade of volume — the error is a *ratio* (twice the budget means the same
thing at ten alerts and at a thousand), and `kp = 1` is roughly deadbeat by construction.

## What each policy achieves

{_tracking_table(study)}

![Alert volume against the budget](../figures/{trace_fig.name})

{_tracking_read(study)}

## Loop gain: the stability boundary, located rather than assumed

{_gain_table(study)}

![Gain sweep](../figures/{gain_fig.name})

{_gain_read(study)}

## Late feedback

{_delay_table(study)}

{_delay_read(study)}

## The control-loop attack

An attacker who can raise alert volume can make the controller raise its own threshold, and then
walk through the gap they created. The flood is {study.decoys_per_batch:,} loud decoy flows per
batch for {study.attack_batches} batches — noisy scanning from throwaway hosts, cheap to
generate, certain to alert — and the flows measured are the *genuine* attacks arriving in those
same batches. The counterfactual is the only honest one available: the same policy, on the same
flows, without the flood.

{_attack_table(study)}

{_attack_read(study)}

## Scope and honest limits

- **The plant is a replay, not a live queue.** Batches arrive at a fixed size and the measurement
  is the alert count; a real loop also contends with analyst throughput varying within a shift,
  which the SOC queue simulation models and this does not.
- **One disturbance profile.** The stream's drift is whatever the later capture days contain,
  plus the injected flood. A controller tuned against this profile is not thereby tuned against
  another — the standing objection to any empirically tuned loop, and the reason the gain sweep
  is reported rather than a single recommended number.
- **Volume is not risk.** Holding the queue at capacity says nothing about whether the *right*
  flows are in it; the alert-queue and cost studies are where that question lives. Reading a
  volume controller as a detection improvement would be a mistake.
- **The attack is a lower bound.** A smarter adversary would shape the flood to the loop's time
  constant rather than flooding flat, and would use the recovery window rather than the flood
  window. The mitigation is designed against the mechanism, not against this schedule."""
