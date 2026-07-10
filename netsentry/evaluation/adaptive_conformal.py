"""Adaptive conformal inference: repairing the coverage that drift broke, online.

The conformal report ends on its most honest finding: the split-conformal
guarantee needs exchangeability, the temporal split breaks it on purpose, and
attack-class coverage collapses (~64% against a 90% target on the stand-in).
Gibbs & Candes (2021) showed the repair — treat the miscoverage level alpha as a
*control variable* and steer it with the realized errors:

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

Under-coverage (err_t = 1 too often) drives alpha_t down, widening the sets;
over-coverage lets it drift back up, tightening them. The update needs no
distributional assumption at all — its long-run coverage guarantee holds for
*arbitrary*, even adversarial, distribution shift — but it does need the ground
truth ``err_t``, i.e. labels, possibly delayed. That is the honest trade this
study prices: adaptive conformal buys back the coverage *guarantee* at the cost
of (a) label feedback and (b) wider sets, i.e. a larger human-review share. It
does not make the underlying model better — the sets widen exactly where the
model is blind, which is the correct behavior for a safety layer.

Class-conditional (Mondrian) like the static baseline: each class's alpha is
updated only by labeled flows of that class, so rare-class coverage is steered
by rare-class evidence rather than swamped by the majority class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.conformal import class_conditional_thresholds
from netsentry.log import get_logger
from netsentry.monitoring.streaming import order_stream
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "adaptive_conformal.md"


class QuantileTable:
    """O(1) split-conformal quantile lookups for a *fixed* calibration set.

    Adaptive conformal re-evaluates the quantile at a new alpha on every step, so
    the calibration scores are sorted once and each lookup is an order-statistic
    index. For alpha in (0, 1) the lookup reproduces ``conformal_quantile`` exactly
    (same finite-sample correction, same ``method="higher"`` index arithmetic), so
    the adaptive run starts from the static baseline's thresholds; alphas outside
    (0, 1) get the ACI reading — alpha <= 0 means "include everything" (+inf),
    alpha >= 1 means the empty set (-inf).
    """

    def __init__(self, nonconformity: np.ndarray) -> None:
        self._sorted = np.sort(np.asarray(nonconformity, dtype=float))

    def threshold(self, alpha: float) -> float:
        n = len(self._sorted)
        if n == 0 or alpha <= 0.0:
            return float("inf")
        if alpha >= 1.0:
            return float("-inf")
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        index = int(np.ceil((n - 1) * level))  # np.quantile's "higher" method
        return float(self._sorted[index])


@dataclass
class AdaptiveAlpha:
    """The Gibbs-Candes update for one class's miscoverage level.

    ``alpha`` is deliberately *not* clamped to [0, 1]: letting it run negative
    (sets forced maximally wide) is what makes the long-run coverage guarantee
    hold under arbitrary shift; the set constructor interprets the excursions.
    """

    target: float
    gamma: float
    alpha: float = field(init=False)
    history: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.alpha = self.target

    def update(self, err: bool) -> float:
        self.alpha += self.gamma * (self.target - float(err))
        self.history.append(self.alpha)
        return self.alpha


@dataclass
class StreamOutcome:
    """Per-flow set membership and coverage errors for one policy over the stream."""

    in_benign: np.ndarray  # bool per flow
    in_attack: np.ndarray
    err: np.ndarray  # 1 where the flow's true class was missing from its set

    def coverage(self, y: np.ndarray, label: int) -> float:
        mask = np.asarray(y) == label
        if not mask.any():
            return float("nan")
        member = self.in_benign if label == 0 else self.in_attack
        return float(np.mean(member[mask]))

    def review_rate(self) -> float:
        ambiguous = self.in_benign & self.in_attack
        empty = ~self.in_benign & ~self.in_attack
        return float(np.mean(ambiguous | empty))


def run_static_stream(
    p: np.ndarray, y: np.ndarray, tau_benign: float, tau_attack: float
) -> StreamOutcome:
    """The frozen split-conformal baseline: fixed thresholds over the whole stream."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    in_benign = p <= tau_benign
    in_attack = (1.0 - p) <= tau_attack
    err = np.where(y == 0, ~in_benign, ~in_attack).astype(int)
    return StreamOutcome(in_benign=in_benign, in_attack=in_attack, err=err)


def run_adaptive_stream(
    p: np.ndarray,
    y: np.ndarray,
    cal_p: np.ndarray,
    cal_y: np.ndarray,
    target_alpha: float,
    gamma: float,
    label_delay: int = 0,
) -> tuple[StreamOutcome, AdaptiveAlpha, AdaptiveAlpha]:
    """Class-conditional ACI over a labeled stream (labels may arrive delayed).

    Each flow is judged at the *current* per-class alphas; then the label of the
    flow ``label_delay`` steps back (if any) updates its class's alpha. With
    delay 0 the update is immediate — the ACI ideal; a positive delay models the
    triage lag before an analyst confirms ground truth.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    cal_p = np.asarray(cal_p, dtype=float)
    cal_y = np.asarray(cal_y, dtype=int)
    benign_table = QuantileTable(cal_p[cal_y == 0])  # benign nonconformity = p
    attack_table = QuantileTable(1.0 - cal_p[cal_y == 1])  # attack nonconformity = 1 - p

    alpha_benign = AdaptiveAlpha(target=target_alpha, gamma=gamma)
    alpha_attack = AdaptiveAlpha(target=target_alpha, gamma=gamma)

    n = len(p)
    in_benign = np.zeros(n, dtype=bool)
    in_attack = np.zeros(n, dtype=bool)
    err = np.zeros(n, dtype=int)
    for t in range(n):
        in_benign[t] = p[t] <= benign_table.threshold(alpha_benign.alpha)
        in_attack[t] = (1.0 - p[t]) <= attack_table.threshold(alpha_attack.alpha)
        err[t] = int(~in_benign[t] if y[t] == 0 else ~in_attack[t])
        feedback = t - label_delay
        if feedback >= 0:
            if y[feedback] == 0:
                alpha_benign.update(bool(err[feedback]))
            else:
                alpha_attack.update(bool(err[feedback]))
    outcome = StreamOutcome(in_benign=in_benign, in_attack=in_attack, err=err)
    return outcome, alpha_benign, alpha_attack


def rolling_class_coverage(
    outcome: StreamOutcome, y: np.ndarray, label: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """(positions, coverage) over trailing windows of the stream, for one class."""
    y = np.asarray(y)
    member = outcome.in_benign if label == 0 else outcome.in_attack
    positions, coverage = [], []
    for end in range(window, len(y) + 1, window):
        mask = y[end - window : end] == label
        if mask.any():
            positions.append(end)
            coverage.append(float(np.mean(member[end - window : end][mask])))
    return np.array(positions), np.array(coverage)


def run_adaptive_conformal_report(settings: Settings) -> Path:
    """Static vs adaptive conformal on the temporal stream; write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    # Calibrated probabilities, matching the static conformal report's scale; the
    # test flows are replayed in capture (day) order so "online" means online.
    val = load_split(variant, "temporal", "val")
    test = order_stream(load_split(variant, "temporal", "test"))
    cal_p = result.bundle.attack_scores(val)
    cal_y = val[BINARY_TARGET].to_numpy().astype(int)
    p_stream = result.bundle.attack_scores(test)
    y_stream = test[BINARY_TARGET].to_numpy().astype(int)

    cfg = settings.adaptive_conformal
    target = settings.conformal.alpha
    tau_b, tau_a = class_conditional_thresholds(cal_p, cal_y, target)
    static = run_static_stream(p_stream, y_stream, tau_b, tau_a)
    adaptive, _alpha_b, alpha_a = run_adaptive_stream(
        p_stream, y_stream, cal_p, cal_y, target, cfg.gamma, cfg.label_delay
    )

    xs, static_cov = rolling_class_coverage(static, y_stream, 1, cfg.window)
    _, adaptive_cov = rolling_class_coverage(adaptive, y_stream, 1, cfg.window)
    fig = plots.plot_lines(
        {
            "target (1 - alpha)": (xs, np.full(len(xs), 1.0 - target)),
            "static conformal": (xs, static_cov),
            "adaptive (ACI)": (xs, adaptive_cov),
        },
        xlabel="Stream position (flows, time order)",
        ylabel=f"Attack coverage (trailing {cfg.window}-flow windows)",
        title="Adaptive conformal restores attack coverage under drift",
        out_path=settings.paths.figures_dir / "adaptive_conformal.png",
    )

    report = _render(settings, static, adaptive, alpha_a, y_stream, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote adaptive-conformal report", extra={"path": str(out_path)})

    with track_run(settings, "adaptive_conformal") as run:
        run.log_params({"gamma": cfg.gamma, "label_delay": cfg.label_delay})
        run.log_metrics(
            {
                "static_attack_coverage": static.coverage(y_stream, 1),
                "adaptive_attack_coverage": adaptive.coverage(y_stream, 1),
                "static_review_rate": static.review_rate(),
                "adaptive_review_rate": adaptive.review_rate(),
                "alpha_attack_final": alpha_a.alpha,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(
    settings: Settings,
    static: StreamOutcome,
    adaptive: StreamOutcome,
    alpha_attack: AdaptiveAlpha,
    y: np.ndarray,
    fig: Path,
) -> str:
    cfg = settings.adaptive_conformal
    target = 1.0 - settings.conformal.alpha
    rows = [
        "| policy | attack coverage | benign coverage | auto-decided | human review |",
        "|---|---|---|---|---|",
        f"| static split-conformal | {static.coverage(y, 1):.1%} | {static.coverage(y, 0):.1%} "
        f"| {1 - static.review_rate():.1%} | {static.review_rate():.1%} |",
        f"| adaptive (ACI) | **{adaptive.coverage(y, 1):.1%}** | {adaptive.coverage(y, 0):.1%} "
        f"| {1 - adaptive.review_rate():.1%} | {adaptive.review_rate():.1%} |",
    ]
    alpha_min = min(alpha_attack.history) if alpha_attack.history else alpha_attack.target
    review_delta = adaptive.review_rate() - static.review_rate()

    static_short = static.coverage(y, 1) < target - 0.02
    adaptive_ok = adaptive.coverage(y, 1) >= target - 0.02
    excursion = (
        " (negative: the sets are forced maximally wide until the drift passes)"
        if alpha_min < 0
        else ""
    )
    if static_short and adaptive_ok:
        read = (
            f"The repair works: static attack coverage runs at {static.coverage(y, 1):.1%} "
            f"against a {target:.0%} target (the exchangeability break the conformal report "
            f"documents), and the online alpha update brings it back to "
            f"{adaptive.coverage(y, 1):.1%}. The price is explicit in the last column — the "
            f"human-review share moves by {review_delta:+.1%} — and in the alpha trajectory: "
            f"alpha_attack is driven from {alpha_attack.target:g} to a minimum of "
            f"{alpha_min:.3f}{excursion}. "
            "Coverage is bought with wider sets, not with a better model."
        )
    elif static_short:
        read = (
            f"Static attack coverage falls short ({static.coverage(y, 1):.1%} vs the "
            f"{target:.0%} target) and the adaptive run recovers only part of it "
            f"({adaptive.coverage(y, 1):.1%}) at these settings — a larger gamma reacts "
            "faster at the cost of noisier set sizes. The direction is right; the tuning "
            "is a deployment choice."
        )
    else:
        read = (
            f"On this run static coverage already meets the target "
            f"({static.coverage(y, 1):.1%} vs {target:.0%}), so the adaptive layer has "
            "little to repair and tracks it — the mechanism only earns its keep under "
            "drift, which is exactly when a static guarantee quietly fails."
        )

    return f"""# NetSentry — Adaptive Conformal Inference (coverage under drift)

_Synthetic stand-in. Temporal split; the binary model's calibrated probabilities;
the test (later-day) flows replayed in capture order as a labeled stream.
Class-conditional split-conformal calibrated on validation is the frozen
baseline; the adaptive run applies the Gibbs-Candes update alpha_(t+1) = alpha_t
+ gamma (alpha - err_t) per class with gamma = {cfg.gamma:g} and a label delay of
{cfg.label_delay} flows. Target coverage {target:.0%}._

## The problem this solves

The conformal report shows the guarantee doing exactly what the theory says: it
holds on the exchangeable split and **fails on the temporal one** — later-day
novel attacks fall outside the calibrated sets, and attack coverage collapses.
Static conformal treats alpha as a constant; adaptive conformal inference (ACI)
treats it as a control signal and steers it with the realized coverage errors.
Its long-run coverage guarantee needs **no distributional assumption at all** —
it holds under arbitrary shift — but it consumes ground-truth labels, so what it
really trades is analyst feedback for a live guarantee.

## Whole-stream results

{chr(10).join(rows)}

![Rolling attack coverage]({"../figures/" + fig.name})

## Read

{read}

## What ACI does and does not buy

It restores the *guarantee*, not the *detector*: the sets widen precisely where
the model is uncertain or blind, which converts silent misses into explicit
review items — the correct failure mode for a safety layer, and the same
"abstention is the review budget" contract the static conformal report
established. It does not improve ranking, detection at a fixed FPR, or any other
model quality; retraining (the streaming study) is the lever for those. And it
needs labels: with a label delay the update lags the drift by exactly that
delay, so the two knobs a deployment tunes are gamma (reaction speed vs
stability) and the freshness of its feedback loop.
"""
