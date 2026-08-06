"""How long must the shadow model run before you can believe it is better?

The serving stack already scores a [shadow challenger](../../netsentry/serving/inference.py)
silently alongside the champion, and the [promotion](promotion.md) study compares two models
with a paired bootstrap. Both leave the operational question unanswered: **when do you stop
watching?** In practice somebody checks the dashboard each morning, sees the challenger
ahead, and promotes — and that habit is a statistical disaster with a specific name.

A fixed-sample test earns its 5% error rate by being evaluated **once**, at a sample size
fixed in advance. Evaluate it repeatedly as data arrives and stop the first time it looks
significant, and the false-positive rate is no longer 5%; under the null the test statistic
random-walks and will eventually wander across any fixed boundary, so peeking inflates the
error toward certainty. This report does not assert that — it measures it, by running the
null (two models that are genuinely equivalent) through exactly the peeking behaviour a real
team exhibits and recording how often the fixed-n test fires.

The fix is a **confidence sequence**: an interval that is valid *simultaneously at every
sample size*, so an operator may look as often as they like, stop whenever they like, and
still have the stated coverage. The construction here is Robbins' normal mixture (1970), the
canonical anytime-valid interval and the foundation of the modern treatment in Howard,
Ramdas, McAuliffe & Sekhon (*Time-uniform Chernoff bounds via nonnegative supermartingales*,
2020/21). Its boundary at accumulated variance `V` is
`sqrt((V + rho) * log((V + rho) / (rho * alpha^2)))`, and the mixing parameter `rho` tunes
which sample size the bound is tightest at — the honest cost of anytime validity, paid as a
slightly wider interval than a fixed-n test that has *earned* the right to be narrow by
committing to its sample size.

The report puts the three regimes side by side on the same paired stream: the fixed-n sample
size a power calculation demands, the measured error rate of peeking at that test, and the
confidence sequence's stopping time and measured error rate. The comparison is the point —
anytime validity is not free, and what it buys is the right to run the deployment process
the way people actually run it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.monitoring.streaming import order_stream
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SequentialABConfig

logger = get_logger(__name__)

REPORT_NAME = "sequential_ab.md"
FIGURE_NAME = "sequential_ab.png"

# Two-sided normal quantiles, hard-coded for the two conventional levels so the module has
# no scipy dependency (the rest of the package treats scipy as optional).
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
_Z_POWER = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def brier_loss(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-flow squared error — a proper scoring rule, bounded, and paired-friendly.

    Bounded matters: the confidence sequence's variance proxy is only honest if the
    observations cannot take arbitrarily large values, and log-loss is unbounded when a
    confident model is wrong.
    """
    error: np.ndarray = np.asarray(scores, dtype=float) - np.asarray(y, dtype=float)
    return error**2


def mixture_boundary(v: float, rho: float, alpha: float) -> float:
    """Robbins' normal-mixture boundary on a sum with accumulated variance ``v``.

    ``sqrt((v + rho) * log((v + rho) / (rho * alpha^2)))``. The bound holds **uniformly over
    all sample sizes** with probability at least ``1 - alpha`` — the property a fixed-n
    interval does not have and the reason peeking is safe here. ``rho`` tunes the sample size
    at which the boundary is tightest: small ``rho`` favours early stopping, large ``rho``
    favours precision later.
    """
    if rho <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("rho must be positive and alpha must lie in (0, 1)")
    return math.sqrt((v + rho) * math.log((v + rho) / (rho * alpha**2)))


def confidence_sequence(
    diffs: np.ndarray, *, rho: float, alpha: float, sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Running mean and anytime-valid interval over a paired difference stream.

    Returns ``(means, lowers, uppers)``, one entry per observation, valid at every ``t``
    **simultaneously** — so an operator may stop whenever the interval excludes zero without
    inflating the error rate.

    ``sigma`` is the sub-Gaussian scale of a single difference and must be **fixed in
    advance**, not re-estimated as the stream grows: the intrinsic time in Robbins' boundary
    is ``t * sigma^2``, and substituting a running sample variance makes the early intervals
    absurdly narrow (the sample variance of one observation is zero) and destroys the very
    coverage the construction exists to provide. Paired Brier differences are bounded in
    [-1, 1], so ``sigma = 1`` is always valid by Hoeffding; a tighter proxy estimated on a
    held-out warm-up is admissible and much less conservative.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    idx = np.arange(1, n + 1)
    means = np.cumsum(d) / idx
    radii = np.array([mixture_boundary(float(t) * sigma**2, rho, alpha) / t for t in idx])
    return means, means - radii, means + radii


def fixed_n_required(delta: float, sd: float, *, alpha: float, power: float) -> int:
    """Sample size a fixed-n two-sided test needs to detect ``delta`` at ``power``.

    ``n = 2 * ((z_alpha/2 + z_power) * sd / delta)^2`` for a paired mean comparison
    (the factor drops to 1 for a paired test, which is what this is — the pairing is why the
    shadow deployment is worth running at all).
    """
    if delta == 0:
        return 0
    z_a = _Z.get(round(alpha, 2), 1.96)
    z_b = _Z_POWER.get(round(power, 2), 0.8416)
    return math.ceil(((z_a + z_b) * sd / abs(delta)) ** 2)


def fixed_n_significant(diffs: np.ndarray, alpha: float) -> bool:
    """Would a fixed-n two-sided test on this whole sample call the difference real?"""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return False
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return bool(d.mean() != 0)
    z = abs(float(d.mean())) / (sd / math.sqrt(n))
    return z > _Z.get(round(alpha, 2), 1.96)


def peeking_error_rate(
    n_trials: int, n_obs: int, checkpoints: int, alpha: float, rng: np.random.Generator
) -> float:
    """Measured false-positive rate of a fixed-n test evaluated repeatedly under the null.

    Draws genuinely null streams (two equivalent models, so the paired difference has mean
    zero), walks each one, applies the fixed-n test at every checkpoint, and counts the
    streams where it *ever* fires. This is the exact behaviour of a team that checks the
    dashboard each morning and promotes on the first green light.
    """
    if n_trials <= 0 or n_obs <= 0:
        return 0.0
    stops = np.linspace(n_obs // checkpoints, n_obs, checkpoints, dtype=int)
    fired = 0
    for _ in range(n_trials):
        stream = rng.normal(0.0, 1.0, size=n_obs)  # the null: no real difference
        if any(fixed_n_significant(stream[:t], alpha) for t in stops if t >= 2):
            fired += 1
    return fired / n_trials


def sequence_error_rate(
    n_trials: int,
    n_obs: int,
    alpha: float,
    rho: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> float:
    """Measured false-positive rate of the confidence sequence under the same null.

    The comparison that makes the point: identical peeking behaviour, an interval built to
    survive it. Should land at or below ``alpha``, and typically well below, because the
    bound is conservative.
    """
    if n_trials <= 0 or n_obs <= 0:
        return 0.0
    fired = 0
    for _ in range(n_trials):
        stream = rng.normal(0.0, 1.0, size=n_obs)
        _, lower, upper = confidence_sequence(stream, rho=rho, alpha=alpha, sigma=sigma)
        if np.any((lower > 0) | (upper < 0)):
            fired += 1
    return fired / n_trials


def first_conclusive(lower: np.ndarray, upper: np.ndarray) -> int:
    """First observation index (1-based) at which the interval excludes zero; 0 if never."""
    conclusive = np.flatnonzero((lower > 0) | (upper < 0))
    return int(conclusive[0]) + 1 if conclusive.size else 0


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class SequentialABStudy:
    """Everything the report renders."""

    n_stream: int
    alpha: float
    power: float
    rho: float
    champion_loss: float
    challenger_loss: float
    observed_delta: float
    diff_sd: float
    fixed_n: int
    fixed_n_verdict: bool
    cs_stop: int
    cs_stop_mean: float
    cs_reopened: bool
    cs_lower: float
    cs_upper: float
    cs_final_mean: float
    peeking_error: float
    sequence_error: float
    n_null_trials: int
    checkpoints: int
    sigma: float
    means: np.ndarray
    lowers: np.ndarray
    uppers: np.ndarray

    @property
    def winner(self) -> str:
        if self.cs_stop == 0:
            return "inconclusive"
        return "challenger" if self.observed_delta > 0 else "champion"


def run_sequential_ab(settings: Settings) -> SequentialABStudy:
    """Compare champion and challenger on one paired stream, three ways."""
    cfg: SequentialABConfig = settings.sequential_ab
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))

    champion = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    challenger_cfg = variant.model_copy(deep=True)
    challenger_cfg.supervised.num_leaves = cfg.challenger_num_leaves
    challenger_cfg.supervised.learning_rate = cfg.challenger_learning_rate
    challenger_cfg.seed = variant.seed + 1
    seed_everything(challenger_cfg.seed)
    challenger = SupervisedClassifier(challenger_cfg).fit(x_train, y_train, eval_set=(x_val, y_val))

    # The stream is ordered as it arrives in deployment, not shuffled: a shadow test watches
    # traffic in time order, and a mid-week distribution change is part of what it observes.
    stream = order_stream(test)
    y_stream = stream[BINARY_TARGET].to_numpy().astype(int)
    x_stream = np.asarray(pipeline.transform(stream))
    if cfg.max_stream and len(y_stream) > cfg.max_stream:
        x_stream, y_stream = x_stream[: cfg.max_stream], y_stream[: cfg.max_stream]

    def _diffs(
        model_a: SupervisedClassifier, model_b: SupervisedClassifier, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Paired per-flow loss advantage of ``model_b`` over ``model_a`` (positive = b better)."""
        sa = attack_probability(np.asarray(model_a.predict_proba(x)), model_a.classes_, benign)
        sb = attack_probability(np.asarray(model_b.predict_proba(x)), model_b.classes_, benign)
        advantage: np.ndarray = brier_loss(sa, y) - brier_loss(sb, y)
        return advantage

    champ_scores = attack_probability(
        np.asarray(champion.predict_proba(x_stream)), champion.classes_, benign
    )
    chal_scores = attack_probability(
        np.asarray(challenger.predict_proba(x_stream)), challenger.classes_, benign
    )
    champ_loss = brier_loss(champ_scores, y_stream)
    chal_loss = brier_loss(chal_scores, y_stream)
    # Positive difference = challenger is better (lower loss). Paired per flow, which is the
    # entire statistical advantage of running a shadow rather than an A/B split.
    diffs = champ_loss - chal_loss

    # The scale proxy is fixed in advance and estimated on **validation**, never on the stream
    # the sequence then judges: calibrating a bound on the data it is about to be applied to
    # is the same circularity this project's split rules exist to prevent, and here it would
    # also be measured on a prefix (one capture day) that is not representative of the rest.
    sigma = max(float(np.std(_diffs(champion, challenger, x_val, y_val), ddof=1)), 1e-6)
    means, lowers, uppers = confidence_sequence(diffs, rho=cfg.rho, alpha=cfg.alpha, sigma=sigma)
    stop = first_conclusive(lowers, uppers)
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    rng = np.random.default_rng(variant.seed)
    peeking = peeking_error_rate(cfg.n_null_trials, cfg.null_obs, cfg.checkpoints, cfg.alpha, rng)
    sequence = sequence_error_rate(cfg.n_null_trials, cfg.null_obs, cfg.alpha, cfg.rho, rng)

    logger.info(
        "Sequential A/B measured",
        extra={"delta": float(diffs.mean()), "cs_stop": stop, "peeking_error": peeking},
    )
    return SequentialABStudy(
        n_stream=len(diffs),
        alpha=cfg.alpha,
        power=cfg.power,
        rho=cfg.rho,
        champion_loss=float(champ_loss.mean()),
        challenger_loss=float(chal_loss.mean()),
        observed_delta=float(diffs.mean()),
        diff_sd=sd,
        fixed_n=fixed_n_required(float(diffs.mean()), sd, alpha=cfg.alpha, power=cfg.power),
        fixed_n_verdict=fixed_n_significant(diffs, cfg.alpha),
        cs_stop=stop,
        cs_stop_mean=float(means[stop - 1]) if stop else 0.0,
        # A confidence sequence may re-include zero after an early crossing: the running mean
        # keeps moving while the interval narrows. Recorded rather than hidden, because the
        # guarantee attaches to the *stopped decision*, not to the interval's later shape.
        cs_reopened=bool(stop and lowers[-1] <= 0.0 <= uppers[-1]),
        cs_lower=float(lowers[-1]),
        cs_upper=float(uppers[-1]),
        cs_final_mean=float(means[-1]),
        peeking_error=peeking,
        sequence_error=sequence,
        n_null_trials=cfg.n_null_trials,
        checkpoints=cfg.checkpoints,
        sigma=sigma,
        means=means,
        lowers=lowers,
        uppers=uppers,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_sequential_ab_report(settings: Settings) -> Path:
    """Run the sequential A/B study and write the report + figure."""
    study = run_sequential_ab(settings)

    # Plot on a log-spaced subsample: the interval's early width spans orders of magnitude.
    n = study.n_stream
    idx = np.unique(np.geomspace(2, max(n, 2), num=min(n, 120)).astype(int)) - 1
    x = (idx + 1).astype(float)
    fig = plots.plot_lines(
        {
            "loss advantage (challenger - champion)": (x, study.means[idx]),
            "anytime-valid lower bound": (x, study.lowers[idx]),
            "anytime-valid upper bound": (x, study.uppers[idx]),
            "no difference": (x, np.zeros(len(idx))),
        },
        xlabel="flows observed by the shadow model",
        ylabel="paired Brier-loss difference",
        title="An interval you may look at as often as you like",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote sequential A/B report", extra={"path": str(out_path)})

    with track_run(settings, "sequential_ab") as run:
        run.log_params({"alpha": study.alpha, "rho": study.rho, "power": study.power})
        run.log_metrics(
            {
                "observed_delta": study.observed_delta,
                "fixed_n_required": float(study.fixed_n),
                "cs_stop": float(study.cs_stop),
                "peeking_error_rate": study.peeking_error,
                "sequence_error_rate": study.sequence_error,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _regimes_table(study: SequentialABStudy) -> str:
    stop = f"{study.cs_stop:,} flows" if study.cs_stop else "never (inconclusive)"
    fixed_verdict = "significant" if study.fixed_n_verdict else "not significant"
    return "\n".join(
        [
            "| procedure | may you peek? | decides after | measured error rate under the null |",
            "|---|---|---|---|",
            f"| fixed-n test, evaluated once | no | {study.fixed_n:,} flows (power calculation) "
            f"| {study.alpha:.0%} by construction |",
            f"| fixed-n test, checked {study.checkpoints}x as data arrives | no (but everyone "
            f"does) | first green light | **{study.peeking_error:.1%}** |",
            f"| confidence sequence (Robbins mixture) | **yes, always** | {stop} "
            f"| {study.sequence_error:.1%} |",
            f"| _(the whole stream, fixed-n verdict)_ | — | {study.n_stream:,} flows "
            f"| {fixed_verdict} |",
        ]
    )


def _peeking_read(study: SequentialABStudy) -> str:
    inflation = study.peeking_error / max(study.alpha, 1e-9)
    return (
        f"The peeking row is measured, not asserted: {study.n_null_trials} streams were drawn "
        f"from a genuine null — two models with no real difference — and the fixed-n test was "
        f"applied at {study.checkpoints} checkpoints along each one, exactly as a team that "
        f"checks the dashboard periodically would. It fired on **{study.peeking_error:.1%}** of "
        f"them, against the {study.alpha:.0%} it advertises: a {inflation:.1f}x inflation of the "
        "false-positive rate, achieved without a single line of bad code. The mechanism is not "
        "subtle — under the null the test statistic random-walks, and given enough looks it will "
        "cross any fixed boundary eventually — but it is invisible in practice, because the "
        "promotion that follows a peeked-at result looks exactly like a promotion that followed "
        "a valid one. The confidence sequence, run through the identical peeking behaviour, "
        f"fired on {study.sequence_error:.1%}."
    )


def _decision_read(study: SequentialABStudy) -> str:
    if study.cs_stop == 0:
        return (
            f"On the real paired stream the challenger's mean Brier loss is "
            f"{study.challenger_loss:.4f} against the champion's {study.champion_loss:.4f}, a "
            f"paired advantage of {study.observed_delta:+.5f} per flow. The anytime-valid "
            f"interval after all {study.n_stream:,} flows is "
            f"[{study.cs_lower:+.5f}, {study.cs_upper:+.5f}] — it still contains zero, so the "
            "honest answer is **not yet**. That is the outcome worth dwelling on: the correct "
            "response to an inconclusive shadow test is to keep the champion and keep watching, "
            "and a procedure that cannot say 'not yet' will instead say something wrong. The "
            f"fixed-n power calculation agrees on the scale of the problem — detecting an effect "
            f"this small at {study.power:.0%} power needs about {study.fixed_n:,} flows."
        )
    return (
        f"On the real paired stream the challenger's mean Brier loss is "
        f"{study.challenger_loss:.4f} against the champion's {study.champion_loss:.4f}, a paired "
        f"advantage of {study.observed_delta:+.5f} per flow. The anytime-valid interval first "
        f"excluded zero after **{study.cs_stop:,} flows** (running advantage "
        f"{study.cs_stop_mean:+.5f} at that point), so the **{study.winner}** wins and the shadow "
        f"test could have been stopped there. A fixed-n design targeting the same effect "
        f"at {study.power:.0%} power would have committed to {study.fixed_n:,} flows in advance "
        + (
            "— more than the sequence needed, which is the usual result: fixed-n sizing must "
            "budget for the effect being exactly as small as specified, while a sequential "
            "procedure stops early when the effect is larger than feared."
            if study.fixed_n > study.cs_stop
            else "— fewer than the sequence needed, which is the price of anytime validity: the "
            "mixture boundary is wider than a fixed-n bound that earned its narrowness by "
            "committing to a sample size in advance."
        )
        + (
            "\n\nOne detail deserves stating rather than smoothing over: by the end of the "
            f"stream the interval has drifted back to [{study.cs_lower:+.5f}, "
            f"{study.cs_upper:+.5f}], which contains zero again. A confidence sequence is not "
            "monotone — the interval narrows, but the running mean keeps moving, and here it "
            "moves because the stream is ordered by capture day and the later day is a different "
            "distribution. The guarantee still holds and is not weakened by this: it attaches to "
            "the **stopped decision**, so a team that stopped at the crossing made a valid call "
            "at the stated error rate. What the re-widening says is something else, and "
            "operationally more useful — the advantage was real on the traffic seen up to that "
            "point and did not persist, which is a drift signal about the models, not a defect "
            "in the test. Pairing this with the [exchangeability martingale](exchangeability.md) "
            "is the natural response: one decides which model is better, the other notices when "
            "the question has changed."
            if study.cs_reopened
            else ""
        )
    )


def _render(study: SequentialABStudy, fig: Path) -> str:
    return f"""# NetSentry — When Can the Shadow Model Be Promoted?

_Synthetic stand-in. Honest temporal/binary split, {study.n_stream:,} flows in deployment
order (not shuffled — a shadow test watches traffic as it arrives). Champion and challenger
score every flow, so the comparison is **paired**, which is the entire statistical advantage
of running a shadow rather than splitting traffic. Loss is per-flow Brier score: proper,
bounded, and therefore honest about its own variance. Error level
{study.alpha:.0%}, mixture parameter rho = {study.rho:g}._

## Why this report exists

The serving stack already scores a shadow challenger silently, and the
[promotion](promotion.md) study compares two models with a paired bootstrap. Neither answers
the operational question: **when do you stop watching?** In practice someone checks the
dashboard each morning, sees the challenger ahead, and promotes. That habit invalidates the
test being consulted — a fixed-sample procedure earns its error rate by being evaluated once,
at a sample size fixed in advance, and evaluating it repeatedly turns a 5% guarantee into
something much worse.

A **confidence sequence** is the interval that survives this: valid simultaneously at every
sample size, so an operator may look as often as they like and stop whenever they like with
the stated coverage intact. This uses Robbins' normal mixture (1970), the canonical
construction and the foundation of the modern treatment in Howard, Ramdas, McAuliffe & Sekhon
(2021).

## Three procedures, one stream

{_regimes_table(study)}

{_peeking_read(study)}

## The decision

![anytime-valid interval](../figures/{fig.name})

{_decision_read(study)}

## What anytime validity costs

Nothing here is free. The mixture boundary is strictly wider than a fixed-n interval at the
sample size that fixed-n design committed to, because it must hold at *every* sample size
simultaneously — that width is the premium paid for the right to stop whenever the evidence
justifies it. The mixing parameter `rho` decides where the premium is cheapest: a small
`rho` tightens the boundary early (good for catching a large effect fast), a large one
tightens it late (good for resolving a small effect eventually), and no choice is uniformly
best. The practical reading is that a confidence sequence is the right default for a
*monitoring* process — a shadow model that runs indefinitely and might be promoted at any
time — while a fixed-n design remains more efficient for a genuine one-shot experiment where
the sample size can honestly be fixed in advance and honoured.

## Scope

The null used to measure the peeking inflation is a synthetic mean-zero stream rather than
two real equivalent models, because a real pair is never *exactly* equivalent and the
measurement needs a true null to be meaningful; the inflation it demonstrates is a property
of the procedure, not of these models. The confidence sequence assumes the per-flow
differences are independent with a bounded variance proxy — network flows arrive in bursts
and correlated bursts inflate the effective sample size, so a production deployment should
either thin the stream or use a bound that tolerates dependence. The comparison is on loss,
not on the operational metric: a challenger can win on Brier score and still lose on
detection at a fixed FPR, which is why [promotion](promotion.md) gates on the operational
number and this report answers only the question of *when there is enough evidence to
decide*. And a shadow test measures the challenger on the champion's traffic; a challenger
that would change what gets blocked, and therefore what traffic is subsequently seen, needs
an interleaved design this does not model."""
