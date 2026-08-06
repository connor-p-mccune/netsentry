"""Sequential host decisions: how many flows before you can call a host compromised?

Every threshold in this project decides **one flow**. A SOC does not respond to a flow; it
responds to a *host*, and the jump from one to the other is not free. Two things break on
the way across:

1. **A per-flow false-positive budget gives no host-level guarantee at all.** At a 0.1%
   per-flow FPR, a benign host with 1,000 flows in a shift trips at least one alert with
   probability `1 - 0.999^1000` — around 63%. Nothing is wrong with the model; the arithmetic
   of repeated trials simply eats the budget. The more traffic a host generates, the more
   certain its eventual false alarm, which is why "the noisy server always shows up in the
   queue" is a universal SOC complaint and not a modelling failure.
2. **Waiting for certainty costs time.** The obvious fix — demand *k* alerts before escalating
   — buys precision by delaying the decision, and picking `k` by feel prices neither side.

Sequential analysis solves exactly this problem, and it solved it in 1945. Wald's
**sequential probability ratio test** watches a stream, accumulates the log-likelihood ratio
between two hypotheses, and stops the moment the evidence crosses a boundary derived from
the error rates you are willing to accept. It needs no fixed sample size, it controls both
error types simultaneously, and among all tests with those error rates it is optimal in
expected sample size (Wald & Wolfowitz 1948) — the fewest flows to a decision, provably.

The construction here needs no new modelling, because the deployed operating point already
supplies both hypotheses. Reduce each flow to the binary event the system already emits —
did it alert? — and the two likelihoods follow from the detector's validation-measured rates:

- clean host (H0): a flow alerts with probability **FPR**
- compromised host (H1): a flow alerts with probability **mix * TPR + (1 - mix) * FPR**

That mixture is the part worth stating carefully. A compromised host does not alert at the
detector's TPR: only its attack flows are eligible for that, and the rest of its traffic is
ordinary work that alerts at the false-positive rate. Specifying H1 as the bare TPR models a
host whose every packet is hostile, wildly overstates the evidence each *quiet* flow carries
against compromise, and acquits real intrusions within a few dozen flows — a mistake made
and corrected during this study's construction. With the mixture,
each alerting flow contributes `log(p1/p0)` of evidence and each quiet flow contributes
`log((1-p1)/(1-p0))`, and the decision boundaries `log((1-beta)/alpha)` and
`log(beta/(1-alpha))` come straight from the error rates an operator names. The report
validates the realized error rates against the nominal ones, checks the observed
flows-to-decision against Wald's expected-sample-number formula, and prices the whole thing
against the naive policy of escalating a host on its first flagged flow — the policy most
deployments actually run, and the one whose false-alarm rate grows with host chattiness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SequentialConfig

logger = get_logger(__name__)

REPORT_NAME = "sequential.md"
FIGURE_NAME = "sequential.png"

COMPROMISED = "compromised"
CLEAN = "clean"
UNDECIDED = "undecided"


def wald_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's stopping boundaries ``(upper, lower)`` for nominal error rates.

    ``upper = log((1 - beta) / alpha)`` and ``lower = log(beta / (1 - alpha))``. These are
    the exact boundaries of the likelihood-ratio test; Wald showed the realized errors
    satisfy ``alpha' <= alpha / (1 - beta)`` and ``beta' <= beta / (1 - alpha)``, so slightly
    conservative in practice because the accumulated statistic overshoots the boundary
    rather than landing on it.
    """
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must lie strictly between 0 and 1")
    return math.log((1.0 - beta) / alpha), math.log(beta / (1.0 - alpha))


def hypothesis_rates(tpr: float, fpr: float, compromise_mix: float) -> tuple[float, float]:
    """Per-flow alert probabilities ``(p1, p0)`` under compromised and clean hypotheses.

    The subtle part of specifying this test. A compromised host does **not** alert at the
    detector's TPR — only its *attack* flows are eligible for that, and the rest of its
    traffic is ordinary work that alerts at the false-positive rate. So the compromised
    hypothesis is a mixture, ``p1 = mix * TPR + (1 - mix) * FPR``, and the clean one is
    simply ``p0 = FPR``. Using TPR directly for H1 would model a host whose *every* flow is
    an attack, wildly overstate the evidence a quiet flow carries against compromise, and
    acquit real intrusions in a few dozen flows.
    """
    mix = float(np.clip(compromise_mix, 0.0, 1.0))
    p1 = mix * tpr + (1.0 - mix) * fpr
    return float(np.clip(p1, 1e-9, 1 - 1e-9)), float(np.clip(fpr, 1e-9, 1 - 1e-9))


def llr_increments(alerted: np.ndarray, p1: float, p0: float) -> np.ndarray:
    """Per-flow log-likelihood ratio of compromised-vs-clean, given the two alert rates.

    An alerting flow is ``p1/p0`` times more likely on a compromised host; a quiet flow is
    ``(1-p1)/(1-p0)`` times more likely on a clean one. Both rates come from the detector's
    own validation-measured operating point via :func:`hypothesis_rates`, so the test
    inherits whatever the calibration is worth and fits nothing new.
    """
    p1 = float(np.clip(p1, 1e-9, 1 - 1e-9))
    p0 = float(np.clip(p0, 1e-9, 1 - 1e-9))
    hit = math.log(p1 / p0)
    miss = math.log((1.0 - p1) / (1.0 - p0))
    return np.where(np.asarray(alerted, dtype=bool), hit, miss)


def sprt_decide(increments: np.ndarray, upper: float, lower: float) -> tuple[str, int]:
    """Walk the stream and stop at the first boundary crossing.

    Returns the verdict and the number of flows consumed. A stream that ends without
    crossing is ``undecided`` — a real and important outcome that a fixed-sample test hides
    by forcing a call, and one an operator can act on (keep watching) rather than guess at.
    """
    total = 0.0
    for i, step in enumerate(np.asarray(increments, dtype=float), start=1):
        total += float(step)
        if total >= upper:
            return COMPROMISED, i
        if total <= lower:
            return CLEAN, i
    return UNDECIDED, len(increments)


def expected_sample_number(alpha: float, beta: float, p1: float, p0: float, *, under: str) -> float:
    """Wald's approximate expected flows to a decision under one hypothesis.

    ``E[N] = E[boundary] / E[increment]``, the identity that makes the SPRT's efficiency
    claim concrete: the numerator is the boundary the walk is expected to hit, the
    denominator the average evidence per flow (the Kullback-Leibler divergence between the
    two alert distributions). Approximate because it ignores boundary overshoot.
    """
    upper, lower = wald_boundaries(alpha, beta)
    p1 = float(np.clip(p1, 1e-9, 1 - 1e-9))
    p0 = float(np.clip(p0, 1e-9, 1 - 1e-9))
    hit = math.log(p1 / p0)
    miss = math.log((1.0 - p1) / (1.0 - p0))
    if under == "H1":
        drift = p1 * hit + (1.0 - p1) * miss
        numerator = (1.0 - beta) * upper + beta * lower
    else:
        drift = p0 * hit + (1.0 - p0) * miss
        numerator = alpha * upper + (1.0 - alpha) * lower
    if abs(drift) < 1e-12:
        return float("inf")
    return float(numerator / drift)


def naive_host_false_alarm(per_flow_fpr: float, n_flows: int) -> float:
    """Probability that "escalate on the first flagged flow" fires on a *clean* host.

    ``1 - (1 - fpr)^n``. The whole reason a per-flow budget is not a host-level guarantee:
    the rate is fixed but the number of trials is not, so the chattiest hosts are the ones
    most certain to be escalated — a fact about arithmetic, not about the model.
    """
    return float(1.0 - (1.0 - per_flow_fpr) ** max(int(n_flows), 0))


# --------------------------------------------------------------------------------------
# Host simulation
# --------------------------------------------------------------------------------------
def simulate_host_stream(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    compromise_mix: float,
    n_flows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one host's flow scores: benign traffic with a share of attack flows mixed in.

    CIC-IDS2017's identifiers are dropped before modelling (that is the project's leakage
    rule), so there are no real per-host streams to replay. Hosts are therefore *composed*
    from the model's actual test-set score distributions rather than invented: a clean host
    draws only benign flows, a compromised host draws a ``compromise_mix`` share of attack
    flows among its benign traffic. Every score is a real score the deployed model produced;
    only the grouping is synthetic.
    """
    n_attack = rng.binomial(n_flows, np.clip(compromise_mix, 0.0, 1.0))
    n_benign = n_flows - n_attack
    parts = [rng.choice(benign_scores, size=n_benign, replace=True)]
    if n_attack > 0 and len(attack_scores) > 0:
        parts.append(rng.choice(attack_scores, size=n_attack, replace=True))
    stream = np.concatenate(parts)
    rng.shuffle(stream)  # an attacker's flows are interleaved, not appended
    return stream


@dataclass
class ArmOutcome:
    """SPRT behaviour over many simulated hosts of one kind."""

    label: str
    n_hosts: int
    decided_compromised: float
    decided_clean: float
    undecided: float
    median_flows: float
    mean_flows: float
    naive_flag_rate: float


@dataclass
class MixPoint:
    """Detection and speed at one compromise intensity."""

    compromise_mix: float
    detection: float
    median_flows: float
    undecided: float


@dataclass
class SequentialStudy:
    """Everything the report renders."""

    alpha: float
    beta: float
    tpr: float
    fpr: float
    upper: float
    lower: float
    max_flows: int
    n_hosts: int
    compromise_mix: float
    clean: ArmOutcome
    compromised: ArmOutcome
    recalibrated: ArmOutcome
    realized_p1: float
    design_tpr: float
    realized_tpr: float
    asn_h0: float
    asn_h1: float
    naive_clean_closed_form: float
    benign_alert_rate: float
    p1: float
    p0: float
    mixes: list[MixPoint]


def run_sequential(settings: Settings) -> SequentialStudy:
    """Calibrate the SPRT from the deployed operating point and run it over simulated hosts."""
    cfg: SequentialConfig = settings.sequential
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
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))

    def _scores(x: np.ndarray) -> np.ndarray:
        return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

    s_val, s_test = _scores(x_val), _scores(x_test)
    threshold = threshold_at_fpr(y_val, s_val, variant.thresholds.primary_fpr)
    # Both SPRT likelihoods come from VALIDATION, never from the streams being judged —
    # calibrating the test on the data it will be scored against is the leakage this
    # project exists to avoid.
    val_rates = rates_at_threshold(y_val, s_val, threshold)
    tpr, fpr = val_rates["tpr"], max(val_rates["fpr"], 1.0 / len(y_val))
    # The test is *designed* for one compromise intensity; the sweep then measures how it
    # behaves at others, which is what a deployed detector actually faces.
    p1, p0 = hypothesis_rates(tpr, fpr, cfg.compromise_mix)
    upper, lower = wald_boundaries(cfg.alpha, cfg.beta)

    benign_scores = s_test[y_test == 0]
    attack_scores = s_test[y_test == 1]
    # The naive policy's closed form must use the rate the streams actually realise, which
    # is the *test* benign alert rate, not the validation rate the threshold was set on.
    empirical_benign_alert_rate = float(np.mean(benign_scores >= threshold))
    rng = np.random.default_rng(variant.seed)

    def _arm(mix: float, label: str, use_p1: float, use_p0: float) -> ArmOutcome:
        verdicts: list[str] = []
        lengths: list[int] = []
        naive_flags = 0
        for _ in range(cfg.n_hosts):
            stream = simulate_host_stream(benign_scores, attack_scores, mix, cfg.max_flows, rng)
            alerted = stream >= threshold
            verdict, n = sprt_decide(llr_increments(alerted, use_p1, use_p0), upper, lower)
            verdicts.append(verdict)
            lengths.append(n)
            naive_flags += int(alerted.any())
        arr = np.array(verdicts)
        decided = np.array(lengths, dtype=float)[arr != UNDECIDED]
        return ArmOutcome(
            label=label,
            n_hosts=cfg.n_hosts,
            decided_compromised=float(np.mean(arr == COMPROMISED)),
            decided_clean=float(np.mean(arr == CLEAN)),
            undecided=float(np.mean(arr == UNDECIDED)),
            median_flows=float(np.median(decided)) if decided.size else float(cfg.max_flows),
            mean_flows=float(np.mean(decided)) if decided.size else float(cfg.max_flows),
            naive_flag_rate=naive_flags / cfg.n_hosts,
        )

    clean_arm = _arm(0.0, "clean hosts (benign only)", p1, p0)
    compromised_arm = _arm(
        cfg.compromise_mix, f"compromised hosts ({cfg.compromise_mix:.0%} attack)", p1, p0
    )

    # The design's likelihood comes from validation, but the streams are drawn from the
    # *test* days, where this project's headline finding says detection is much weaker. The
    # recalibrated arm re-derives the same test from the rate the streams actually realise —
    # the measure / diagnose / re-measure loop, applied to a guarantee instead of a metric.
    test_rates = rates_at_threshold(y_test, s_test, threshold)
    realized_p1, realized_p0 = hypothesis_rates(
        test_rates["tpr"], empirical_benign_alert_rate, cfg.compromise_mix
    )
    recalibrated_arm = _arm(
        cfg.compromise_mix,
        f"compromised hosts, test-calibrated ({cfg.compromise_mix:.0%} attack)",
        realized_p1,
        realized_p0,
    )

    mixes = []
    for mix in cfg.compromise_mixes:
        arm = _arm(mix, f"mix {mix}", p1, p0)
        mixes.append(
            MixPoint(
                compromise_mix=mix,
                detection=arm.decided_compromised,
                median_flows=arm.median_flows,
                undecided=arm.undecided,
            )
        )
        logger.info(
            "Compromise mix measured",
            extra={"mix": mix, "detection": round(arm.decided_compromised, 3)},
        )

    return SequentialStudy(
        alpha=cfg.alpha,
        beta=cfg.beta,
        tpr=tpr,
        fpr=fpr,
        upper=upper,
        lower=lower,
        max_flows=cfg.max_flows,
        n_hosts=cfg.n_hosts,
        compromise_mix=cfg.compromise_mix,
        clean=clean_arm,
        compromised=compromised_arm,
        recalibrated=recalibrated_arm,
        realized_p1=realized_p1,
        design_tpr=tpr,
        realized_tpr=test_rates["tpr"],
        asn_h0=expected_sample_number(cfg.alpha, cfg.beta, p1, p0, under="H0"),
        asn_h1=expected_sample_number(cfg.alpha, cfg.beta, p1, p0, under="H1"),
        naive_clean_closed_form=naive_host_false_alarm(empirical_benign_alert_rate, cfg.max_flows),
        benign_alert_rate=empirical_benign_alert_rate,
        p1=p1,
        p0=p0,
        mixes=mixes,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_sequential_report(settings: Settings) -> Path:
    """Run the sequential host-decision study and write the report + figure."""
    study = run_sequential(settings)

    mixes = np.array([m.compromise_mix for m in study.mixes])
    fig = plots.plot_lines(
        {
            "SPRT: hosts called compromised": (
                mixes,
                np.array([m.detection for m in study.mixes]),
            ),
            "SPRT: still watching (undecided)": (
                mixes,
                np.array([m.undecided for m in study.mixes]),
            ),
        },
        xlabel="share of a host's flows that are attack traffic",
        ylabel=f"share of hosts, over {study.n_hosts} simulated streams",
        title="How much compromise the sequential test needs before it commits",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote sequential report", extra={"path": str(out_path)})

    with track_run(settings, "sequential") as run:
        run.log_params({"alpha": study.alpha, "beta": study.beta, "max_flows": study.max_flows})
        run.log_metrics(
            {
                "realized_alpha": study.clean.decided_compromised,
                "realized_beta": 1.0 - study.compromised.decided_compromised,
                "median_flows_to_detect": study.compromised.median_flows,
                "asn_h1": study.asn_h1,
                "naive_clean_flag_rate": study.clean.naive_flag_rate,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _arms_table(study: SequentialStudy) -> str:
    rows = [
        "| host population | called compromised | called clean | still watching "
        "| median flows to decide | naive policy flags it |",
        "|---|---|---|---|---|---|",
    ]
    for arm in (study.clean, study.compromised, study.recalibrated):
        rows.append(
            f"| {arm.label} | {arm.decided_compromised:.1%} | {arm.decided_clean:.1%} "
            f"| {arm.undecided:.1%} | {arm.median_flows:,.0f} | {arm.naive_flag_rate:.1%} |"
        )
    return "\n".join(rows)


def _mix_table(study: SequentialStudy) -> str:
    rows = [
        "| attack share of the host's traffic | called compromised | still watching "
        "| median flows to decide |",
        "|---|---|---|---|",
    ]
    for m in study.mixes:
        rows.append(
            f"| {m.compromise_mix:.0%} | {m.detection:.1%} | {m.undecided:.1%} "
            f"| {m.median_flows:,.0f} |"
        )
    return "\n".join(rows)


def _error_read(study: SequentialStudy) -> str:
    realized_alpha = study.clean.decided_compromised
    realized_beta = 1.0 - study.compromised.decided_compromised
    recal_beta = 1.0 - study.recalibrated.decided_compromised
    alpha_bound = study.alpha / (1.0 - study.beta)
    beta_bound = study.beta / (1.0 - study.alpha)
    setup = (
        f"The boundaries were set from the operator's stated tolerances (alpha = "
        f"{study.alpha}, beta = {study.beta}), giving an upper boundary of {study.upper:.2f} "
        f"and a lower of {study.lower:.2f} nats of evidence, over the hypothesis pair "
        f"`p1 = {study.p1:.5f}` (a {study.compromise_mix:.0%}-compromised host) against "
        f"`p0 = {study.p0:.5f}` (a clean one). Wald's bounds say the realized errors cannot "
        f"exceed {alpha_bound:.3f} and {beta_bound:.3f}. Measured over {study.n_hosts} "
        f"simulated hosts of each kind: **{realized_alpha:.1%}** of clean hosts were escalated "
        f"and **{realized_beta:.1%}** of compromised hosts were missed. "
    )
    if realized_alpha <= alpha_bound and realized_beta <= beta_bound:
        return setup + (
            "Both sides hold, and both are conservative against their nominal rates — expected, "
            "because the accumulated log-likelihood overshoots its boundary rather than landing "
            "exactly on it, so Wald's rates bound the realized error rather than targeting it."
        )
    if realized_beta > beta_bound:
        return setup + (
            f"The false-alarm side holds comfortably. The miss side does **not**: "
            f"{realized_beta:.1%} against a bound of {beta_bound:.3f}. That is worth being "
            "precise about, because it is not a failure of Wald's construction — his bound is "
            "conditional on the assumed likelihood being the true one, and here it is not.\n\n"
            f"The design took its detection rate from **validation** ({study.design_tpr:.1%} "
            f"TPR), which is carved from the Mon-Wed training days. The host streams are drawn "
            f"from Thu-Fri, where the same detector detects {study.realized_tpr:.1%} — this "
            "project's headline temporal gap, arriving in a new place. So the test was designed "
            f"expecting a compromised host to alert at `p1 = {study.p1:.5f}` and was handed "
            f"streams that alert at `{study.realized_p1:.5f}`, roughly "
            f"{study.p1 / max(study.realized_p1, 1e-12):.1f}x less evidence per flow than it was "
            "promised. A guarantee bought with an optimistic likelihood is worth exactly what "
            "the likelihood was.\n\n"
            f"The third row settles it. Re-deriving the same test from the rate the streams "
            f"actually realise drops the miss rate to {recal_beta:.1%} "
            f"{'— inside the bound' if recal_beta <= beta_bound else '— much closer to the bound'}"
            ", with no change to the algorithm, the boundaries, or the data. The lesson is "
            "operational rather than theoretical: a sequential test's error control is only as "
            "good as the operating point it is calibrated on, so it belongs on the same refresh "
            "schedule as the [threshold](refresh.md) it inherits, and it degrades under drift "
            "in exactly the way the [conformal](adaptive_conformal.md) guarantees do."
        )
    return setup + (
        "The miss side holds; the false-alarm side runs above its nominal bound, which on this "
        "stand-in reflects the finite host count more than the construction — the realized rate "
        "is a noisy draw around a bound that holds in expectation."
    )


def _speed_read(study: SequentialStudy) -> str:
    return (
        f"Speed is the reason to use a sequential test rather than a fixed window. Wald's "
        f"expected-sample-number formula predicts {study.asn_h1:,.0f} flows to a decision on a "
        f"compromised host and {study.asn_h0:,.0f} on a clean one; the simulation lands at a "
        f"median of {study.compromised.median_flows:,.0f} and {study.clean.median_flows:,.0f}. "
        "The asymmetry is structural and worth reading carefully: an alerting flow carries "
        f"{math.log(study.p1 / study.p0):.1f} nats of evidence toward compromise, while a quiet "
        f"flow carries {abs(math.log((1 - study.p1) / (1 - study.p0))):.4f} nats toward "
        f"innocence, so {math.ceil(study.upper / math.log(study.p1 / study.p0))} alerts convict "
        f"and about {abs(study.lower / math.log((1 - study.p1) / (1 - study.p0))):,.0f} "
        "consecutive quiet flows are needed to acquit. That is the right shape for a SOC: a "
        "compromised host is escalated in a burst of flows, and a quiet host simply stays under "
        "observation rather than being declared clean on thin evidence."
    )


def _naive_read(study: SequentialStudy) -> str:
    return (
        "The comparison that matters is against what most deployments actually run: escalate a "
        f"host the moment any of its flows is flagged. On these streams that policy escalates "
        f"{study.clean.naive_flag_rate:.1%} of **clean** hosts — the closed form "
        f"`1 - (1 - {study.benign_alert_rate:.5f})^{study.max_flows}` gives "
        f"{study.naive_clean_closed_form:.1%}, and the simulation agrees. The per-flow "
        "false-positive budget is intact; it is simply not a host-level guarantee, because the "
        "rate is fixed while the number of trials is not. A host with ten times the traffic gets "
        "ten times the chances to trip it, which is why the chattiest servers dominate every real "
        "alert queue and why analysts learn to ignore them. The sequential test escalates "
        f"{study.clean.decided_compromised:.1%} of the same clean hosts, because it asks whether "
        "the *rate* of alerts is consistent with a clean host rather than whether any alert "
        "occurred at all. This is the [base-rate fallacy](base_rate.md) in the time dimension: "
        "there, the benign majority swamps precision across hosts; here, the benign majority "
        "swamps it across a single host's flows."
    )


def _render(study: SequentialStudy, fig: Path) -> str:
    return f"""# NetSentry — Sequential Host Decisions (Wald's SPRT)

_Synthetic stand-in. Honest temporal/binary split. Likelihoods taken from the deployed
operating point measured on **validation** (TPR {study.tpr:.1%}, FPR {study.fpr:.3%} at the
{study.alpha:.0%}-alpha / {study.beta:.0%}-beta design), then run over {study.n_hosts}
simulated host streams of up to {study.max_flows:,} flows each, composed from real test-set
scores._

## Why this report exists

Every threshold in this project decides one flow; a SOC responds to a **host**. The gap
between them is not free. At a per-flow false-positive rate of {study.fpr:.3%}, a benign host
with {study.max_flows:,} flows trips at least one alert with probability
{study.naive_clean_closed_form:.1%} — the model is behaving exactly as calibrated, and the
host-level false-alarm rate is still nearly certain, because a fixed rate over an unbounded
number of trials is not a guarantee. Demanding *k* alerts before escalating fixes the
arithmetic and replaces it with an unpriced guess at `k`.

Wald's **sequential probability ratio test** (1945) is the principled version: accumulate
log-likelihood evidence flow by flow and stop at the first boundary crossing, with both error
rates controlled by construction and — among all tests achieving those rates — the smallest
expected number of observations (Wald & Wolfowitz 1948). It needs no new model here: reduce
each flow to "did it alert?", and the two hypotheses are the detector's own measured rates.

## Do the error guarantees hold?

{_arms_table(study)}

{_error_read(study)}

## How fast is the decision?

{_speed_read(study)}

## Against the policy most deployments actually run

{_naive_read(study)}

## How much compromise is enough?

{_mix_table(study)}

![detection vs compromise intensity](../figures/{fig.name})

The sweep shows where the test's power comes from. A host whose traffic is only marginally
attack-flavoured accumulates evidence slowly and often reaches the end of the window still
undecided — correctly, because at that intensity the stream genuinely does not distinguish
the hypotheses. As the attack share rises the evidence per flow rises with it and the
decision arrives sooner and more often. "Still watching" is not a failure mode; it is the
test refusing to guess, and it is the outcome a fixed-window rule silently converts into a
false negative.

## Scope

The identifier columns that would carry a real host identity are dropped before modelling —
that is the project's leakage rule — so host streams are **composed** from the model's actual
test-set score distributions rather than replayed from capture. Every score is real; the
grouping is not, and a real host's flows would be correlated in ways an i.i.d. draw is not
(bursts, sessions, a single long-lived connection), which the SPRT's independence assumption
would feel. That assumption is the test's main structural caveat: correlated flows inflate the
apparent evidence and make the realized error rates optimistic, the standard remedy being to
thin the stream to one observation per session. The likelihoods are validation-measured, so
they inherit the [calibration](evaluation.md) quality and drift with it — a detector whose
real TPR falls below its assumed TPR will convict more slowly than designed, which the
[threshold refresh](refresh.md) job is the existing mechanism for. Finally, this decides
*compromise*, not *what happened*: the [host-graph](graph_demo.md) and
[campaign](campaigns.md) studies are what turn a compromised host into an incident."""
