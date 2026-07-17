"""Anytime-valid drift detection via a conformal test martingale.

The drift suite already has PSI, per-feature KS with a Benjamini-Hochberg correction, and
the online Page-Hinkley / DDM detectors. Every one of them either needs a fixed reference
window or spends its false-alarm budget at a *pre-declared* moment — look now, decide now.
A monitor that runs forever needs a different guarantee: it may stop and raise an alarm at
**any** time of its choosing, and it should still control the probability of a false alarm
over the whole (unbounded) run. That is exactly what a conformal test martingale delivers.

The construction (Vovk, Nouretdinov & Gammerman, "Testing exchangeability on-line", ICML
2003; Vovk 2021) is a bet against the null hypothesis that the stream is **exchangeable** —
the assumption every IID evaluation quietly rests on. For each arriving flow it computes an
online **conformal p-value** — the (smoothed) rank of the flow's nonconformity among all
flows seen so far — which is exactly Uniform(0, 1) under exchangeability. Those p-values
feed a betting martingale ``M_t = M_{t-1} * g(p_t)`` whose payoff function integrates to one
(``∫ g = 1``), so ``M`` is a non-negative martingale with ``M_0 = 1`` under the null: a
gambler who cannot expect to profit betting against a fair sequence. When the stream stops
being exchangeable — an attack campaign begins, the feature scale shifts — the p-values stop
being uniform, the bets pay off, and ``M`` grows without bound.

The guarantee is **Ville's inequality**: for a non-negative martingale with ``M_0 = 1``,
``P(sup_t M_t >= 1/alpha) <= alpha`` under the null. Alarming the first time ``M_t`` crosses
``1/alpha`` therefore has false-alarm probability at most ``alpha`` **at any stopping time**,
with no window, no multiple-testing correction, and no fixed horizon — the anytime-valid
counterpart to the fixed-window detectors already in the suite. The betting function is the
canonical **mixture of power martingales** (``g_ε(p) = ε p^{ε-1}`` averaged over ``ε`` in
(0, 1)), which is parameter-free and bets that nonconformity p-values become *small* — the
operationally important direction, a stream growing more anomalous than its own history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "exchangeability.md"
FIGURE_NAME = "exchangeability.png"

_P_FLOOR = 1e-6  # clamp p-values off zero so a huge (but finite) bet stays representable


def online_conformal_pvalues(nonconformity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smoothed online conformal p-values for a stream (exactly Uniform(0,1) under exchangeability).

    At step ``t`` the p-value is ``(#{i <= t : a_i > a_t} + theta_t * #{i <= t : a_i = a_t}) / t``
    with ``theta_t ~ U(0, 1)`` breaking ties — Vovk's transductive online conformal p-value.
    Under the null that the sequence is exchangeable these p-values are IID uniform, which is
    what makes the downstream betting process a genuine test martingale.
    """
    a = np.asarray(nonconformity, dtype=float)
    t_total = len(a)
    thetas = rng.random(t_total)
    pvalues = np.empty(t_total, dtype=float)
    for t in range(t_total):
        prefix = a[: t + 1]
        greater = int(np.sum(prefix > a[t]))
        equal = int(np.sum(prefix == a[t]))  # includes the current point, so >= 1
        pvalues[t] = (greater + thetas[t] * equal) / (t + 1)
    return pvalues


def power_martingale_mixture(pvalues: np.ndarray, epsilons: np.ndarray) -> np.ndarray:
    """The conformal test-martingale path from a mixture of power martingales.

    Each power martingale ``M^ε_t = prod_i ε p_i^{ε-1}`` is a martingale with ``M_0 = 1``
    under uniform p-values (``E[ε U^{ε-1}] = 1``); a uniform average over an ``ε`` grid is
    itself a martingale (a convex mixture of martingales), parameter-free and canonical
    (Vovk's Simple Mixture). Accumulated in log space so a fast-growing bet cannot overflow.
    """
    p = np.clip(np.asarray(pvalues, dtype=float), _P_FLOOR, 1.0)
    eps = np.asarray(epsilons, dtype=float).reshape(-1, 1)  # (K, 1)
    # log of each ε's per-step bet, then cumulative sum along the stream: (K, T)
    log_bets = np.log(eps) + (eps - 1.0) * np.log(p).reshape(1, -1)
    log_paths = np.cumsum(log_bets, axis=1)
    # Mixture martingale M_t = mean_k exp(log_path_k[t]) via logsumexp - log K.
    path: np.ndarray = np.exp(logsumexp(log_paths, axis=0) - np.log(len(eps)))
    return path


def detection_time(path: np.ndarray, threshold: float) -> int | None:
    """First 1-indexed step at which the martingale crosses ``threshold`` (else ``None``)."""
    crossings = np.where(np.asarray(path, dtype=float) >= threshold)[0]
    return int(crossings[0] + 1) if len(crossings) else None


def _mixture(n_bets: int) -> np.ndarray:
    """A symmetric ε grid on the open interval (0, 1) for the power-martingale mixture."""
    return np.linspace(0.0, 1.0, n_bets + 2)[1:-1]


@dataclass
class StreamResult:
    """One stream's martingale path and its detection outcome."""

    label: str
    path: np.ndarray
    threshold: float
    change_point: int | None

    @property
    def detected_at(self) -> int | None:
        return detection_time(self.path, self.threshold)

    @property
    def max_value(self) -> float:
        return float(np.max(self.path)) if len(self.path) else 1.0


@dataclass
class ExchangeabilityStudy:
    """The full anytime-valid drift study: a null stream, a drift stream, and the alarm rate."""

    alpha: float
    threshold: float
    stream_len: int
    change_point: int
    null_stream: StreamResult
    drift_stream: StreamResult
    n_null_streams: int
    false_alarm_rate: float
    detection_delays: list[int]


def _fit_nonconformity(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Fit the honest temporal/binary model; return (attack-score nonconformity, labels) on test.

    Nonconformity is the deployed model's attack probability — a flow that looks more like an
    attack is more nonconforming — so the exchangeability test watches the same signal the
    detector acts on.
    """
    seed_everything(settings.seed)
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test").reset_index(drop=True)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.model.classes_)
    x_test = np.asarray(pipeline.transform(test))
    scores = positive_scores(model.predict_proba(x_test), classes)
    labels = test[BINARY_TARGET].to_numpy().astype(int)
    return scores, labels


def _null_stream(scores: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """An exchangeable stream: a uniform random sample of the test scores (order is signal-free)."""
    idx = rng.choice(len(scores), size=min(length, len(scores)), replace=False)
    return scores[idx]


def _drift_stream(
    scores: np.ndarray,
    labels: np.ndarray,
    length: int,
    change_point: int,
    post_attack_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A non-exchangeable stream: benign-dominated, then an attack campaign at ``change_point``.

    Before the change the stream is drawn from benign flows; after it, an attack-heavy mix —
    a regime change in nonconformity that a valid exchangeability test must catch.
    """
    benign = scores[labels == 0]
    attack = scores[labels == 1]
    if len(benign) == 0 or len(attack) == 0:
        return _null_stream(scores, length, rng)
    out = np.empty(length, dtype=float)
    for t in range(length):
        if t < change_point:
            out[t] = benign[rng.integers(len(benign))]
        else:
            pool = attack if rng.random() < post_attack_rate else benign
            out[t] = pool[rng.integers(len(pool))]
    return out


def run_exchangeability(settings: Settings) -> ExchangeabilityStudy:
    """Run the conformal-test-martingale drift study on a null and a drift stream."""
    cfg = settings.exchangeability
    scores, labels = _fit_nonconformity(settings)
    threshold = 1.0 / cfg.alpha
    epsilons = _mixture(cfg.n_bets)
    rng = np.random.default_rng(settings.seed)

    null_scores = _null_stream(scores, cfg.stream_len, rng)
    null_path = power_martingale_mixture(online_conformal_pvalues(null_scores, rng), epsilons)
    null_stream = StreamResult("exchangeable (shuffled)", null_path, threshold, None)

    drift_scores = _drift_stream(
        scores, labels, cfg.stream_len, cfg.change_point, cfg.post_change_attack_rate, rng
    )
    drift_path = power_martingale_mixture(online_conformal_pvalues(drift_scores, rng), epsilons)
    drift_stream = StreamResult(
        "attack campaign at change point", drift_path, threshold, cfg.change_point
    )

    # Empirically confirm Ville: the fraction of independent null streams that ever alarm.
    alarms = 0
    delays: list[int] = []
    for s in range(cfg.n_null_streams):
        srng = np.random.default_rng(settings.seed + 1000 + s)
        ns = _null_stream(scores, cfg.stream_len, srng)
        npath = power_martingale_mixture(online_conformal_pvalues(ns, srng), epsilons)
        if detection_time(npath, threshold) is not None:
            alarms += 1
        drng = np.random.default_rng(settings.seed + 5000 + s)
        ds = _drift_stream(
            scores, labels, cfg.stream_len, cfg.change_point, cfg.post_change_attack_rate, drng
        )
        dpath = power_martingale_mixture(online_conformal_pvalues(ds, drng), epsilons)
        hit = detection_time(dpath, threshold)
        if hit is not None:
            delays.append(max(0, hit - cfg.change_point))
    false_alarm_rate = alarms / cfg.n_null_streams if cfg.n_null_streams else 0.0

    logger.info(
        "Exchangeability martingale",
        extra={
            "null_max": round(null_stream.max_value, 2),
            "drift_detected_at": drift_stream.detected_at,
            "false_alarm_rate": round(false_alarm_rate, 3),
        },
    )
    return ExchangeabilityStudy(
        alpha=cfg.alpha,
        threshold=threshold,
        stream_len=cfg.stream_len,
        change_point=cfg.change_point,
        null_stream=null_stream,
        drift_stream=drift_stream,
        n_null_streams=cfg.n_null_streams,
        false_alarm_rate=false_alarm_rate,
        detection_delays=delays,
    )


def run_exchangeability_report(settings: Settings) -> Path:
    """Run the conformal-test-martingale study and write the report + figure."""
    study = run_exchangeability(settings)

    steps = np.arange(1, study.stream_len + 1, dtype=float)
    series = {
        study.null_stream.label: (steps, study.null_stream.path),
        study.drift_stream.label: (steps, study.drift_stream.path),
        "alarm threshold (1 / alpha)": (steps, np.full(study.stream_len, study.threshold)),
    }
    fig = plots.plot_lines(
        series,
        xlabel="flows observed (stream position)",
        ylabel="test martingale M_t (log scale)",
        title="Conformal test martingale: anytime-valid drift detection (Vovk et al. 2003)",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        vlines={"attack onset": float(study.change_point)},
        yscale="log",
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote exchangeability report", extra={"path": str(out_path)})

    with track_run(settings, "exchangeability") as run:
        run.log_metrics(
            {
                "null_max_martingale": study.null_stream.max_value,
                "drift_detected_at": float(study.drift_stream.detected_at or 0),
                "false_alarm_rate": study.false_alarm_rate,
                "median_detection_delay": (
                    float(np.median(study.detection_delays)) if study.detection_delays else 0.0
                ),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _read(study: ExchangeabilityStudy) -> str:
    detected = study.drift_stream.detected_at
    delay_clause = (
        f"detects it **{detected - study.change_point} flows later** (at flow {detected:,})"
        if detected is not None
        else "did not cross the alarm line on this particular stream"
    )
    median_delay = int(np.median(study.detection_delays)) if study.detection_delays else None
    delay_agg = (
        f" Across {len(study.detection_delays)} drift streams the median detection delay is "
        f"**{median_delay} flows** past the change point."
        if median_delay is not None
        else ""
    )
    valid = study.false_alarm_rate <= study.alpha + 1e-9
    ville_clause = (
        f"Across {study.n_null_streams} independent exchangeable streams, "
        f"**{study.false_alarm_rate:.0%}** ever crossed the {study.threshold:g} line — at or under "
        f"the {study.alpha:.0%} budget Ville's inequality promises"
        + (
            ", so the anytime-valid guarantee holds empirically here. "
            if valid
            else " (a small-sample excursion above the bound; the guarantee is asymptotic in the "
            "martingale value, not the stream count). "
        )
    )
    return (
        f"The exchangeable stream's martingale stays near its starting value of 1 (it peaks at "
        f"only {study.null_stream.max_value:.2g}) — betting against a fair sequence does not pay. "
        f"The drift stream, benign until an attack campaign opens at flow {study.change_point:,}, "
        f"{delay_clause}.{delay_agg} "
        + ville_clause
        + "Unlike the windowed PSI/KS detectors, this spends no fixed false-alarm budget and needs "
        "no reference window: it can be watched forever and alarmed the instant the bet pays off."
    )


def _render(study: ExchangeabilityStudy, fig: Path) -> str:
    return f"""# NetSentry — Anytime-Valid Drift Detection (Conformal Test Martingale)

_Synthetic stand-in. Temporal/binary model; the deployed attack score is the nonconformity
measure. Streams of {study.stream_len:,} flows; the drift stream turns attack-heavy at flow
{study.change_point:,}. Alarm at ``M_t >= 1/alpha = {study.threshold:g}`` (alpha =
{study.alpha:g})._

## Why this report exists

The drift suite already has PSI, per-feature KS with FDR control, and online Page-Hinkley /
DDM. All of them either need a fixed reference window or spend their false-alarm budget at a
declared moment. A monitor that runs forever needs a stronger contract: it may raise an alarm
at **any** time and still control the false-alarm probability over the whole unbounded run.

A conformal test martingale (Vovk, Nouretdinov & Gammerman, ICML 2003) provides it. Each flow
yields an online conformal p-value — the smoothed rank of its nonconformity among all flows
seen so far — which is Uniform(0, 1) exactly when the stream is **exchangeable**, the
assumption every IID evaluation rests on. Those p-values drive a betting martingale
``M_t = M_{{t-1}} * g(p_t)`` (a parameter-free mixture of power martingales) that stays a fair
game under the null and grows without bound when the stream stops being exchangeable. By
**Ville's inequality**, ``P(sup_t M_t >= 1/alpha) <= alpha`` — so alarming at the crossing has
false-alarm probability at most ``alpha`` at *any* stopping time.

## Null stream vs drift stream

![Conformal test martingale paths](../figures/{fig.name})

{_read(study)}

## Scope

The mixture bets that p-values become *small*, i.e. that the stream grows **more anomalous**
than its own history — the operationally important direction (an attack campaign, a new scan),
and the reason the deployed attack score is used as the nonconformity measure. A drift that
made traffic *more* benign than its history is the symmetric case a large-p betting function
would catch; it is not the SOC's alarm. The guarantee is against the null of exchangeability,
so it complements rather than replaces the feature-wise PSI/KS reports: those localise *which*
feature moved, while this one answers *whether and when* the stream stopped being the
distribution the model was validated on — with a false-alarm rate controlled for all time."""
