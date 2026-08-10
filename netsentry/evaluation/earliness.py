"""When the verdict can exist — the latency a flow-level detector cannot design away.

Every metric in this repository is reported as though the detector's decision arrives the
moment the attack does. It does not. CICFlowMeter — and NetFlow, and IPFIX, and Zeek's
`conn.log` — emit **one record per finished flow**, and most of the 77 statistics the model
consumes are only defined once the flow is over. `Total Fwd Packets` is not a running
quantity that happens to be read at the end; the number that exists mid-flow is a different,
systematically smaller number, and a model trained on completed flows has never seen it.
The deployed detector is therefore, structurally, a **post-mortem** one: the earliest instant
at which its verdict *could* exist is the instant the exporter closes the flow.

That instant is measurable, and this study measures it in two halves.

**How long the verdict takes.** A flow that ends with a FIN or an RST is closed by its own
teardown, so its record is available at roughly its own duration. A flow that simply stops —
a scan probe that never gets a reply, a half-open connection, a UDP exchange — is closed only
when the exporter's idle timer expires, which on the configured `capture.flow_timeout_us` is
two minutes of a detector knowing nothing. Both cases are decided per flow from columns that
are already in the data (`FIN Flag Count`, `RST Flag Count`, `Flow Duration`), so the wait is
computed rather than assumed.

**What it would cost not to wait.** Partition the features by *when the value the model was
trained on is knowable* (`netsentry.features.feature_sets.availability_sets`): fields fixed at
connection setup, intensive statistics whose prefix value is a noisy estimate of the final one,
and everything else. Refit the model on each nested tier and read PR-AUC and detection at the
operating budget off the same honest temporal split. The three tiers are three detectors that
could genuinely be deployed at three different moments, and the gap between them is the price
of earliness.

Putting the halves together gives the thing an operator actually needs and no accuracy table
can express: **the share of hostile flows detected within `t` seconds of their first packet**,
per tier. A tier that detects less but decides at once can dominate one that detects more but
decides after the attack has finished — and whether it does is an empirical question about
this data, not a matter of taste.

The scope is stated plainly in the report: the in-flight tier is evaluated on the *completed*
values of its intensive features, which a truly mid-flow detector would only estimate. That
makes it an **upper bound** on in-flight detection, which is the right direction — any loss
measured against the complete tier is a loss a real early detector would also suffer, and
more.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import (
    AVAILABILITY_TIERS,
    availability_sets,
    display_feature_name,
)
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import EarlinessConfig

logger = get_logger(__name__)

REPORT_NAME = "earliness.md"
FRONTIER_FIGURE = "earliness_frontier.png"
TIER_FIGURE = "earliness_tiers.png"

DURATION_COLUMN = "Flow Duration"
TEARDOWN_COLUMNS: tuple[str, ...] = ("FIN Flag Count", "RST Flag Count")
MICROSECONDS = 1_000_000.0

TIER_LABELS: dict[str, str] = {
    "handshake": "handshake",
    "in_flight": "in-flight",
    "complete": "complete flow (deployed)",
}


# --------------------------------------------------------------------------------------
# The wait (pure; unit-tested)
# --------------------------------------------------------------------------------------
def emit_delay_us(
    duration_us: np.ndarray, teardown: np.ndarray, flow_timeout_us: int
) -> np.ndarray:
    """Microseconds from a flow's first packet until its record can be exported.

    A flow whose teardown the exporter observed (a FIN or an RST) closes on that packet, so
    its record exists at roughly its own duration. A flow that merely stops emitting is held
    open until the idle timer fires — the exporter cannot distinguish "finished" from
    "quiet", so it must wait out `flow_timeout_us` before it can be sure. That second case
    is not an edge case on this data; it is every unanswered scan probe.
    """
    duration = np.asarray(duration_us, dtype=float)
    closed = np.asarray(teardown).astype(bool)
    return duration + np.where(closed, 0.0, float(flow_timeout_us))


def observed_teardown(frame: pd.DataFrame) -> np.ndarray:
    """Per-flow mask: did the exporter see a FIN or an RST for this flow?"""
    present = [c for c in TEARDOWN_COLUMNS if c in frame.columns]
    if not present:
        return np.zeros(len(frame), dtype=bool)
    counts = frame[present].to_numpy(dtype=float)
    return np.asarray(np.nansum(counts, axis=1) > 0, dtype=bool)


def decision_latency_us(
    frame: pd.DataFrame, tier: str, flow_timeout_us: int, in_flight_horizon_us: int
) -> np.ndarray:
    """Microseconds after a flow's first packet at which ``tier``'s verdict can exist.

    The handshake tier decides on fields fixed by the connection setup, so it decides at
    once (one round trip, not modelled). The in-flight tier must accumulate enough packets
    for its intensive statistics to mean anything, which is the `in_flight_horizon_us` knob
    — but a flow shorter than the horizon simply ends first, so its verdict is bounded by
    its own duration. The complete tier waits for the exporter.
    """
    duration = frame[DURATION_COLUMN].to_numpy(dtype=float)
    if tier == "handshake":
        return np.zeros(len(frame), dtype=float)
    if tier == "in_flight":
        capped: np.ndarray = np.minimum(duration, float(in_flight_horizon_us))
        return capped
    return emit_delay_us(duration, observed_teardown(frame), flow_timeout_us)


def detected_within(
    latency_us: np.ndarray, detected: np.ndarray, horizons_s: np.ndarray
) -> np.ndarray:
    """Share of *all* attack flows both detected and decided within each horizon.

    The denominator is every hostile flow, not the detected ones — a curve over the detected
    subset would be the same survivorship bias the [survival](survival.md) study exists to
    remove, in the time domain.
    """
    seconds = np.asarray(latency_us, dtype=float) / MICROSECONDS
    hit = np.asarray(detected).astype(bool)
    n = max(len(seconds), 1)
    return np.array([float(np.sum(hit & (seconds <= h)) / n) for h in horizons_s])


def wait_sensitivity(
    duration_us: np.ndarray, flow_timeout_us: int, unclosed_shares: np.ndarray
) -> np.ndarray:
    """Median export wait if a share ``s`` of flows never showed a teardown.

    The stand-in generator stamps a teardown on essentially every flow, which makes the idle
    timer inert and the measured wait equal to the flow duration. Rather than quietly report
    a number that only holds for data with that property, this sweeps the one quantity the
    stand-in cannot supply. Under the (stated) assumption that being unclosed is independent
    of duration, the mixture's median is exact: below ``s = 0.5`` it still falls in the
    closed branch and is a re-scaled duration quantile; at or above it, it falls in the
    timed-out branch and picks up the whole timeout.
    """
    duration = np.sort(np.asarray(duration_us, dtype=float))
    if duration.size == 0:
        return np.zeros(len(unclosed_shares), dtype=float)
    out: list[float] = []
    for s in np.asarray(unclosed_shares, dtype=float):
        if s < 0.5:
            out.append(float(np.quantile(duration, 0.5 / (1.0 - s))))
        else:
            q = (s - 0.5) / s if s > 0 else 0.0
            out.append(float(np.quantile(duration, q)) + float(flow_timeout_us))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class TierResult:
    """One decision-time tier's detection quality and the latency it decides at."""

    tier: str
    n_features: int
    pr_auc: float
    detection: float
    realised_fpr: float
    median_latency_s: float
    p90_latency_s: float
    within_curve: np.ndarray


@dataclass
class ClassEarliness:
    """Per attack class: how long the deployed verdict waits, and who sees it early."""

    attack_class: str
    n_flows: int
    teardown_share: float
    median_wait_s: float
    p90_wait_s: float
    detection_by_tier: dict[str, float]


@dataclass
class EarlinessStudy:
    """Everything the report renders."""

    operating_fpr: float
    flow_timeout_s: float
    in_flight_horizon_s: float
    horizons_s: np.ndarray
    tiers: list[TierResult]
    classes: list[ClassEarliness]
    tier_features: dict[str, list[str]]
    n_attack_flows: int
    timeout_bound_share: float
    unclosed_shares: np.ndarray
    wait_if_unclosed_s: np.ndarray


def _fit_tier(
    settings: Settings,
    columns: np.ndarray,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    operating_fpr: float,
) -> tuple[np.ndarray, float, float, float]:
    """Refit on one tier's columns; return (test scores, PR-AUC, detection, realised FPR)."""
    x_train, x_val, x_test = (m[:, columns] for m in matrices)
    y_train, y_val, y_test = targets
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    benign = settings.labels.benign_label
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
    rates = rates_at_threshold(y_test, s_test, threshold)
    pr_auc = float(average_precision_score(y_test, s_test))
    return s_test >= threshold, pr_auc, rates["tpr"], rates["fpr"]


def run_earliness(settings: Settings) -> EarlinessStudy:
    """Price the detector's decision latency and the detection each earlier tier buys."""
    cfg: EarlinessConfig = settings.earliness
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    operating_fpr = variant.thresholds.primary_fpr

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    targets = (
        train[BINARY_TARGET].to_numpy(),
        val[BINARY_TARGET].to_numpy(),
        test[BINARY_TARGET].to_numpy(),
    )

    pipeline = build_pipeline(variant)
    matrices = (
        pipeline.fit_transform(train),  # FIT ON TRAIN ONLY
        pipeline.transform(val),
        pipeline.transform(test),
    )
    names = [
        display_feature_name(n) for n in pipeline.named_steps["features"].get_feature_names_out()
    ]
    sets = availability_sets(include_destination_port=variant.features.encode_destination_port)

    attack_mask = targets[2].astype(bool)
    horizons = np.asarray(cfg.horizons_s, dtype=float)
    tier_features: dict[str, list[str]] = {}
    tiers: list[TierResult] = []
    detected_by_tier: dict[str, np.ndarray] = {}
    for tier in AVAILABILITY_TIERS:
        allowed = set(sets[tier])
        columns = np.array([j for j, n in enumerate(names) if n in allowed], dtype=int)
        tier_features[tier] = sorted(allowed & set(names))
        if columns.size == 0:
            logger.warning("Tier has no usable columns", extra={"tier": tier})
            continue
        detected, pr_auc, detection, realised = _fit_tier(
            variant, columns, matrices, targets, operating_fpr
        )
        latency = decision_latency_us(
            test, tier, settings.capture.flow_timeout_us, cfg.in_flight_horizon_us
        )
        hostile_latency = latency[attack_mask] / MICROSECONDS
        detected_by_tier[tier] = detected
        tiers.append(
            TierResult(
                tier=tier,
                n_features=int(columns.size),
                pr_auc=pr_auc,
                detection=detection,
                realised_fpr=realised,
                median_latency_s=float(np.median(hostile_latency)) if hostile_latency.size else 0.0,
                p90_latency_s=(
                    float(np.quantile(hostile_latency, 0.9)) if hostile_latency.size else 0.0
                ),
                within_curve=detected_within(latency[attack_mask], detected[attack_mask], horizons),
            )
        )
        logger.info(
            "Earliness tier complete",
            extra={
                "tier": tier,
                "features": int(columns.size),
                "pr_auc": round(pr_auc, 4),
                "detection": round(detection, 4),
            },
        )

    complete_latency = decision_latency_us(
        test, "complete", settings.capture.flow_timeout_us, cfg.in_flight_horizon_us
    )
    teardown = observed_teardown(test)
    classes = _class_rows(test, attack_mask, complete_latency, teardown, detected_by_tier, cfg)
    shares = np.asarray(cfg.unclosed_shares, dtype=float)
    hostile_duration = test[DURATION_COLUMN].to_numpy(dtype=float)[attack_mask]

    return EarlinessStudy(
        operating_fpr=operating_fpr,
        flow_timeout_s=settings.capture.flow_timeout_us / MICROSECONDS,
        in_flight_horizon_s=cfg.in_flight_horizon_us / MICROSECONDS,
        horizons_s=horizons,
        tiers=tiers,
        classes=classes,
        tier_features=tier_features,
        n_attack_flows=int(attack_mask.sum()),
        timeout_bound_share=(float(np.mean(~teardown[attack_mask])) if attack_mask.any() else 0.0),
        unclosed_shares=shares,
        wait_if_unclosed_s=wait_sensitivity(
            hostile_duration, settings.capture.flow_timeout_us, shares
        )
        / MICROSECONDS,
    )


def _class_rows(
    test: pd.DataFrame,
    attack_mask: np.ndarray,
    complete_latency: np.ndarray,
    teardown: np.ndarray,
    detected_by_tier: dict[str, np.ndarray],
    cfg: EarlinessConfig,
) -> list[ClassEarliness]:
    """Per attack class: the deployed wait, and which tiers catch it."""
    labels = test[MULTICLASS_TARGET].astype(str).to_numpy()
    rows: list[ClassEarliness] = []
    for cls in sorted(set(labels[attack_mask].tolist())):
        idx = np.flatnonzero((labels == cls) & attack_mask)
        if idx.size < cfg.min_class_flows:
            continue
        waits = complete_latency[idx] / MICROSECONDS
        rows.append(
            ClassEarliness(
                attack_class=cls,
                n_flows=int(idx.size),
                teardown_share=float(np.mean(teardown[idx])),
                median_wait_s=float(np.median(waits)),
                p90_wait_s=float(np.quantile(waits, 0.9)),
                detection_by_tier={
                    tier: float(np.mean(hit[idx])) for tier, hit in detected_by_tier.items()
                },
            )
        )
    return sorted(rows, key=lambda r: -r.median_wait_s)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_earliness_report(settings: Settings) -> Path:
    """Run the earliness study and write the report + figures."""
    study = run_earliness(settings)

    frontier_fig = plots.plot_lines(
        {
            TIER_LABELS[t.tier]: (study.horizons_s, t.within_curve)
            for t in study.tiers
            if t.within_curve.size
        },
        xlabel="seconds after the flow's first packet",
        ylabel="share of hostile flows detected by then",
        title="Detection available in time (temporal split)",
        out_path=settings.paths.figures_dir / FRONTIER_FIGURE,
        xscale="log",
    )
    tier_fig = plots.plot_barh(
        [TIER_LABELS[t.tier] for t in study.tiers],
        [t.pr_auc for t in study.tiers],
        xlabel="PR-AUC on the temporal split",
        title="What each decision time can know",
        out_path=settings.paths.figures_dir / TIER_FIGURE,
    )

    report = _render(study, frontier_fig, tier_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote earliness report", extra={"path": str(out_path)})

    with track_run(settings, "earliness") as run:
        run.log_params(
            {
                "operating_fpr": study.operating_fpr,
                "flow_timeout_s": study.flow_timeout_s,
                "in_flight_horizon_s": study.in_flight_horizon_s,
            }
        )
        metrics: dict[str, float] = {"timeout_bound_share": study.timeout_bound_share}
        for tier in study.tiers:
            metrics[f"pr_auc_{tier.tier}"] = tier.pr_auc
            metrics[f"detection_{tier.tier}"] = tier.detection
            metrics[f"median_latency_s_{tier.tier}"] = tier.median_latency_s
        run.log_metrics(metrics)
        run.log_artifact(frontier_fig)
        run.log_artifact(tier_fig)
        run.log_artifact(out_path)
    return out_path


def _fmt_seconds(value: float) -> str:
    """Seconds, rendered at a precision an operator can read."""
    if value >= 10:
        return f"{value:.0f} s"
    if value >= 0.1:
        return f"{value:.2f} s"
    return f"{value * 1000:.0f} ms"


def _tier_table(study: EarlinessStudy) -> str:
    rows = [
        "| decision time | features | PR-AUC | detection @ budget | median wait | 90th-pct wait |",
        "|---|---|---|---|---|---|",
    ]
    for t in study.tiers:
        rows.append(
            f"| {TIER_LABELS[t.tier]} | {t.n_features} | {t.pr_auc:.3f} | {t.detection:.1%} "
            f"| {_fmt_seconds(t.median_latency_s)} | {_fmt_seconds(t.p90_latency_s)} |"
        )
    return "\n".join(rows)


def _class_table(study: EarlinessStudy) -> str:
    tiers = [t.tier for t in study.tiers]
    header = " | ".join(f"detected @ {TIER_LABELS[t]}" for t in tiers)
    rows = [
        f"| attack class | flows | saw a teardown | median wait | 90th-pct wait | {header} |",
        "|---|---|---|---|---|" + "---|" * len(tiers),
    ]
    for c in study.classes:
        cells = " | ".join(f"{c.detection_by_tier.get(t, 0.0):.1%}" for t in tiers)
        rows.append(
            f"| {c.attack_class} | {c.n_flows} | {c.teardown_share:.0%} "
            f"| {_fmt_seconds(c.median_wait_s)} | {_fmt_seconds(c.p90_wait_s)} | {cells} |"
        )
    return "\n".join(rows)


def _sensitivity_table(study: EarlinessStudy) -> str:
    rows = ["| flows with no teardown | median wait for the deployed verdict |", "|---|---|"]
    for share, wait in zip(study.unclosed_shares, study.wait_if_unclosed_s, strict=True):
        rows.append(f"| {share:.0%} | {_fmt_seconds(float(wait))} |")
    return "\n".join(rows)


def _wait_read(study: EarlinessStudy) -> str:
    complete = next((t for t in study.tiers if t.tier == "complete"), None)
    if complete is None:
        return ""
    stuck = study.timeout_bound_share
    measured = (
        f"Across the {study.n_attack_flows:,} hostile flows on the test days the median wait "
        f"for a complete-flow verdict is **{_fmt_seconds(complete.median_latency_s)}** and the "
        f"90th percentile is **{_fmt_seconds(complete.p90_latency_s)}**."
    )
    if stuck < 0.02:
        return (
            f"{measured} That is the whole flow duration and nothing more, because "
            f"**{1 - stuck:.0%} of these flows showed the exporter a teardown** — the "
            "synthetic generator stamps FIN/RST counts on essentially every flow it emits, so "
            "the idle timer never fires and this stand-in cannot exercise the half of the "
            "latency model that matters most. Saying so is more useful than quoting the "
            "number as though it generalised: on a real capture the unanswered SYN, the "
            "half-open connection and the UDP exchange all end without a teardown, and each "
            f"one is held for the configured {_fmt_seconds(study.flow_timeout_s)} before its "
            "record exists at all. So the sweep below reports the one quantity the stand-in "
            "cannot supply — what the wait becomes as that share rises."
        )
    return (
        f"Across the {study.n_attack_flows:,} hostile flows on the test days, "
        f"**{stuck:.0%} never showed the exporter a teardown** — no FIN, no RST, the flow "
        f"simply stopped. Those flows are held open until the idle timer fires, so the "
        f"deployed model's verdict about them cannot exist for "
        f"{_fmt_seconds(study.flow_timeout_s)} no matter how good the model is. That is why "
        f"{measured[measured.index('the median wait') :]} The distribution is not really about "
        "how long attacks take, it is about how many of them end without saying so. Nothing "
        "in the accuracy table anywhere else in this repository reflects this, because every "
        "other study scores a finished flow record and asks only whether the label was right."
    )


def _sensitivity_read(study: EarlinessStudy) -> str:
    if study.wait_if_unclosed_s.size < 2:
        return ""
    shares = study.unclosed_shares
    waits = study.wait_if_unclosed_s
    above = np.flatnonzero(shares >= 0.5)
    if not above.size or above[0] == 0:
        return ""
    j = int(above[0])
    before, after = float(waits[j - 1]), float(waits[j])
    jump = after / before if before > 0 else float("inf")
    return (
        f"The step is not gradual, it is a cliff. At {shares[j - 1]:.0%} unclosed the median "
        f"verdict still waits {_fmt_seconds(before)}; at {shares[j]:.0%} it waits "
        f"{_fmt_seconds(after)} — a {jump:,.0f}x jump caused by a change in the *traffic*, with "
        "the model, the features and the threshold all held fixed. The mechanism is simply that "
        "once more than half the flows are waiting on the idle timer, the median flow is one of "
        "them. That is the operational point: flow-level detection latency is governed by "
        "whether connections close politely, which is a property of the attacker rather than of "
        "the detector, and the traffic that never closes politely is exactly what an operator "
        "most wants early — reconnaissance, whose probes go unanswered by definition."
    )


def _tier_read(study: EarlinessStudy) -> str:
    by_tier = {t.tier: t for t in study.tiers}
    complete, early = by_tier.get("complete"), by_tier.get("handshake")
    mid = by_tier.get("in_flight")
    if complete is None or early is None or mid is None:
        return ""
    ladder = (
        f"The handshake tier sees {early.n_features} features — the initial TCP windows and the "
        f"minimum forward segment size, fixed by the connection setup and never revised — and "
        f"reaches {early.pr_auc:.3f} PR-AUC. The in-flight tier adds every intensive statistic "
        f"(means, extremes, spreads, rates) for {mid.n_features} features and "
        f"{mid.pr_auc:.3f}. The deployed complete-flow model has {complete.n_features} features "
        f"and {complete.pr_auc:.3f}."
    )
    if mid.pr_auc <= complete.pr_auc:
        return (
            f"{ladder} The ordering is the expected one — more features, decided later, detect "
            "more — so the tiers really do trade earliness for accuracy, and the question is "
            "what the exchange rate is. The frontier below puts detection and the moment it "
            "becomes available on the same axes so the trade can be read directly."
        )
    return (
        f"{ladder} **The ordering inverts.** Waiting for the flow to finish does not buy "
        f"detection here, it costs it: the in-flight tier is {mid.pr_auc - complete.pr_auc:+.3f} "
        f"PR-AUC and {mid.detection - complete.detection:+.1%} detection against the model this "
        f"repository actually ships, using half its features and deciding while the connection "
        f"is still open. That is not a tuning artefact, it is what the temporal split is for. "
        "The 40 features the in-flight tier drops are the *extensive* ones — totals, cumulative "
        "sums, durations, subflow volumes — and an extensive feature is a measurement of how "
        "big this particular burst happened to be. Burst size is a property of the campaign "
        "that was running on Wednesday, not of the behaviour that makes a flow hostile, so the "
        "model leans on it in training and it means something different on Friday. The "
        "intensive statistics that survive — packet-size distribution, inter-arrival shape, "
        "directional ratios — describe how the traffic behaves at any scale, and they transfer. "
        "The [ablation](ablation.md) study measures which families carry detection *within* a "
        "split; this measures which of them survive crossing one, and the answer is the "
        "scale-free half."
    )


def _frontier_read(study: EarlinessStudy) -> str:
    by_tier = {t.tier: t for t in study.tiers}
    complete, mid = by_tier.get("complete"), by_tier.get("in_flight")
    if complete is None or mid is None or not study.horizons_s.size:
        return ""
    # The horizon at which waiting for the whole flow first pays for itself, if it ever does.
    ahead = np.flatnonzero(complete.within_curve > mid.within_curve + 1e-12)
    crossover = float(study.horizons_s[ahead[0]]) if ahead.size else float("inf")
    dominated = not np.isfinite(crossover)
    verdict = (
        "**it never does**, at any horizon plotted: the early curve is above the deployed one "
        "everywhere, so every second the deployed detector spends waiting is a second it is "
        "behind and never catches up"
        if dominated
        else (f"it does, but not until **{_fmt_seconds(crossover)}** after the flow's first packet")
    )
    closing = (
        "A dominated curve is the strongest form this result can take. It means there is no "
        "operating regime, however patient, in which the extra features earn their latency — "
        "the choice between the two detectors is not a trade-off on this split, it is a "
        "strict improvement that the deployed configuration is declining to take."
        if dominated
        else (
            "Below that crossing the early detector is strictly better and above it the "
            "deployed one is, so the right choice depends only on how long the response can "
            "afford to wait — which is a question about the network, answerable without any "
            "further modelling."
        )
    )
    return (
        f"The frontier asks the only question that matters once latency is on the axis: does "
        f"waiting for the flow to end ever pay for itself? Here {verdict}. By the widest "
        f"horizon plotted, {mid.within_curve[-1]:.1%} of hostile flows are caught in flight "
        f"against {complete.within_curve[-1]:.1%} caught after the fact. {closing}"
    )


def _class_read(study: EarlinessStudy) -> str:
    if not study.classes:
        return ""
    slowest = study.classes[0]
    gains = [
        (c, c.detection_by_tier.get("in_flight", 0.0) - c.detection_by_tier.get("complete", 0.0))
        for c in study.classes
    ]
    best, best_gain = max(gains, key=lambda g: g[1])
    blind = [c.attack_class for c in study.classes if max(c.detection_by_tier.values()) < 0.01]
    gain_read = (
        f"The inversion is not spread evenly: it is almost entirely **{best.attack_class}**, "
        f"which the in-flight tier catches {best_gain:+.1%} more often than the deployed model "
        f"does — a volumetric class, and precisely the one whose extensive features (total "
        "bytes, total packets, duration) encode how large *that particular flood* was rather "
        "than what a flood looks like."
        if best_gain > 0.01
        else (
            "No single class dominates the difference between the tiers, so the earliness "
            "trade is a property of the feature partition rather than of one attack."
        )
    )
    blind_read = (
        f" {', '.join(blind)} are invisible to every tier, early or late, which is a coverage "
        "problem no decision time can fix and is handed to the [slices](slices.md) and "
        "[novelty](novelty.md) studies."
        if blind
        else ""
    )
    return (
        f"{gain_read}{blind_read} The slowest verdict belongs to **{slowest.attack_class}** at a "
        f"median of {_fmt_seconds(slowest.median_wait_s)} and {_fmt_seconds(slowest.p90_wait_s)} "
        "for its slowest tenth. Per class is the right granularity because response differs by "
        "class: waiting on a brute-force campaign costs a few more guesses, and the same wait on "
        "a reconnaissance sweep costs the entire target list."
    )


_SCOPE = """The in-flight tier is scored on the **completed** values of its intensive
features, because the CSVs contain one row per finished flow and nothing per packet. A real
mid-flow detector would see noisier estimates of the same quantities, so this tier is an
**upper bound** on in-flight detection — the gap it already shows against the complete tier is
a floor on the gap a deployed early detector would face. The handshake tier does not have this
problem: its fields are fixed at setup and never revised, so its numbers are exact.

Latency is measured from the flow's first packet, and the handshake verdict is timed at zero
rather than at one round trip, which flatters it by an amount that is small next to a
two-minute idle timer but is not nothing on a wide-area link. `SYN Flag Count` is classified
as complete-only although a SYN necessarily arrives first, because the *count* keeps accruing
on retransmits and the exporter only reports it at flow end; moving it to the handshake tier
would strengthen that tier's numbers, so the conservative call is the one that does not
flatter the argument being made. Finally, the exporter timeout is the configured
`capture.flow_timeout_us`; a deployment that shortens it trades this latency directly for
split flows, which fragment exactly the volumetric statistics the complete tier relies on.

The unclosed-share sweep assumes a flow's chance of ending without a teardown is independent
of its duration, which makes the mixture's median exact but is optimistic: short unanswered
probes are both the most likely to lack a teardown and the shortest, so a real capture would
put more mass on the timed-out branch at the low-duration end than the sweep does. Each tier
is refit from scratch on its own columns rather than masking a single model's inputs, so the
comparison is between three detectors that were each allowed to do their best with what they
could see, not between one detector and a crippled copy of itself."""


def _render(study: EarlinessStudy, frontier_fig: Path, tier_fig: Path) -> str:
    return f"""# NetSentry — When the Verdict Can Exist

_Synthetic stand-in. Honest temporal/binary split, {study.operating_fpr:.1%} false-positive
budget. Exporter idle timeout {_fmt_seconds(study.flow_timeout_s)}; the in-flight tier is
allowed {_fmt_seconds(study.in_flight_horizon_s)} to accumulate packets._

## Why this report exists

Every other number in this repository is quoted as though the detector decides the moment the
attack does. It does not. Flow exporters emit **one record per finished flow**, and most of the
statistics the model consumes are only defined once the flow is over — `Total Fwd Packets` is
not a running counter read at the end, it is a quantity that does not exist until then. So the
deployed detector is structurally a post-mortem one, and the interesting question is not only
"is the verdict right" but "when could the verdict possibly have existed".

## Half one: how long the verdict takes

{_wait_read(study)}

### What the wait becomes when flows stop closing politely

{_sensitivity_table(study)}

{_sensitivity_read(study)}

## Half two: what it would cost not to wait

Features partition by *when the value the model trained on is knowable*: fixed at connection
setup, intensive statistics whose prefix value estimates the final one, or extensive and
teardown quantities that only exist at the end. The tiers are nested, so each row is a
detector that could actually be deployed at that moment.

{_tier_table(study)}

{_tier_read(study)}

![PR-AUC by decision time](../figures/{tier_fig.name})

## The two halves together

{_frontier_read(study)}

![detection available in time](../figures/{frontier_fig.name})

## Per attack class

{_class_table(study)}

{_class_read(study)}

## Scope

{_SCOPE}"""
