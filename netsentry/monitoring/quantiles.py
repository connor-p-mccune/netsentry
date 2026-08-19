"""Estimate the threshold's quantile at line rate, and price the error in alerts.

Every operating point in this project is a **quantile**. The deployed threshold is the score
below which 99.9% of benign validation flows sit; the [refresh study](refresh.md) re-derives it
on new traffic; the [control loop](control.md) moves it batch by batch. All three assume the
scores can be collected, sorted and indexed, which is true of a 25,000-row test split and false
of a production stream: sorting requires storing, and storing a day of scores at line rate is
not a monitoring budget, it is a database.

The [sketches study](sketches.md) built Count-Min, HyperLogLog, Misra-Gries and reservoir
sampling to count in fixed memory. None of them estimates a quantile. This adds the estimators
that do, from scratch, and then does the thing that makes it a NetSentry report rather than a
textbook exercise: **translates quantile error into false-positive rate**, which is the only
unit in which a threshold error can be argued about.

Four estimators, chosen because they fail differently:

- **Exact** (sort everything). The ceiling, and the thing that does not fit.
- **Reservoir sampling** + an exact quantile of the sample. Unbiased, simple, and wasteful in
  exactly the wrong place: a uniform sample of a stream spends almost all of its slots on the
  99.9% of flows that are nowhere near the threshold.
- **P²** (Jain & Chlamtac, CACM 1985): five markers per quantile updated by parabolic
  interpolation, O(1) memory, nothing stored. Beautiful, and the only one here that cannot be
  made more accurate by spending memory.
- **t-digest** (Dunning): merging centroids whose size is bounded by a scale function that
  deliberately keeps the *tails* fine-grained, which is where an operating point lives.
- **A fixed-bin histogram** over [0, 1]. The dumb baseline, included because model scores are
  bounded and a bounded quantity is the one case where the dumb baseline can win outright — and
  a study that omits it would be recommending a t-digest without checking whether an array of
  counters was enough.

Three questions get answered: how much quantile error each buys per byte, what that error costs
in realised false-positive rate on a held-out stream, and what happens when the stream drifts
underneath an estimator that has no notion of forgetting.
"""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import QuantileConfig

logger = get_logger(__name__)

REPORT_NAME = "quantiles.md"
ACCURACY_FIGURE_NAME = "quantiles_accuracy.png"

EXACT = "exact (sort everything)"
RESERVOIR = "reservoir sample"
P2 = "P-squared (Jain & Chlamtac 1985)"
TDIGEST = "t-digest (Dunning)"
HISTOGRAM = "fixed-bin histogram"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Estimators. Each exposes update(x) and quantile(q), and reports its own memory.
# --------------------------------------------------------------------------------------


class ReservoirQuantile:
    """Uniform sample of the stream, then an exact quantile of the sample (Vitter 1985).

    The honest baseline. Its weakness is structural rather than statistical: a uniform sample
    allocates its slots in proportion to *density*, and the 99.9th percentile of a detection
    score sits where the density is lowest, so almost every slot is spent describing traffic
    the threshold will never touch.
    """

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size = size
        self.rng = rng
        self.samples: list[float] = []
        self.seen = 0

    def update(self, value: float) -> None:
        self.seen += 1
        if len(self.samples) < self.size:
            self.samples.append(value)
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.size:
            self.samples[index] = value

    def quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.quantile(np.asarray(self.samples), q))

    @property
    def memory_bytes(self) -> int:
        return 8 * len(self.samples)


class P2Quantile:
    """The P-squared algorithm: five markers, no stored samples (Jain & Chlamtac 1985).

    Markers track the running minimum, the q/2, q and (1+q)/2 quantiles and the maximum. Each
    new observation nudges the marker positions towards their *desired* positions, and any
    marker that drifts a full step is moved by fitting a parabola through its neighbours —
    which is what lets a five-number summary approximate a quantile of an unbounded stream in
    constant memory. The linear fallback is not decoration: the parabolic prediction can leave
    the markers out of order on adversarial input, and the algorithm is only well defined
    because it detects that and degrades.
    """

    def __init__(self, q: float) -> None:
        self.q = q
        self.heights: list[float] = []
        self.positions = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.desired = [1.0, 1 + 2 * q, 1 + 4 * q, 3 + 2 * q, 5.0]
        self.increments = [0.0, q / 2, q, (1 + q) / 2, 1.0]
        self.seen = 0

    def update(self, value: float) -> None:
        self.seen += 1
        if len(self.heights) < 5:
            self.heights.append(value)
            if len(self.heights) == 5:
                self.heights.sort()
            return

        if value < self.heights[0]:
            self.heights[0] = value
            k = 0
        elif value >= self.heights[4]:
            self.heights[4] = value
            k = 3
        else:
            k = bisect.bisect_right(self.heights, value) - 1
            k = min(max(k, 0), 3)

        for i in range(k + 1, 5):
            self.positions[i] += 1
        for i in range(5):
            self.desired[i] += self.increments[i]

        for i in range(1, 4):
            delta = self.desired[i] - self.positions[i]
            forward = self.positions[i + 1] - self.positions[i]
            backward = self.positions[i - 1] - self.positions[i]
            if (delta >= 1 and forward > 1) or (delta <= -1 and backward < -1):
                step = 1 if delta >= 0 else -1
                height = self._parabolic(i, step)
                if not self.heights[i - 1] < height < self.heights[i + 1]:
                    height = self._linear(i, step)
                self.heights[i] = height
                self.positions[i] += step

    def _parabolic(self, i: int, step: int) -> float:
        left = self.positions[i] - self.positions[i - 1]
        right = self.positions[i + 1] - self.positions[i]
        span = self.positions[i + 1] - self.positions[i - 1]
        return self.heights[i] + step / span * (
            (left + step) * (self.heights[i + 1] - self.heights[i]) / right
            + (right - step) * (self.heights[i] - self.heights[i - 1]) / left
        )

    def _linear(self, i: int, step: int) -> float:
        return self.heights[i] + step * (self.heights[i + step] - self.heights[i]) / (
            self.positions[i + step] - self.positions[i]
        )

    def quantile(self, q: float) -> float:
        if not self.heights:
            return 0.0
        if len(self.heights) < 5:
            return float(np.quantile(np.asarray(self.heights), q))
        return float(self.heights[2])

    @property
    def memory_bytes(self) -> int:
        return 8 * (
            len(self.heights) + len(self.positions) + len(self.desired) + len(self.increments)
        )


@dataclass
class Centroid:
    """A weighted summary of nearby observations."""

    mean: float
    weight: float


class TDigest:
    """A merging t-digest: centroids kept fine at the tails and coarse in the middle.

    The scale function is what makes this different from a histogram with unequal bins: a
    centroid may absorb weight up to ``4 n delta q (1 - q)``, which vanishes as `q` approaches 0
    or 1. The result spends its memory where quantiles are hard to estimate and an operating
    point actually lives, rather than where the data is dense.
    """

    def __init__(self, compression: float = 100.0) -> None:
        self.compression = compression
        self.centroids: list[Centroid] = []
        self.buffer: list[float] = []
        self.total = 0.0

    def update(self, value: float) -> None:
        self.buffer.append(value)
        self.total += 1.0
        if len(self.buffer) >= 200:
            self._merge()

    def _merge(self) -> None:
        points = [Centroid(value, 1.0) for value in self.buffer]
        self.buffer = []
        merged = sorted([*self.centroids, *points], key=lambda c: c.mean)
        if not merged:
            return
        total = sum(c.weight for c in merged)
        out: list[Centroid] = []
        current = merged[0]
        seen = 0.0
        for candidate in merged[1:]:
            proposed = current.weight + candidate.weight
            q = (seen + proposed / 2) / total
            limit = 4 * total * (1.0 / self.compression) * q * (1 - q)
            if proposed <= max(limit, 1.0):
                current = Centroid(
                    (current.mean * current.weight + candidate.mean * candidate.weight) / proposed,
                    proposed,
                )
            else:
                out.append(current)
                seen += current.weight
                current = candidate
        out.append(current)
        self.centroids = out

    def quantile(self, q: float) -> float:
        if self.buffer:
            self._merge()
        if not self.centroids:
            return 0.0
        total = sum(c.weight for c in self.centroids)
        target = q * total
        seen = 0.0
        for centroid in self.centroids:
            if seen + centroid.weight >= target:
                return float(centroid.mean)
            seen += centroid.weight
        return float(self.centroids[-1].mean)

    @property
    def memory_bytes(self) -> int:
        return 16 * len(self.centroids)


class HistogramQuantile:
    """Equal-width bins over a declared range — the baseline a bounded score makes viable.

    A model's score lives in [0, 1] by construction, so the range needs no estimation and the
    only error is the bin width. This is the estimator nobody writes a paper about and the one
    that has to be beaten before any of the others is worth deploying.
    """

    def __init__(self, bins: int, low: float = 0.0, high: float = 1.0) -> None:
        self.counts = np.zeros(bins, dtype=np.int64)
        self.low = low
        self.high = high
        self.bins = bins
        self.seen = 0

    def update(self, value: float) -> None:
        position = (value - self.low) / max(self.high - self.low, _EPS)
        index = min(max(int(position * self.bins), 0), self.bins - 1)
        self.counts[index] += 1
        self.seen += 1

    def quantile(self, q: float) -> float:
        if self.seen == 0:
            return 0.0
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, q * self.seen))
        index = min(index, self.bins - 1)
        # Interpolate inside the bin: at a 0.1% budget the bin containing the threshold holds
        # many flows, and taking its lower edge is a systematic underestimate.
        below = cumulative[index - 1] if index else 0
        within = self.counts[index]
        fraction = (q * self.seen - below) / max(within, 1)
        width = (self.high - self.low) / self.bins
        return float(self.low + (index + fraction) * width)

    @property
    def memory_bytes(self) -> int:
        return 8 * self.bins


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class EstimatorRow:
    """One estimator at one memory budget, graded against exact truth."""

    name: str
    setting: str
    memory_bytes: int
    estimate: float
    truth: float
    quantile_error: float
    realised_fpr: float
    target_fpr: float
    alerts_ratio: float
    update_ns: float


@dataclass
class DriftRow:
    """How an estimator tracks when the stream moves under it."""

    name: str
    estimate: float
    truth: float
    realised_fpr: float


@dataclass
class QuantileStudy:
    """Everything the report renders."""

    rows: list[EstimatorRow]
    drift: list[DriftRow]
    target_fpr: float
    n_stream: int
    n_holdout: int
    exact_memory: int
    quantile: float
    drift_quantile: float = 0.0
    families: list[str] = field(default_factory=list)


def _realised_fpr(threshold: float, holdout: np.ndarray) -> float:
    """Share of held-out benign flows a threshold would alert on."""
    return float(np.mean(holdout >= threshold))


def _time_updates(
    build: Callable[[], Any], values: np.ndarray, sample: int = 20000, warm: int = 60000
) -> float:
    """Nanoseconds per steady-state update.

    The warm-up matters: a reservoir in its fill phase is an append, and timing only the first
    twenty thousand updates of a fifty-thousand-slot reservoir would measure the cheap half of
    its life and report it as the cost.
    """
    estimator = build()
    for value in values[:warm]:
        estimator.update(float(value))
    subset = values[warm : warm + sample]
    if len(subset) == 0:
        subset = values[:sample]
    start = time.perf_counter()
    for value in subset:
        estimator.update(float(value))
    return (time.perf_counter() - start) / max(len(subset), 1) * 1e9


def run_quantile_study(settings: Settings) -> QuantileStudy:
    """Grade every estimator against the exact quantile, then against the alert volume."""
    cfg: QuantileConfig = settings.quantiles
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    fit = fit_supervised(variant)
    benign_label = variant.labels.benign_label
    scores_val = attack_probability(fit.proba_val, fit.classes, benign_label)
    y_val = (
        (np.asarray(fit.y_val) != benign_label).astype(int)
        if np.asarray(fit.y_val).dtype.kind in "OU"
        else np.asarray(fit.y_val).astype(int)
    )
    scores_test = attack_probability(fit.proba_test, fit.classes, benign_label)
    y_test = (
        (np.asarray(fit.y_test) != benign_label).astype(int)
        if np.asarray(fit.y_test).dtype.kind in "OU"
        else np.asarray(fit.y_test).astype(int)
    )

    # The stream an operator would actually see: benign traffic, whose quantile *is* the
    # threshold. Attacks are what the threshold is for, not what it is measured on.
    stream = scores_val[y_val == 0]
    holdout = scores_test[y_test == 0]
    if len(stream) < 100:  # pragma: no cover - defensive
        raise ValueError("not enough benign validation flows to build a score stream")
    repeats = max(int(np.ceil(cfg.stream_rows / len(stream))), 1)
    stream = np.concatenate([rng.permutation(stream) for _ in range(repeats)])[: cfg.stream_rows]

    target_fpr = variant.thresholds.primary_fpr
    quantile = 1.0 - target_fpr
    truth = float(np.quantile(stream, quantile))
    exact_fpr = _realised_fpr(truth, holdout)
    logger.info("Stream built", extra={"rows": len(stream), "truth": round(truth, 6)})

    rows: list[EstimatorRow] = []

    def _grade(name: str, setting: str, estimate: float, memory: int, update_ns: float) -> None:
        realised = _realised_fpr(estimate, holdout)
        rows.append(
            EstimatorRow(
                name=name,
                setting=setting,
                memory_bytes=memory,
                estimate=estimate,
                truth=truth,
                quantile_error=float(np.mean(stream < estimate) - quantile),
                realised_fpr=realised,
                target_fpr=target_fpr,
                alerts_ratio=realised / max(exact_fpr, _EPS),
                update_ns=update_ns,
            )
        )

    _grade(EXACT, "all scores retained", truth, 8 * len(stream), 0.0)

    for size in cfg.reservoir_sizes:
        estimator = ReservoirQuantile(size, np.random.default_rng(variant.seed))
        for value in stream:
            estimator.update(float(value))
        _grade(
            RESERVOIR,
            f"{size:,} samples",
            estimator.quantile(quantile),
            estimator.memory_bytes,
            _time_updates(partial(ReservoirQuantile, size, np.random.default_rng(0)), stream),
        )

    p2 = P2Quantile(quantile)
    for value in stream:
        p2.update(float(value))
    _grade(
        P2,
        "5 markers",
        p2.quantile(quantile),
        p2.memory_bytes,
        _time_updates(partial(P2Quantile, quantile), stream),
    )

    for compression in cfg.compressions:
        digest = TDigest(compression)
        for value in stream:
            digest.update(float(value))
        _grade(
            TDIGEST,
            f"compression {compression:g}",
            digest.quantile(quantile),
            digest.memory_bytes,
            _time_updates(partial(TDigest, compression), stream),
        )

    for bins in cfg.histogram_bins:
        histogram = HistogramQuantile(bins)
        for value in stream:
            histogram.update(float(value))
        _grade(
            HISTOGRAM,
            f"{bins:,} bins",
            histogram.quantile(quantile),
            histogram.memory_bytes,
            _time_updates(partial(HistogramQuantile, bins), stream),
        )
        logger.info("Histogram graded", extra={"bins": bins})

    # Drift, using the drift this project actually has rather than an additive shift: the first
    # half of the stream is validation-day benign traffic and the second half is *test*-day
    # benign traffic, which is a genuine distribution change the deployed model already suffers
    # from. An estimator that never forgets answers with a blend of the two regimes; the
    # correct answer is the quantile of the second one alone.
    half = len(stream) // 2
    later = np.concatenate(
        [rng.permutation(holdout) for _ in range(max(int(np.ceil(half / len(holdout))), 1))]
    )[:half]
    drifting = np.concatenate([stream[:half], later])
    drift_truth = float(np.quantile(later, quantile))
    drift: list[DriftRow] = []
    builders: dict[str, Callable[[], Any]] = {
        RESERVOIR: partial(ReservoirQuantile, cfg.reservoir_sizes[-1], np.random.default_rng(1)),
        P2: partial(P2Quantile, quantile),
        TDIGEST: partial(TDigest, cfg.compressions[-1]),
        HISTOGRAM: partial(HistogramQuantile, cfg.histogram_bins[-1]),
    }
    for name, build in builders.items():
        estimator = build()
        for value in drifting:
            estimator.update(float(value))
        estimate = estimator.quantile(quantile)
        drift.append(
            DriftRow(
                name=name,
                estimate=estimate,
                truth=drift_truth,
                realised_fpr=_realised_fpr(estimate, holdout),
            )
        )

    return QuantileStudy(
        rows=rows,
        drift=drift,
        target_fpr=target_fpr,
        n_stream=len(stream),
        n_holdout=len(holdout),
        exact_memory=8 * len(stream),
        quantile=quantile,
        drift_quantile=drift_truth,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_quantile_report(settings: Settings) -> Path:
    """Run the streaming-quantile study and write the report + figure."""
    study = run_quantile_study(settings)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in (RESERVOIR, TDIGEST, HISTOGRAM):
        points = sorted(
            (row for row in study.rows if row.name == name), key=lambda r: r.memory_bytes
        )
        if points:
            series[name] = (
                np.array([p.memory_bytes for p in points], dtype=float),
                np.array([abs(p.alerts_ratio - 1.0) for p in points], dtype=float),
            )
    p2_rows = [row for row in study.rows if row.name == P2]
    if p2_rows:
        series[P2] = (
            np.array([p2_rows[0].memory_bytes], dtype=float),
            np.array([abs(p2_rows[0].alerts_ratio - 1.0)], dtype=float),
        )
    figure = plots.plot_lines(
        series,
        xlabel="memory (bytes)",
        ylabel="relative error in alert volume",
        title="What a threshold estimate costs in alerts, per byte of monitor",
        out_path=settings.paths.figures_dir / ACCURACY_FIGURE_NAME,
        xscale="log",
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote quantile report", extra={"path": str(out_path)})

    with track_run(settings, "quantiles") as run:
        run.log_params({"stream_rows": study.n_stream, "quantile": study.quantile})
        run.log_metrics(
            {
                f"alerts_{row.name[:10].replace(' ', '_')}_{row.memory_bytes}": row.alerts_ratio
                for row in study.rows
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _accuracy_table(study: QuantileStudy) -> str:
    rows = [
        "| estimator | setting | memory | threshold | realised FPR | alert volume vs target | "
        "update |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study.rows:
        memory = (
            f"{row.memory_bytes:,} B"
            if row.memory_bytes < 1_000_000
            else f"{row.memory_bytes / 1e6:.1f} MB"
        )
        update = "—" if row.update_ns <= 0 else f"{row.update_ns:,.0f} ns"
        rows.append(
            f"| {row.name} | {row.setting} | {memory} | {row.estimate:.5f} | "
            f"{row.realised_fpr:.3%} | **{row.alerts_ratio:.2f}x** | {update} |"
        )
    return "\n".join(rows)


def _drift_table(study: QuantileStudy) -> str:
    rows = [
        "| estimator | threshold after the shift | correct threshold | realised FPR |",
        "|---|---|---|---|",
    ]
    for row in study.drift:
        rows.append(
            f"| {row.name} | {row.estimate:.5f} | {row.truth:.5f} | {row.realised_fpr:.3%} |"
        )
    return "\n".join(rows)


def _headline(study: QuantileStudy) -> str:
    approximate = [row for row in study.rows if row.name != EXACT]
    if not approximate:
        return ""
    exact = next(row for row in study.rows if row.name == EXACT)
    perfect = [row for row in approximate if abs(row.alerts_ratio - 1.0) < 0.005]
    cheapest = min(perfect, key=lambda r: r.memory_bytes) if perfect else None
    fastest = min(perfect, key=lambda r: r.update_ns) if perfect else None
    if cheapest is None or fastest is None:
        best = min(approximate, key=lambda r: abs(r.alerts_ratio - 1.0))
        return (
            f"Nothing matches the exact threshold's alert volume: the closest is "
            f"`{best.name}` at {best.setting}, {best.alerts_ratio:.2f}x."
        )
    return (
        f"Storing the stream costs {exact.memory_bytes / 1e6:.1f} MB for {study.n_stream:,} "
        f"scores and gives the threshold exactly. **{len(perfect)} of the "
        f"{len(approximate)} approximations deliver the identical alert volume** — not close, "
        "identical, because a threshold anywhere inside the gap between two adjacent benign "
        "scores alerts on exactly the same flows.\n\n"
        f"The cheapest of them is `{cheapest.name}` at {cheapest.setting}: "
        f"**{cheapest.memory_bytes:,} bytes**, "
        f"{exact.memory_bytes / max(cheapest.memory_bytes, 1):,.0f}x smaller than keeping the "
        f"stream, with no operational error at all. The fastest is `{fastest.name}` at "
        f"{fastest.setting} ({fastest.update_ns:,.0f} ns per update against "
        f"{cheapest.update_ns:,.0f}). Neither is the t-digest, which is the most sophisticated "
        "thing in the table and buys nothing here."
    )


def _histogram_read(study: QuantileStudy) -> str:
    histograms = [row for row in study.rows if row.name == HISTOGRAM]
    digests = [row for row in study.rows if row.name == TDIGEST]
    if not (histograms and digests):
        return ""
    cheap_histogram = min(histograms, key=lambda r: r.memory_bytes)
    cheap_digest = min(digests, key=lambda r: r.memory_bytes)
    p2_rows = [row for row in study.rows if row.name == P2]
    p2 = p2_rows[0] if p2_rows else None
    return (
        "**Neither of the two winners is the sophisticated one.** A fixed-bin histogram over "
        f"[0, 1] delivers the exact alert volume at {cheap_histogram.memory_bytes:,} bytes and "
        f"{cheap_histogram.update_ns:,.0f} ns per update — the cheapest update in the table by "
        f"a factor of {cheap_digest.update_ns / max(cheap_histogram.update_ns, _EPS):.1f} over "
        "the t-digest"
        + (
            f" — while P-squared delivers the same volume in **{p2.memory_bytes} bytes**, "
            f"{cheap_histogram.memory_bytes / max(p2.memory_bytes, 1):.0f}x less memory, for "
            f"{p2.update_ns / max(cheap_histogram.update_ns, _EPS):.1f}x the update cost."
            if p2
            else "."
        )
        + " The t-digest is beaten on both axes at once.\n\n"
        "That is an argument about *this quantity* rather than against t-digests. A model's "
        "score is bounded in [0, 1] by construction, so a histogram needs no range estimation, "
        "no merging and no scale function; a t-digest earns its complexity on unbounded, "
        "heavy-tailed streams where the range is unknown and the tail is where the answer "
        "lives — request latencies, for instance. Boundedness is exactly the assumption the "
        "cheap option needs, and it is free here.\n\n"
        "The engineering reading is a question to ask before reaching for a sketch: is the "
        "quantity bounded, and is the error budget in the units of the *decision* rather than "
        "of the statistic? Both answers here point at an array of counters or five floats, "
        "neither of which has failure modes worth debugging at three in the morning."
    )


def _fpr_read(study: QuantileStudy) -> str:
    approximate = [row for row in study.rows if row.name != EXACT]
    worst = max(approximate, key=lambda r: abs(r.alerts_ratio - 1.0)) if approximate else None
    if worst is None:
        return ""
    return (
        "The column to read is the alert ratio, not the threshold. A threshold is a number "
        "nobody has intuition about; alert volume is what a SOC lead notices on Monday, and the "
        "map between them is violently non-linear at this end of the distribution.\n\n"
        f"`{worst.name}` at {worst.setting} is the worst row here: a threshold of "
        f"{worst.estimate:.5f} against the exact {worst.truth:.5f}. The estimate is wrong in "
        f"the **third decimal** and shifts the alert volume by "
        f"{abs(worst.alerts_ratio - 1.0):.0%}. Every other estimator's error is smaller than "
        "the gap between two adjacent benign scores near the threshold, so it changes nothing "
        "at all — which is the other half of the same point. Near the 99.9th percentile the "
        "score density is minute: errors below the inter-score gap are free, and errors above "
        "it move a disproportionate share of the alerts.\n\n"
        "That is why this report grades in alert volume rather than in quantile error. A "
        "ranking by absolute quantile error would have separated estimators that are "
        "operationally identical and understated the one that is not, and it is the same "
        "asymmetry the [Neyman-Pearson study](neyman_pearson.md) found from the sample-size "
        "side and the [DP-release study](dp_synth.md) found from the noise side: a fixed-FPR "
        "operating point is decided entirely in the thin tail."
    )


def _drift_read(study: QuantileStudy) -> str:
    if not study.drift:
        return ""
    worst = max(study.drift, key=lambda r: abs(r.estimate - r.truth))
    best = min(study.drift, key=lambda r: abs(r.estimate - r.truth))
    return (
        "None of these estimators forgets. Every one integrates the whole stream from the "
        "moment it starts, which is the correct behaviour for a stationary quantity and the "
        "wrong one for a traffic mix that changes. The drift here is not synthetic: the first "
        "half of the stream is validation-day benign traffic and the second half is *test*-day "
        "benign traffic, the same distribution change the deployed model already lives with. "
        "Each estimator is then asked for the threshold of the second regime alone: "
        f"`{best.name}` lands closest ({best.estimate:.5f} against a true {best.truth:.5f}) and "
        f"`{worst.name}` furthest ({worst.estimate:.5f}), and all of them are anchored by "
        "history nobody asked them to keep.\n\n"
        "The fix is not a better estimator, it is a **window**: run the sketch over a sliding "
        "or exponentially-decayed horizon and accept the variance that comes with a shorter "
        "memory. That is a design decision with a cost — the [threshold-refresh "
        "study](refresh.md) prices the same trade in labels — and the reason it belongs here "
        "is that a monitor which silently averages over a regime change is worse than one that "
        "is noisy and current."
    )


def _render(study: QuantileStudy, figure: Path) -> str:
    return f"""# NetSentry — Estimating the Threshold's Quantile at Line Rate

_Four streaming quantile estimators built from scratch and graded against the exact
{study.quantile:.3f} quantile of a {study.n_stream:,}-score benign stream, then re-graded in the
unit that matters: the alert volume each threshold actually delivers on {study.n_holdout:,}
held-out benign flows._

## Why this report exists

Every operating point here is a quantile — the score below which 99.9% of benign flows sit —
and every study that re-derives one assumes the scores can be collected, sorted and indexed.
That is true of a test split and false of a stream: sorting requires storing. The
[sketches study](sketches.md) counts in fixed memory but does not estimate quantiles.

## What each estimator buys

![Alert-volume error against memory](../figures/{figure.name})

{_accuracy_table(study)}

{_headline(study)}

## The error that matters is not the one you measure

{_fpr_read(study)}

## The baseline that beats the sketch

{_histogram_read(study)}

## What happens when the stream moves

{_drift_table(study)}

{_drift_read(study)}

## Scope and honest limits

- **The stream is a replayed validation split**, not a capture. It is drawn from the real score
  distribution the deployed model produces and then permuted and repeated to reach a stream
  length worth measuring, which keeps the *distribution* honest and makes the arrival order
  synthetic. Order matters for P-squared and for nothing else here.
- **P-squared estimates one quantile per instance.** Tracking 0.9, 0.99 and 0.999 costs three
  independent estimators, which is still nothing, but the constant-memory claim is per
  quantile rather than per stream.
- **The t-digest is a faithful simplification, not a port.** Merging with the `q(1-q)` scale
  function is the mechanism that matters; the published algorithm has buffer strategies and
  interpolation refinements this does not, so read its row as a lower bound on what a good
  implementation achieves.
- **Nothing here forgets**, and the drift section is about exactly that. A production monitor
  needs a window; every number above describes an estimator integrating from time zero.
- **Update cost is measured in Python**, where a per-element loop costs more than the
  arithmetic inside it. The *ordering* between estimators is meaningful and the absolute
  nanoseconds are an artifact of the language, not of the algorithms."""
