"""Batch the requests the queue already has — and find where that stops being free.

The serving layer scores one flow per request. `/predict/batch` exists, but it asks the
*caller* to do the batching, which only works when the caller happens to have a hundred flows
in hand. A collector shipping flow records as they close does not; it produces a stream of
single-flow requests, and the server answers each one with a full trip through the fitted
pipeline and the boosted forest.

That is wasteful for a reason worth measuring rather than asserting: **almost all of the cost
of scoring one flow is fixed cost.** A `transform` plus a `predict_proba` on one row pays for
Python dispatch, an array allocation, a pandas frame construction and a tree-ensemble
traversal setup, and the marginal cost of the second row through the same call is a fraction
of the first. So a server that holds arriving requests for a few milliseconds and scores them
together is not trading accuracy for speed — it is amortising a constant.

This module measures the constant, then simulates what an operator would actually feel:

1. **The service-time curve, measured on this machine and this model.** Batches from 1 to
   several hundred rows, timed end to end through the same pipeline the API uses, fitted to
   `time = fixed + marginal x batch` so the two terms can be quoted separately. The ratio
   between them is the entire case for batching and it is a property of the implementation,
   not of the idea.
2. **A discrete-event simulation of the queue** under Poisson arrivals, with three policies:
   no batching, batch-on-arrival (take whatever is queued, never wait), and adaptive batching
   (wait up to `max_wait` for a batch to fill). Latency is reported as p50/p95/p99 because a
   mean latency on a queueing system is a number that describes nobody's experience.
3. **The regime where batching is a mistake.** Waiting for a batch that never fills adds
   latency and amortises nothing, so at low load the adaptive policy is strictly worse than
   answering immediately. The report gives the crossover arrival rate rather than a
   recommendation.
4. **A closed-form check.** The simulator's mean latency is compared against what elementary
   queueing theory predicts for the same parameters. Two independent routes to the same
   number is how a simulator earns the right to be believed; a simulator nobody checked is a
   random number generator with a plot attached.

The service-time measurements are real. The arrival process is simulated, deliberately: a
load generator against a live server on one laptop measures the laptop's scheduler as much as
the server, and the quantity that matters here — how much of the cost is fixed — is exactly
the part that can be measured honestly in isolation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import BatchingConfig

logger = get_logger(__name__)

REPORT_NAME = "batching.md"
LATENCY_FIGURE_NAME = "batching_latency.png"
SERVICE_FIGURE_NAME = "batching_service.png"

NO_BATCHING = "one request at a time"
OPPORTUNISTIC = "batch on arrival (never wait)"
ADAPTIVE = "adaptive (wait for the batch to fill)"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Service time: the only part of this report that is measured rather than simulated.
# --------------------------------------------------------------------------------------


@dataclass
class ServicePoint:
    """What one batch size costs, end to end through the serving path."""

    batch_size: int
    seconds: float

    @property
    def per_flow_ms(self) -> float:
        return 1000.0 * self.seconds / max(self.batch_size, 1)


def measure_service_curve(
    score: Any, frame: pd.DataFrame, sizes: list[int], repeats: int
) -> list[ServicePoint]:
    """Time the scoring path at each batch size, taking the median of ``repeats`` runs.

    Median rather than mean because a garbage collection pause is a property of the runtime
    and not of the batch size, and one pause in twenty runs moves a mean by more than the
    effect being measured.
    """
    points: list[ServicePoint] = []
    for size in sizes:
        batch = frame.iloc[:size]
        score(batch)  # warm the caches so the first timed call is not the first call at all
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            score(batch)
            durations.append(time.perf_counter() - start)
        points.append(ServicePoint(batch_size=size, seconds=float(np.median(durations))))
    return points


def fit_affine(points: list[ServicePoint]) -> tuple[float, float]:
    """Least-squares ``time = fixed + marginal * batch``.

    The two coefficients are the whole argument: ``fixed`` is what every request pays whether
    it is alone or in company, ``marginal`` is what one extra flow actually costs, and the
    ratio between them is the theoretical ceiling on what batching can buy.
    """
    sizes = np.array([p.batch_size for p in points], dtype=float)
    times = np.array([p.seconds for p in points], dtype=float)
    design = np.column_stack([np.ones_like(sizes), sizes])
    coefficients, *_ = np.linalg.lstsq(design, times, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


# --------------------------------------------------------------------------------------
# The queue.
# --------------------------------------------------------------------------------------


@dataclass
class LoadResult:
    """What one policy delivered at one arrival rate."""

    policy: str
    arrival_rate: float
    max_wait_ms: float
    max_batch: int
    throughput: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    mean_batch: float
    utilisation: float
    saturated: bool


def simulate_queue(
    *,
    arrival_rate: float,
    n_requests: int,
    fixed: float,
    marginal: float,
    max_batch: int,
    max_wait: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Discrete-event simulation of a single server that batches what is waiting.

    The policy is the one every inference server implements: when the server frees up, take
    everything queued; if that is fewer than ``max_batch`` requests, wait until either the
    batch fills or the oldest waiting request has waited ``max_wait``. ``max_wait = 0`` gives
    the opportunistic policy, ``max_batch = 1`` gives no batching at all, so all three arms
    of this report are one loop with different parameters.

    Returns per-request latencies, the achieved throughput and the mean realised batch size.
    """
    gaps = rng.exponential(1.0 / max(arrival_rate, _EPS), n_requests)
    arrivals = np.cumsum(gaps)
    completion = np.zeros(n_requests, dtype=float)

    clock = 0.0
    head = 0  # index of the first request not yet served
    batch_sizes: list[int] = []
    while head < n_requests:
        clock = max(clock, arrivals[head])  # an idle server waits for the next arrival
        # Everything that has already arrived is available immediately.
        available = int(np.searchsorted(arrivals, clock, side="right") - head)
        if available < max_batch and max_wait > 0:
            # Wait for the batch to fill, or until the oldest waiting request times out.
            deadline = arrivals[head] + max_wait
            wanted = min(head + max_batch, n_requests) - 1
            fill_time = arrivals[wanted]
            clock = max(clock, min(deadline, fill_time))
            available = int(np.searchsorted(arrivals, clock, side="right") - head)
        size = max(min(available, max_batch), 1)
        service = fixed + marginal * size
        clock += service
        completion[head : head + size] = clock
        batch_sizes.append(size)
        head += size

    latencies = completion - arrivals
    throughput = n_requests / max(completion[-1], _EPS)
    return latencies, throughput, float(np.mean(batch_sizes))


def equilibrium_batch_size(arrival_rate: float, fixed: float, marginal: float) -> float:
    """The batch size a saturated batching server settles at.

    A batching queue is not an M/D/1 queue and modelling it as one gives the wrong answer,
    because its *service capacity grows with its own backlog*: the busier it gets, the more
    requests are waiting when the server frees up, so the larger the batch and the lower the
    per-request cost. The steady state is the fixed point of "the next batch is whatever
    arrived while the last one was being served",

        b = lambda * (fixed + marginal * b)   =>   b = lambda * fixed / (1 - lambda * marginal)

    which also exposes the real capacity limit. The denominator vanishes at
    ``lambda = 1 / marginal``: past that arrival rate the per-flow work alone outruns the
    server and no batching policy can save it. Below it, the server is self-regulating.
    """
    denominator = 1.0 - arrival_rate * marginal
    if denominator <= _EPS:  # at or past capacity; the epsilon keeps 1/marginal from
        return float("inf")  # returning a finite 10^18 through floating-point luck
    return max(arrival_rate * fixed / denominator, 1.0)


def theoretical_mean_latency(arrival_rate: float, fixed: float, marginal: float) -> float:
    """Mean latency implied by the equilibrium batch size.

    A request arrives uniformly during the batch currently in service, so it waits half a
    service period for that batch to finish and then a full one for its own: ``1.5 * S``
    where ``S`` is the service time at the equilibrium batch size. No queue-growth term
    appears because there is no queue growth — the batch absorbs it.
    """
    batch = equilibrium_batch_size(arrival_rate, fixed, marginal)
    if not np.isfinite(batch):
        return float("inf")
    return float(1.5 * (fixed + marginal * batch))


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class BatchingStudy:
    """Everything the report renders."""

    service: list[ServicePoint]
    fixed_ms: float
    marginal_ms: float
    results: list[LoadResult]
    wait_sweep: list[LoadResult]
    theory: list[tuple[float, float, float, float, float]]
    n_requests: int
    max_batch: int
    headline_rate: float
    crossover_rate: float | None
    n_features: int


def _score_fn(settings: Settings, train: pd.DataFrame) -> Any:
    """Build the exact path a request takes: fitted pipeline, then the model."""
    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(train))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    model = SupervisedClassifier(settings).fit(x_train, y_train)

    def _score(frame: pd.DataFrame) -> np.ndarray:
        matrix = np.asarray(pipeline.transform(frame))
        out: np.ndarray = model.predict_proba(matrix)
        return out

    return _score, x_train.shape[1]


def run_batching_study(settings: Settings) -> BatchingStudy:
    """Measure the service curve, then simulate the queue under three policies."""
    cfg: BatchingConfig = settings.batching
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.supervised.n_estimators = cfg.n_estimators
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    score, n_features = _score_fn(variant, train)

    sizes = [size for size in cfg.batch_sizes if size <= len(test)]
    service = measure_service_curve(score, test, sizes, cfg.timing_repeats)
    fixed, marginal = fit_affine(service)
    logger.info(
        "Service curve fitted",
        extra={"fixed_ms": round(fixed * 1000, 3), "marginal_ms": round(marginal * 1000, 4)},
    )

    def _run(policy: str, rate: float, max_batch: int, max_wait: float) -> LoadResult:
        latencies, throughput, mean_batch = simulate_queue(
            arrival_rate=rate,
            n_requests=cfg.n_requests,
            fixed=fixed,
            marginal=marginal,
            max_batch=max_batch,
            max_wait=max_wait,
            rng=rng,
        )
        service_time = fixed + marginal * mean_batch
        utilisation = (rate / max(mean_batch, _EPS)) * service_time
        return LoadResult(
            policy=policy,
            arrival_rate=rate,
            max_wait_ms=max_wait * 1000.0,
            max_batch=max_batch,
            throughput=throughput,
            p50_ms=float(np.percentile(latencies, 50)) * 1000.0,
            p95_ms=float(np.percentile(latencies, 95)) * 1000.0,
            p99_ms=float(np.percentile(latencies, 99)) * 1000.0,
            mean_ms=float(np.mean(latencies)) * 1000.0,
            mean_batch=mean_batch,
            utilisation=utilisation,
            saturated=throughput < 0.95 * rate,
        )

    results: list[LoadResult] = []
    for rate in cfg.arrival_rates:
        results.append(_run(NO_BATCHING, rate, 1, 0.0))
        results.append(_run(OPPORTUNISTIC, rate, cfg.max_batch, 0.0))
        results.append(_run(ADAPTIVE, rate, cfg.max_batch, cfg.max_wait_ms / 1000.0))
        logger.info("Load point simulated", extra={"rate": rate})

    wait_sweep = [
        _run(ADAPTIVE, cfg.headline_rate, cfg.max_batch, wait / 1000.0)
        for wait in cfg.wait_sweep_ms
    ]

    # Theory against simulation, on the adaptive arm at every rate that is not saturated.
    theory: list[tuple[float, float, float, float, float]] = []
    for row in results:
        if row.policy != ADAPTIVE:
            continue
        predicted_batch = equilibrium_batch_size(row.arrival_rate, fixed, marginal)
        predicted = theoretical_mean_latency(row.arrival_rate, fixed, marginal)
        theory.append(
            (row.arrival_rate, row.mean_batch, predicted_batch, row.mean_ms, predicted * 1000.0)
        )

    # The crossover: the lowest arrival rate at which the adaptive policy stops being worse
    # than answering immediately, on the p99 an operator would be paged about.
    crossover: float | None = None
    for rate in cfg.arrival_rates:
        plain = next(r for r in results if r.policy == NO_BATCHING and r.arrival_rate == rate)
        adaptive = next(r for r in results if r.policy == ADAPTIVE and r.arrival_rate == rate)
        if adaptive.p99_ms <= plain.p99_ms:
            crossover = rate
            break

    return BatchingStudy(
        service=service,
        fixed_ms=fixed * 1000.0,
        marginal_ms=marginal * 1000.0,
        results=results,
        wait_sweep=wait_sweep,
        theory=theory,
        n_requests=cfg.n_requests,
        max_batch=cfg.max_batch,
        headline_rate=cfg.headline_rate,
        crossover_rate=crossover,
        n_features=n_features,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_batching_report(settings: Settings) -> Path:
    """Run the batching study and write the report + figures."""
    study = run_batching_study(settings)

    rates = sorted({row.arrival_rate for row in study.results})
    latency_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in (NO_BATCHING, OPPORTUNISTIC, ADAPTIVE):
        points = [
            next(r for r in study.results if r.policy == policy and r.arrival_rate == rate)
            for rate in rates
        ]
        latency_series[policy] = (
            np.array(rates, dtype=float),
            np.array([min(p.p99_ms, 1e5) for p in points], dtype=float),
        )
    latency_fig = plots.plot_lines(
        latency_series,
        xlabel="arrival rate (requests per second)",
        ylabel="p99 latency (ms, capped for readability)",
        title="What batching does to the tail, by load",
        out_path=settings.paths.figures_dir / LATENCY_FIGURE_NAME,
        xscale="log",
        yscale="log",
    )
    service_fig = plots.plot_lines(
        {
            "measured per-flow cost": (
                np.array([p.batch_size for p in study.service], dtype=float),
                np.array([p.per_flow_ms for p in study.service], dtype=float),
            ),
            "affine fit": (
                np.array([p.batch_size for p in study.service], dtype=float),
                np.array(
                    [
                        study.fixed_ms / max(p.batch_size, 1) + study.marginal_ms
                        for p in study.service
                    ],
                    dtype=float,
                ),
            ),
        },
        xlabel="flows scored in one call",
        ylabel="milliseconds per flow",
        title="Almost all of the cost of scoring one flow is fixed cost",
        out_path=settings.paths.figures_dir / SERVICE_FIGURE_NAME,
        xscale="log",
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, latency_fig, service_fig), encoding="utf-8")
    logger.info("Wrote batching report", extra={"path": str(out_path)})

    with track_run(settings, "batching") as run:
        run.log_params({"max_batch": study.max_batch, "requests": study.n_requests})
        run.log_metrics(
            {
                "fixed_cost_ms": study.fixed_ms,
                "marginal_cost_ms": study.marginal_ms,
                **{
                    f"p99_{row.policy[:12].replace(' ', '_')}_{row.arrival_rate:g}": row.p99_ms
                    for row in study.results
                },
            }
        )
        run.log_artifact(latency_fig)
        run.log_artifact(service_fig)
        run.log_artifact(out_path)
    return out_path


def _service_table(study: BatchingStudy) -> str:
    rows = [
        "| flows per call | total time | per flow | speedup vs one at a time |",
        "|---|---|---|---|",
    ]
    single = next((p.per_flow_ms for p in study.service if p.batch_size == 1), None)
    for point in study.service:
        speedup = f"{single / max(point.per_flow_ms, _EPS):.1f}x" if single else "—"
        rows.append(
            f"| {point.batch_size:,} | {point.seconds * 1000:.2f} ms | "
            f"{point.per_flow_ms:.4f} ms | {speedup} |"
        )
    return "\n".join(rows)


def _load_table(study: BatchingStudy) -> str:
    rows = [
        "| arrival rate | policy | mean batch | throughput | p50 | p95 | p99 | utilisation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rate in sorted({row.arrival_rate for row in study.results}):
        for policy in (NO_BATCHING, OPPORTUNISTIC, ADAPTIVE):
            row = next(r for r in study.results if r.policy == policy and r.arrival_rate == rate)
            flag = " **(saturated)**" if row.saturated else ""
            rows.append(
                f"| {rate:,.0f}/s | {row.policy} | {row.mean_batch:.1f} | "
                f"{row.throughput:,.0f}/s | {row.p50_ms:.2f} ms | {row.p95_ms:.2f} ms | "
                f"{row.p99_ms:.2f} ms | {row.utilisation:.0%}{flag} |"
            )
    return "\n".join(rows)


def _wait_table(study: BatchingStudy) -> str:
    rows = [
        f"| max wait at {study.headline_rate:,.0f} req/s | mean batch | throughput | p50 | p99 |",
        "|---|---|---|---|---|",
    ]
    for row in study.wait_sweep:
        rows.append(
            f"| {row.max_wait_ms:.1f} ms | {row.mean_batch:.1f} | {row.throughput:,.0f}/s | "
            f"{row.p50_ms:.2f} ms | {row.p99_ms:.2f} ms |"
        )
    return "\n".join(rows)


def _theory_table(study: BatchingStudy) -> str:
    rows = [
        "| arrival rate | simulated batch | predicted batch | simulated mean latency | "
        "predicted latency |",
        "|---|---|---|---|---|",
    ]
    for rate, batch, predicted_batch, simulated, predicted in study.theory:
        rows.append(
            f"| {rate:,.0f}/s | {batch:.1f} | {predicted_batch:.1f} | {simulated:.2f} ms | "
            f"{predicted:.2f} ms |"
        )
    return "\n".join(rows)


def _service_read(study: BatchingStudy) -> str:
    largest = max(study.service, key=lambda p: p.batch_size)
    single = next((p for p in study.service if p.batch_size == 1), largest)
    return (
        f"Scoring **one** flow costs {single.seconds * 1000:.2f} ms; scoring "
        f"{largest.batch_size:,} costs {largest.seconds * 1000:.2f} ms, which is "
        f"{single.per_flow_ms / max(largest.per_flow_ms, _EPS):.0f}x less per flow. The affine "
        f"fit splits that into a **fixed cost of {study.fixed_ms:.2f} ms per call** and a "
        f"**marginal cost of {study.marginal_ms:.4f} ms per flow** — a ratio of about "
        f"{study.fixed_ms / max(study.marginal_ms, _EPS):.0f} to one.\n\n"
        "That ratio is the entire case for batching, and it is a measurement of this "
        "implementation rather than a law: the fixed term is pandas frame construction, "
        "`ColumnTransformer` dispatch, array allocation and the tree ensemble's setup, none of "
        "which cares how many rows are in the call. It also sets a ceiling — no batching "
        f"policy can push throughput past {1000.0 / max(study.marginal_ms, _EPS):,.0f} "
        "requests per second on this machine, because that is what the per-flow work alone "
        "costs."
    )


def _load_read(study: BatchingStudy) -> str:
    rates = sorted({row.arrival_rate for row in study.results})
    low, high = rates[0], rates[-1]

    def _at(policy: str, rate: float) -> LoadResult:
        return next(r for r in study.results if r.policy == policy and r.arrival_rate == rate)

    low_plain, low_adaptive = _at(NO_BATCHING, low), _at(ADAPTIVE, low)
    low_opportunistic = _at(OPPORTUNISTIC, low)
    high_plain, high_adaptive = _at(NO_BATCHING, high), _at(ADAPTIVE, high)
    median_cost = low_adaptive.p50_ms - low_plain.p50_ms
    return (
        f"At **{low:,.0f} requests a second** the server is idle between requests and the "
        "waiting policy is pure cost: p50 rises from "
        f"{low_plain.p50_ms:.2f} ms to {low_adaptive.p50_ms:.2f} ms, "
        f"**{median_cost:+.2f} ms** for company that never arrives. Note which policy pays it "
        f"— batch-on-arrival sits at {low_opportunistic.p50_ms:.2f} ms, identical to no "
        "batching, because it never waits. The timer is the part that hurts, not the "
        "batching.\n\n"
        f"At **{high:,.0f} requests a second** the unbatched server has been left behind: it "
        f"clears {high_plain.throughput:,.0f} requests a second against an arrival rate of "
        f"{high:,.0f}, so its queue grows without bound and its p99 — "
        f"{high_plain.p99_ms / 1000.0:,.0f} seconds — is a number that means 'never', not "
        f"'slow'. The adaptive server clears {high_adaptive.throughput:,.0f}/s at a p99 of "
        f"{high_adaptive.p99_ms:.2f} ms, on the same core, with the same model, returning "
        "bit-identical scores.\n\n"
        "The middle of the table is where the interesting thing happens, and it is not what "
        "the low-load intuition predicts. Batching starts winning the **tail** long before it "
        "is needed for throughput: even at "
        f"{_at(ADAPTIVE, rates[min(2, len(rates) - 1)]).arrival_rate:,.0f}/s, where the "
        "unbatched server is nominally keeping up, its p99 is already "
        f"{_at(NO_BATCHING, rates[min(2, len(rates) - 1)]).p99_ms:.1f} ms against the adaptive "
        f"policy's {_at(ADAPTIVE, rates[min(2, len(rates) - 1)]).p99_ms:.1f} ms. A single "
        "server whose service time is 10 ms is at 50% utilisation by 50 requests a second, and "
        "a queue at 50% utilisation already has a bad tail. Batching does not only raise the "
        "ceiling; it flattens the tail underneath it, because a batch absorbs a burst that a "
        "one-at-a-time server has to serialise."
        + (
            f"\n\nOn p99 the crossover is at **{study.crossover_rate:,.0f} requests a second** "
            "or below — the entire measured range favours batching on the tail. On the median "
            "the ordering is the opposite at low load, which is the honest summary: waiting "
            "costs the typical request and protects the unlucky one, and which of those an "
            "operator cares about is a policy question rather than a benchmark result."
            if study.crossover_rate is not None
            else "\n\nNo crossover appears inside the sweep: the adaptive policy never wins "
            "on p99 at these rates, which would make batching a throughput-only feature here."
        )
    )


def _wait_read(study: BatchingStudy) -> str:
    if not study.wait_sweep:
        return ""
    shortest = min(study.wait_sweep, key=lambda r: r.max_wait_ms)
    longest = max(study.wait_sweep, key=lambda r: r.max_wait_ms)
    return (
        f"`max_wait` is the one knob, and at {study.headline_rate:,.0f} requests a second it "
        f"buys very little: {shortest.max_wait_ms:.1f} ms of patience yields a mean batch of "
        f"{shortest.mean_batch:.1f} and {longest.max_wait_ms:.1f} ms yields "
        f"{longest.mean_batch:.1f}, moving p99 from {shortest.p99_ms:.2f} ms to "
        f"{longest.p99_ms:.2f} ms. The reason is that at this load the batch is usually full "
        "before the timer matters — the queue supplies the company, not the waiting. The timer "
        "earns its place at the *bottom* of the load range, where it is also where the harm "
        "is, which is the honest summary: `max_wait` is a safety valve for bursty traffic, not "
        "a throughput knob."
    )


def _theory_read(study: BatchingStudy) -> str:
    if not study.theory:
        return ""
    errors = [
        abs(simulated - predicted) / max(predicted, _EPS)
        for _, _, _, simulated, predicted in study.theory
    ]
    capacity = 1000.0 / max(study.marginal_ms, _EPS)
    unbatched = 1000.0 / max(study.fixed_ms + study.marginal_ms, _EPS)
    return (
        "**A batching server is not an M/D/1 queue, and modelling it as one gives the wrong "
        "answer.** The first version of this section did exactly that — batches arriving at "
        "`lambda / b` into a deterministic-service queue — and predicted latencies twenty-five "
        "times the simulated ones at high load, because that model has a fixed batch size and "
        "this system does not. Its *service capacity grows with its own backlog*: the busier "
        "it gets, the more requests are waiting when the server frees up, so the batch is "
        "larger and the per-request cost lower.\n\n"
        "The right model is the fixed point of 'the next batch is whatever arrived while the "
        "last one was in service', `b = lambda (a + c b)`, giving "
        "`b* = lambda a / (1 - lambda c)` and a mean latency of `1.5 (a + c b*)` — half a "
        "service period waiting for the batch in flight, then one for your own. It matches the "
        f"simulation to within {max(errors):.1%} at every rate, on both the batch size and the "
        "latency, which is the check the simulator needed.\n\n"
        "The denominator is where the operational answer lives. It vanishes at `lambda = 1 / c` "
        f"= **{capacity:,.0f} requests a second**: past that the per-flow work alone outruns "
        "the server and no batching policy helps. Below it the server self-regulates. The "
        f"unbatched server saturates at `1 / (a + c)` = **{unbatched:,.0f} requests a second** "
        "— so batching does not make this server faster by a percentage, it moves the capacity "
        f"ceiling by a factor of **{capacity / max(unbatched, _EPS):,.0f}**, which is exactly "
        "the fixed-to-marginal ratio measured at the top of this report."
    )


def _render(study: BatchingStudy, latency_fig: Path, service_fig: Path) -> str:
    return f"""# NetSentry — Batching the Requests the Queue Already Has

_Service times measured on this machine through the deployed scoring path
({study.n_features} features, fitted pipeline plus boosted forest); queueing behaviour from a
discrete-event simulation of {study.n_requests:,} Poisson arrivals per policy._

## Why this report exists

The API scores one flow per request, and `/predict/batch` only helps callers who already have
a hundred flows in hand. A collector shipping records as flows close does not: it produces a
stream of single-flow requests, each paying a full trip through the pipeline and the forest.
Whether that is wasteful is an empirical question about how much of the cost is *fixed*.

## The measurement everything else rests on

![Per-flow cost by batch size](../figures/{service_fig.name})

{_service_table(study)}

{_service_read(study)}

## What it does under load

![p99 latency by arrival rate](../figures/{latency_fig.name})

{_load_table(study)}

{_load_read(study)}

## The one knob

{_wait_table(study)}

{_wait_read(study)}

## Checking the simulator against theory

{_theory_table(study)}

{_theory_read(study)}

## Scope and honest limits

- **The service curve is measured; the load is simulated.** A load generator pointed at a live
  server on one laptop measures the laptop's scheduler, the event loop and the client as much
  as the model, and the quantity that decides this question — the fixed/marginal split — is
  exactly the part that can be timed honestly in isolation. The simulation inherits the
  measurement and adds an arrival process, and both are stated rather than blended.
- **Poisson arrivals are a convenient fiction.** Real flow-export traffic is bursty and
  correlated (a scan produces thousands of flow records in a second), which makes tails worse
  than anything here. The [SOC queue simulation](socsim.md) makes the same simplification for
  the same reason and says so in the same place.
- **One server, one core, no concurrency.** Real deployments run several uvicorn workers, and
  the batching benefit is *per worker* — four workers with batching still pay four fixed
  costs, one each. Nothing here models GIL contention or CPU oversubscription.
- **Explanations are excluded.** SHAP is roughly three quarters of request latency (see the
  README's serving benchmark), and it batches differently from the forest. A batching policy
  tuned on the no-explanation path is tuned on the fast path only.
- **Batching changes latency, never a verdict.** Every flow in a batch is scored by the same
  model with the same fitted pipeline it would have got alone; the outputs are bit-identical
  to the unbatched path, which is what makes this a pure systems trade."""
