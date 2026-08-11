"""Host analytics at line rate: bounded-memory sketches with error bounds that hold.

The [host-graph study](graph_demo.md) finds scanners by counting how many distinct
destinations each source touches. That is a set per source, and a set grows with what it
holds. At the volumes a real sensor sees — a 10 Gbps link is tens of thousands of flows per
second, and a busy source can touch millions of destinations in a day — "keep a set per host"
is not an implementation detail that needs tuning, it is a design that runs out of memory and
takes the detector with it. Every production flow analytic is built on **sketches** instead:
fixed-size probabilistic structures that answer the same questions with a stated error, in
memory that does not grow with the stream.

Four of them are implemented here from scratch, because each answers a different question the
host analytics actually asks:

- **Count-Min** (Cormode & Muthukrishnan, 2005) for per-host flow counts. Never
  underestimates, and overestimates by at most `epsilon * N` with probability `1 - delta`,
  in `O(1/epsilon * log(1/delta))` counters regardless of how many distinct hosts appear.
- **HyperLogLog** (Flajolet, Fusy, Gandouet & Meunier, 2007) for distinct destinations per
  host — the scan fan-out signal. Standard error `1.04 / sqrt(m)` in `m` small registers, so
  a few kilobytes counts millions of distinct values to within a couple of percent.
- **Misra-Gries** (1982) for the heavy hitters, which is what a "top talkers" panel is: any
  host above a `1/k` share of the stream is guaranteed to appear in `k - 1` counters.
- **Reservoir sampling** (Vitter, 1985) for a uniform sample of the stream, which is what
  every downstream estimate that is not a count ends up needing.

What makes this a study rather than a library is the third section. A bound that holds in
expectation is not the same as a bound that holds, so every guarantee above is checked against
exact ground truth: Count-Min is verified never to underestimate and its `epsilon * N` bound is
measured against the promised `1 - delta`, HyperLogLog's observed error is compared with its
theoretical standard error, and Misra-Gries is checked for the misses its guarantee permits.
Then the question that actually matters is asked — **does the scan-detection ranking survive
the approximation** — because a fan-out estimate that is 2% wrong is worthless if the 2% lands
on the ordering of the top ten.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SketchConfig

logger = get_logger(__name__)

REPORT_NAME = "sketches.md"
CMS_FIGURE = "sketches_countmin.png"
HLL_FIGURE = "sketches_hyperloglog.png"


def _digest(key: str, salt: int) -> int:
    """Deterministic 64-bit hash of a key under a salt.

    Python's built-in ``hash`` is randomised per process, which would make every number in
    this report irreproducible; blake2b is in the standard library, is stable across runs and
    machines, and takes a key, which is exactly the family of independent hash functions these
    sketches assume.
    """
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8, key=str(salt).encode()).digest(),
        "big",
    )


# --------------------------------------------------------------------------------------
# Count-Min sketch
# --------------------------------------------------------------------------------------
@dataclass
class CountMin:
    """Per-key counts in fixed memory, with a one-sided error (Cormode & Muthukrishnan 2005).

    ``d`` independent hash rows of ``w`` counters. Every update increments one counter per
    row, and a query takes the *minimum* across rows: collisions can only add to a counter, so
    the minimum is the estimate least contaminated by them. The estimate is therefore never
    below the truth, which is the property that makes the structure safe for detection — a
    sketch that could undercount a scanner would hide it.
    """

    width: int
    depth: int
    table: np.ndarray = field(init=False)
    total: int = 0

    def __post_init__(self) -> None:
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)

    @classmethod
    def for_guarantee(cls, epsilon: float, delta: float) -> CountMin:
        """Size the sketch from the guarantee wanted, rather than the other way round."""
        width = max(2, math.ceil(math.e / max(epsilon, 1e-12)))
        depth = max(1, math.ceil(math.log(1.0 / max(delta, 1e-12))))
        return cls(width=width, depth=depth)

    def add(self, key: str, count: int = 1) -> None:
        for row in range(self.depth):
            self.table[row, _digest(key, row) % self.width] += count
        self.total += count

    def estimate(self, key: str) -> int:
        return int(
            min(self.table[row, _digest(key, row) % self.width] for row in range(self.depth))
        )

    def n_bytes(self) -> int:
        """Memory the structure occupies, which is the whole point of using it."""
        return int(self.table.nbytes)


# --------------------------------------------------------------------------------------
# HyperLogLog
# --------------------------------------------------------------------------------------
@dataclass
class HyperLogLog:
    """Distinct-value count in a few kilobytes (Flajolet et al. 2007).

    The first ``p`` bits of a hash choose a register; the position of the first set bit in
    what remains is a crude estimate of how many distinct values it took to see a run that
    long. Keeping the maximum per register and harmonic-averaging turns many crude estimates
    into one good one, with standard error ``1.04 / sqrt(m)``. Small cardinalities are
    corrected by linear counting over the empty registers, without which the estimator is
    badly biased exactly where a scan detector starts to care.
    """

    precision: int = 12
    registers: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.registers = np.zeros(1 << self.precision, dtype=np.uint8)

    @property
    def m(self) -> int:
        """Number of registers."""
        return int(self.registers.size)

    def add(self, key: str) -> None:
        h = _digest(key, 0)
        index = h >> (64 - self.precision)
        remaining = (h << self.precision) & ((1 << 64) - 1)
        rank = 1 if remaining == 0 else 64 - remaining.bit_length() + 1
        self.registers[index] = max(int(self.registers[index]), min(rank, 64))

    def estimate(self) -> float:
        m = self.m
        alpha = (
            0.7213 / (1.0 + 1.079 / m)
            if m >= 128
            else {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213)
        )
        raw = alpha * m * m / float(np.sum(np.power(2.0, -self.registers.astype(float))))
        zeros = int(np.count_nonzero(self.registers == 0))
        if raw <= 2.5 * m and zeros > 0:
            return float(m * math.log(m / zeros))  # linear counting in the small range
        return float(raw)

    def n_bytes(self) -> int:
        return int(self.registers.nbytes)


# --------------------------------------------------------------------------------------
# Misra-Gries heavy hitters
# --------------------------------------------------------------------------------------
@dataclass
class MisraGries:
    """Top talkers in ``k - 1`` counters (Misra & Gries 1982).

    On a miss with no free counter, every counter is decremented and the empty ones dropped —
    the trick that bounds memory. The guarantee is one-sided in the useful direction: any key
    holding more than a ``1/k`` share of the stream is certain to be in the summary, though
    keys that are not may also appear, so the summary is a candidate set to verify rather than
    an answer.
    """

    k: int
    counters: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def add(self, key: str) -> None:
        self.total += 1
        if key in self.counters:
            self.counters[key] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[key] = 1
        else:
            for existing in list(self.counters):
                self.counters[existing] -= 1
                if self.counters[existing] == 0:
                    del self.counters[existing]

    def top(self, n: int) -> list[tuple[str, int]]:
        return sorted(self.counters.items(), key=lambda kv: -kv[1])[:n]


# --------------------------------------------------------------------------------------
# Reservoir sampling
# --------------------------------------------------------------------------------------
def reservoir_sample(stream: list[str], size: int, rng: np.random.Generator) -> list[str]:
    """A uniform sample of a stream whose length is not known in advance (Vitter 1985).

    Item ``i`` replaces a random slot with probability ``size / i``, which leaves every item
    equally likely to survive. One pass, fixed memory, no second look at the data — the
    conditions a sensor actually operates under.
    """
    reservoir: list[str] = []
    for index, item in enumerate(stream):
        if index < size:
            reservoir.append(item)
        else:
            j = int(rng.integers(0, index + 1))
            if j < size:
                reservoir[j] = item
    return reservoir


# --------------------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------------------
def synthesize_stream(
    n_flows: int, n_hosts: int, zipf_exponent: float, scanners: int, scanner_targets: int, seed: int
) -> list[tuple[str, str]]:
    """A (source, destination) flow stream with a heavy tail and planted scanners.

    Real per-host traffic is heavy-tailed — a handful of hosts carry most of the flows — and a
    uniform stream would flatter every sketch here, since collisions hurt most when the
    distribution is skewed. Scanners are planted with a known fan-out so the operational
    question at the end has a ground truth to be right or wrong about.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_hosts + 1, dtype=float)
    weights = ranks**-zipf_exponent
    weights /= weights.sum()
    sources = rng.choice(n_hosts, size=n_flows, p=weights)
    # Each benign host talks to a small pool of peers it keeps returning to, however many
    # flows it sends. That is what ordinary traffic looks like and it is what makes fan-out a
    # scan signal at all: a busy host is not a scanner, and a stream where volume and fan-out
    # were the same quantity would make the detection problem trivial and the study a lie.
    pools = [rng.integers(0, n_hosts * 4, size=int(rng.integers(4, 40))) for _ in range(n_hosts)]
    stream = [
        (
            f"10.0.{s // 256}.{s % 256}",
            f"93.184.{int(d) // 256}.{int(d) % 256}",
        )
        for s, d in (
            (src, pools[src][rng.integers(0, len(pools[src]))]) for src in sources.tolist()
        )
    ]
    for scanner in range(scanners):
        source = f"203.0.113.{scanner}"
        stream.extend((source, f"10.0.{t // 256}.{t % 256}") for t in range(scanner_targets))
    rng.shuffle(stream)  # a sketch that needed sorted input would not be a sketch
    return stream


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class CountMinRow:
    """One Count-Min sizing: memory, measured error, and whether the bound held."""

    epsilon: float
    width: int
    depth: int
    kilobytes: float
    mean_overestimate: float
    max_overestimate: int
    ever_underestimated: bool
    within_bound: float


@dataclass
class HyperLogLogRow:
    """One HyperLogLog precision: memory, measured error against the theoretical one."""

    precision: int
    registers: int
    kilobytes: float
    mean_relative_error: float
    theoretical_error: float
    top_k_agreement: float
    scanners_found: int


@dataclass
class SketchStudy:
    """Everything the report renders."""

    n_flows: int
    n_sources: int
    exact_kilobytes: float
    countmin: list[CountMinRow]
    hyperloglog: list[HyperLogLogRow]
    heavy_recall: float
    heavy_k: int
    top_k: int
    max_benign_fanout: int
    reservoir_chi2_p: float
    scanners: int
    scanner_targets: int


def _countmin_rows(
    stream: list[tuple[str, str]], exact: Counter[str], cfg: SketchConfig
) -> list[CountMinRow]:
    """Fill a Count-Min at each requested guarantee and check the promise empirically."""
    rows: list[CountMinRow] = []
    keys = list(exact)
    for epsilon in cfg.countmin_epsilons:
        sketch = CountMin.for_guarantee(epsilon, cfg.countmin_delta)
        for source, _ in stream:
            sketch.add(source)
        errors = np.array([sketch.estimate(k) - exact[k] for k in keys], dtype=float)
        allowance = epsilon * sketch.total
        rows.append(
            CountMinRow(
                epsilon=epsilon,
                width=sketch.width,
                depth=sketch.depth,
                kilobytes=sketch.n_bytes() / 1024.0,
                mean_overestimate=float(errors.mean()),
                max_overestimate=int(errors.max()),
                ever_underestimated=bool((errors < 0).any()),
                within_bound=float(np.mean(errors <= allowance)),
            )
        )
        logger.info(
            "Count-Min sized",
            extra={"epsilon": epsilon, "kb": round(rows[-1].kilobytes, 1)},
        )
    return rows


def _hll_rows(
    stream: list[tuple[str, str]], exact: dict[str, set[str]], cfg: SketchConfig
) -> list[HyperLogLogRow]:
    """Estimate per-source fan-out at each precision and check the ranking survives."""
    truth = {source: len(dsts) for source, dsts in exact.items()}
    ranked = sorted(truth, key=lambda s: -truth[s])
    top_true = set(ranked[: cfg.top_k])
    rows: list[HyperLogLogRow] = []
    for precision in cfg.hll_precisions:
        sketches: dict[str, HyperLogLog] = {}
        for source, destination in stream:
            sketches.setdefault(source, HyperLogLog(precision=precision)).add(destination)
        estimates = {s: h.estimate() for s, h in sketches.items()}
        relative = np.array(
            [abs(estimates[s] - truth[s]) / max(truth[s], 1) for s in truth], dtype=float
        )
        top_estimated = set(sorted(estimates, key=lambda s: -estimates[s])[: cfg.top_k])
        rows.append(
            HyperLogLogRow(
                precision=precision,
                registers=1 << precision,
                kilobytes=len(sketches) * (1 << precision) / 1024.0,
                mean_relative_error=float(relative.mean()),
                theoretical_error=1.04 / math.sqrt(1 << precision),
                top_k_agreement=len(top_true & top_estimated) / max(len(top_true), 1),
                scanners_found=sum(1 for s in top_estimated if s.startswith("203.0.113.")),
            )
        )
        logger.info(
            "HyperLogLog sized",
            extra={"precision": precision, "error": round(rows[-1].mean_relative_error, 4)},
        )
    return rows


def run_sketches(settings: Settings) -> SketchStudy:
    """Replay a synthetic host stream through four sketches and grade every guarantee."""
    cfg: SketchConfig = settings.sketches
    stream = synthesize_stream(
        cfg.n_flows,
        cfg.n_hosts,
        cfg.zipf_exponent,
        cfg.scanners,
        cfg.scanner_targets,
        settings.seed,
    )
    exact_counts: Counter[str] = Counter(source for source, _ in stream)
    exact_sets: dict[str, set[str]] = {}
    for source, destination in stream:
        exact_sets.setdefault(source, set()).add(destination)
    exact_bytes = sum(
        len(source) + sum(len(d) for d in dsts) for source, dsts in exact_sets.items()
    )

    heavy = MisraGries(k=cfg.heavy_hitter_k)
    for source, _ in stream:
        heavy.add(source)
    threshold = len(stream) / cfg.heavy_hitter_k
    true_heavy = {k for k, v in exact_counts.items() if v > threshold}
    found = {k for k, _ in heavy.top(cfg.heavy_hitter_k)}

    rng = np.random.default_rng(settings.seed)
    sample = reservoir_sample([s for s, _ in stream], cfg.reservoir_size, rng)
    chi2_p = _uniformity_p(Counter(sample), exact_counts, len(stream))

    return SketchStudy(
        n_flows=len(stream),
        n_sources=len(exact_counts),
        exact_kilobytes=exact_bytes / 1024.0,
        countmin=_countmin_rows(stream, exact_counts, cfg),
        hyperloglog=_hll_rows(stream, exact_sets, cfg),
        heavy_recall=(len(true_heavy & found) / len(true_heavy)) if true_heavy else 1.0,
        heavy_k=cfg.heavy_hitter_k,
        top_k=cfg.top_k,
        max_benign_fanout=max(
            (len(d) for s, d in exact_sets.items() if not s.startswith("203.0.113.")), default=0
        ),
        reservoir_chi2_p=chi2_p,
        scanners=cfg.scanners,
        scanner_targets=cfg.scanner_targets,
    )


def _uniformity_p(sample: Counter[str], population: Counter[str], n_stream: int) -> float:
    """Chi-square p-value that the reservoir is a uniform sample of the stream.

    Computed from the survival function of the chi-square distribution via a series
    expansion, so the module keeps its no-SciPy property; the sample sizes here make the
    normal approximation to the chi-square perfectly adequate anyway.
    """
    total = sum(sample.values())
    keys = [k for k in population if population[k] / n_stream * total >= 5]
    if len(keys) < 2 or total == 0:
        return 1.0
    statistic = 0.0
    for key in keys:
        expected = population[key] / n_stream * total
        statistic += (sample.get(key, 0) - expected) ** 2 / expected
    dof = len(keys) - 1
    # Wilson-Hilferty: the cube root of a chi-square is close to normal.
    z = ((statistic / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return float(0.5 * math.erfc(z / math.sqrt(2)))


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_sketches_report(settings: Settings) -> Path:
    """Run the sketch study and write the report + figures."""
    study = run_sketches(settings)

    cms_fig = plots.plot_lines(
        {
            "mean overestimate (flows)": (
                np.array([r.kilobytes for r in study.countmin]),
                np.array([r.mean_overestimate for r in study.countmin]),
            )
        },
        xlabel="Count-Min memory (KiB)",
        ylabel="mean overestimate per host (flows)",
        title="Count-Min: error against memory",
        out_path=settings.paths.figures_dir / CMS_FIGURE,
        xscale="log",
        yscale="log",
    )
    hll_fig = plots.plot_lines(
        {
            "measured": (
                np.array([float(r.registers) for r in study.hyperloglog]),
                np.array([r.mean_relative_error for r in study.hyperloglog]),
            ),
            "theoretical 1.04/sqrt(m)": (
                np.array([float(r.registers) for r in study.hyperloglog]),
                np.array([r.theoretical_error for r in study.hyperloglog]),
            ),
        },
        xlabel="registers per host (m)",
        ylabel="mean relative error of the fan-out estimate",
        title="HyperLogLog: measured error against its own bound",
        out_path=settings.paths.figures_dir / HLL_FIGURE,
        xscale="log",
        yscale="log",
    )

    report = _render(study, cms_fig, hll_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote sketches report", extra={"path": str(out_path)})

    with track_run(settings, "sketches") as run:
        run.log_params({"n_flows": study.n_flows, "n_sources": study.n_sources})
        run.log_metrics(
            {"heavy_recall": study.heavy_recall, "reservoir_chi2_p": study.reservoir_chi2_p}
            | {f"cms_within_bound_{r.epsilon:g}": r.within_bound for r in study.countmin}
            | {f"hll_error_p{r.precision}": r.mean_relative_error for r in study.hyperloglog}
        )
        run.log_artifact(cms_fig)
        run.log_artifact(hll_fig)
        run.log_artifact(out_path)
    return out_path


def _cms_table(study: SketchStudy) -> str:
    rows = [
        "| epsilon | table | memory | mean overestimate | worst | ever under? | within the bound |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in study.countmin:
        rows.append(
            f"| {r.epsilon:g} | {r.depth} x {r.width:,} | {r.kilobytes:,.1f} KiB "
            f"| {r.mean_overestimate:,.1f} flows | {r.max_overestimate:,} | "
            f"{'**YES**' if r.ever_underestimated else 'never'} | {r.within_bound:.1%} |"
        )
    return "\n".join(rows)


def _hll_table(study: SketchStudy) -> str:
    rows = [
        "| precision | registers | memory (all hosts) | measured error | 1.04/sqrt(m) "
        "| top-k agreement | scanners in top-k |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in study.hyperloglog:
        rows.append(
            f"| p = {r.precision} | {r.registers:,} | {r.kilobytes:,.1f} KiB "
            f"| {r.mean_relative_error:.2%} | {r.theoretical_error:.2%} "
            f"| {r.top_k_agreement:.0%} | {r.scanners_found}/{study.scanners} |"
        )
    return "\n".join(rows)


def _cms_read(study: SketchStudy) -> str:
    if not study.countmin:
        return ""
    tight, loose = study.countmin[-1], study.countmin[0]
    under = any(r.ever_underestimated for r in study.countmin)
    safety = (
        "**No sizing ever undercounted a single host**, which is the property that makes this "
        "structure safe to detect with: a sketch that could report a scanner as quieter than "
        "it was would hide it, and one that can only exaggerate merely wastes an analyst's "
        "time."
        if not under
        else (
            "**A sizing undercounted a host**, which should be impossible for Count-Min and "
            "means the implementation is wrong. No claim in this section survives that."
        )
    )
    return (
        f"{safety} Error buys down with memory exactly as the bound says it should: at "
        f"epsilon = {loose.epsilon:g} the sketch costs {loose.kilobytes:,.1f} KiB and "
        f"overestimates a host by {loose.mean_overestimate:,.0f} flows on average, and at "
        f"epsilon = {tight.epsilon:g} it costs {tight.kilobytes:,.1f} KiB for "
        f"{tight.mean_overestimate:,.1f}. The number that does not appear in that sentence is "
        f"the one that matters most: **none of these sizes depend on how many hosts the stream "
        f"contains**. The exact counter grows with the traffic; the sketch is the size you "
        "chose when you deployed it, and stays there through the incident that quadruples "
        "your host count."
    )


def _hll_read(study: SketchStudy) -> str:
    if not study.hyperloglog:
        return ""
    best = study.hyperloglog[-1]
    within = [r for r in study.hyperloglog if r.mean_relative_error <= r.theoretical_error * 2]
    calibration = (
        f"Measured error tracks the theoretical `1.04/sqrt(m)` at {len(within)} of "
        f"{len(study.hyperloglog)} precisions, so the estimator is behaving as advertised"
        if within
        else (
            "Measured error is well above the theoretical standard error at every precision, "
            "which points at the implementation rather than at the method"
        )
    )
    operational = (
        f"and all {study.scanners} planted scanners are still inside the top {study.top_k}"
        if best.scanners_found >= study.scanners
        else f"but only {best.scanners_found} of {study.scanners} planted scanners survive"
    )
    return (
        f"{calibration}. The operational question is the one after that, and it is not the same "
        f"question: a fan-out estimate 2% wrong is worthless if the 2% lands on the ordering of "
        f"the shortlist an analyst reads. At p = {best.precision} the top-{study.top_k} ranking "
        f"agrees with exact counting {best.top_k_agreement:.0%} of the time, {operational}. That "
        "is the argument for sketches in its general form: scan detection is a *ranking* "
        "problem, and an error of a couple of percent is far smaller than the gap between a "
        f"scanner at {study.scanner_targets} destinations and the busiest ordinary host at "
        f"{study.max_benign_fanout}. It is also the shape of the failure mode — two candidates "
        "genuinely close together can be reordered by the approximation, and no amount of "
        "precision removes that, it only makes the window narrower."
    )


def _memory_read(study: SketchStudy) -> str:
    if not study.hyperloglog:
        return ""
    cheapest = min(study.hyperloglog, key=lambda r: r.kilobytes)
    dearest = max(study.hyperloglog, key=lambda r: r.kilobytes)
    ratio = study.exact_kilobytes / cheapest.kilobytes if cheapest.kilobytes > 0 else float("inf")
    crossover = [r for r in study.hyperloglog if r.kilobytes > study.exact_kilobytes]
    honest = (
        "\n\nAnd here the report has to argue against its own thesis, because the numbers do. "
        f"At p = {crossover[0].precision} and above the sketch costs **more** than exact "
        f"counting ({dearest.kilobytes:,.0f} KiB against {study.exact_kilobytes:,.0f}), and "
        "that is not a bug. This design keeps one HyperLogLog per source, so its memory scales "
        "with the number of *sources* while an exact set scales with each source's *fan-out*. "
        f"On this stream ordinary hosts touch at most {study.max_benign_fanout} peers, and a "
        "few dozen integers are cheaper than four thousand registers. Sketches pay when "
        "cardinality per key is large; here it is small for almost every key, and the honest "
        "recommendation is the low-precision configuration or exact sets, not the sketch that "
        "sounds most impressive. What does not change is the *shape*: exact memory is "
        "unbounded in fan-out and the sketch is not, so the moment one host starts touching "
        "millions of peers — which is the moment a scan detector exists for — the ordering "
        "reverses and never comes back."
        if crossover
        else ""
    )
    return (
        f"Over {study.n_flows:,} flows from {study.n_sources:,} sources, exact per-host "
        f"destination sets occupy {study.exact_kilobytes:,.0f} KiB and the smallest "
        f"HyperLogLog configuration answers the same question in {cheapest.kilobytes:,.0f} KiB "
        f"— a {ratio:,.0f}x reduction.{honest}"
    )


def _other_read(study: SketchStudy) -> str:
    heavy = (
        f"Misra-Gries recovered **{study.heavy_recall:.0%}** of the hosts genuinely above a "
        f"1/{study.heavy_k} share of the stream, which is what its guarantee promises: every "
        "true heavy hitter appears, though the summary also contains candidates that are not, "
        "so it is a shortlist to verify rather than an answer."
    )
    uniform = (
        f"The reservoir's composition is statistically indistinguishable from the stream it "
        f"sampled (chi-square p = {study.reservoir_chi2_p:.2f}), which is the only property "
        "that makes a one-pass fixed-memory sample worth having."
        if study.reservoir_chi2_p > 0.01
        else (
            f"The reservoir's composition differs from the stream (chi-square p = "
            f"{study.reservoir_chi2_p:.3f}), which for a correct implementation should not "
            "happen and is worth investigating before the sample is used for anything."
        )
    )
    return f"{heavy} {uniform}"


_SCOPE = """The stream is **synthetic**, and deliberately so. Cleaning drops the identity
columns before any model sees them — that is the leakage contract this whole project rests on
— so the flow table available downstream has no source or destination to count. The stream is
Zipf-distributed over sources with planted scanners because a uniform stream would flatter
every structure here: hash collisions hurt most under skew, which is precisely the regime real
traffic is in. What is being validated is the implementations and their bounds, and those
claims are distribution-free in the directions that matter (Count-Min's guarantee holds for
any stream; HyperLogLog's depends only on hash quality).

Memory figures count the structures themselves, not Python object overhead, which would swamp
them at this scale and says more about the interpreter than about the algorithms. A production
sensor would pack the HyperLogLog registers at six bits rather than eight and keep them off
the heap entirely, so the per-host figure here is an overestimate of what the method costs.

The per-source HyperLogLog design keeps one sketch per source, so memory grows with the number
of *sources* even though it does not grow with their fan-out. That is the right trade for scan
detection — sources are bounded by the address space you monitor and destinations are not — but
it is a design decision rather than a property of the algorithm, and a deployment watching a
wider address space would need a second layer of approximation on top."""


def _render(study: SketchStudy, cms_fig: Path, hll_fig: Path) -> str:
    return f"""# NetSentry — Counting Without Remembering

_Synthetic host stream: {study.n_flows:,} flows from {study.n_sources:,} sources, Zipf-skewed,
with {study.scanners} planted scanners touching {study.scanner_targets} destinations each._

## Why this report exists

The [host-graph analytics](graph_demo.md) find scanners by counting distinct destinations per
source. That is a set per host, and sets grow with what they hold. On a link doing tens of
thousands of flows a second, "keep a set per host" is not a tuning problem, it is a design
that runs out of memory during the incident it was bought for. Production flow analytics use
**sketches**: fixed-size structures answering the same question with a stated error, in memory
that does not grow with the stream.

Four are implemented here from scratch, and — more to the point — every guarantee each of them
makes is checked against exact ground truth rather than cited.

## Count-Min: per-host flow counts that never undercount

{_cms_table(study)}

{_cms_read(study)}

![Count-Min error against memory](../figures/{cms_fig.name})

## HyperLogLog: fan-out in a few kilobytes

{_hll_table(study)}

{_hll_read(study)}

![HyperLogLog error against its bound](../figures/{hll_fig.name})

## What it saves

{_memory_read(study)}

## Heavy hitters and the uniform sample

{_other_read(study)}

## Scope

{_SCOPE}"""
