"""Multivariate drift: the change every per-feature monitor is blind to, by construction.

The deployed drift monitor is a **marginal** one. PSI bins each feature on its own; the KS
suite tests each feature on its own and controls the false-discovery rate across them. Both
answer the question "did any single feature's distribution move?", and that question has a
well-defined blind spot: a change that leaves every marginal *exactly* intact while destroying
the joint structure. The sensor-failure study already walked into it — a collector that
mis-assembles records shuffles fields between flows, and PSI moved by nothing, because
permuting rows leaves every column's multiset identical. That was reported as a limitation.
This module is the instrument that removes it.

The **maximum mean discrepancy** (Gretton et al., JMLR 2012) embeds each sample as a mean in a
reproducing-kernel Hilbert space and measures the distance between the two embeddings. With a
characteristic kernel (the Gaussian RBF used here) that distance is zero *if and only if* the
two distributions are equal, so the test is consistent against **any** alternative, dependence
structure included. Nothing about it is specific to the moments a human thought to check.

Four things are measured, in the order a reviewer should want them:

1. **The null, first.** A test is worth nothing until its false-alarm rate is known, so the
   whole comparison is first run on two windows drawn from the *same* distribution. A monitor
   that fires on stationary traffic will be switched off within a week, whatever its power.
2. **Each fault, judged by every monitor on identical windows** — a marginal shift both
   families catch, a dependence-only fault the marginal family cannot catch (and the report
   shows *why* it cannot: the KS statistics come back bit-identical to the null run), and the
   real temporal shift the project already knows about.
3. **How much structure the joint test needs.** The stand-in's modelled features are very
   nearly independent, and under independence a row-permutation fault is a *no-op* rather than
   an invisible change — so a single measurement on them would report the generator rather than
   the monitor. The reach is measured instead on controlled windows whose dependence is a dial
   and whose marginals are identical at every setting.
4. **The cost.** The quadratic-time estimator is O(n^2) in time and memory; the linear-time
   estimator (Gretton 2012, section 6) trades power for a streaming budget. Both are measured
   here rather than asserted, because the choice between them is the operational decision.

The permutation null is computed as a single matrix product rather than one kernel rebuild per
permutation: with indicator vectors stacked into a matrix, every permuted statistic falls out
of one GEMM against the pooled kernel, which is what makes a 200-permutation test affordable
inside a monitoring loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.evaluation import plots
from netsentry.features.feature_sets import numeric_features
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.monitoring.detectors import ks_feature_tests
from netsentry.monitoring.drift import population_stability_index
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import MMDConfig

logger = get_logger(__name__)

REPORT_NAME = "mmd.md"
POWER_FIGURE_NAME = "mmd_power.png"
COST_FIGURE_NAME = "mmd_cost.png"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Kernel primitives. Pure functions on arrays, unit-tested directly.
# --------------------------------------------------------------------------------------


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at zero.

    The expansion ``|a|^2 + |b|^2 - 2 a.b`` is used rather than an explicit difference
    tensor: it is one GEMM instead of an ``(n, m, d)`` allocation, which is the difference
    between a monitor that fits in a request budget and one that does not. Floating-point
    cancellation can push near-zero entries slightly negative, so the result is clipped.
    """
    a2: np.ndarray = np.einsum("ij,ij->i", a, a)
    b2: np.ndarray = np.einsum("ij,ij->i", b, b)
    d2: np.ndarray = a2[:, None] + b2[None, :] - 2.0 * (a @ b.T)
    clipped: np.ndarray = np.maximum(d2, 0.0)
    return clipped


def median_bandwidth(
    x: np.ndarray, y: np.ndarray, *, rng: np.random.Generator, max_points: int = 500
) -> float:
    """Gaussian bandwidth by the median heuristic, returned as ``gamma = 1 / (2 sigma^2)``.

    The heuristic (median pairwise distance of the pooled sample) is the standard default and
    matters more than it looks: too small a bandwidth makes every point its own island and the
    statistic saturates, too large and the kernel cannot resolve anything. It is computed on a
    subsample because the median is stable long before the full pairwise matrix is affordable.
    """
    pooled: np.ndarray = np.vstack([x, y])
    if len(pooled) > max_points:
        pooled = pooled[rng.choice(len(pooled), max_points, replace=False)]
    d2: np.ndarray = squared_distances(pooled, pooled)
    iu = np.triu_indices(len(pooled), k=1)
    med = float(np.median(d2[iu])) if len(iu[0]) else 0.0
    if med <= _EPS:  # degenerate (identical points): fall back to a unit bandwidth
        return 0.5
    return 1.0 / med


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix ``exp(-gamma |a - b|^2)`` -- characteristic, hence consistent."""
    kernel: np.ndarray = np.exp(-gamma * squared_distances(a, b))
    return kernel


def mmd2_from_kernel(kernel: np.ndarray, indicators: np.ndarray) -> np.ndarray:
    """Unbiased MMD^2 for one or many splits of a pooled kernel matrix.

    ``indicators`` is ``(n_pooled,)`` or ``(n_pooled, n_splits)`` with 1 marking the first
    sample. Every permutation of a permutation test is one column here, so the whole null
    distribution costs a single ``kernel @ indicators`` product instead of one kernel rebuild
    per permutation. The diagonal is removed explicitly (``k(x, x) = 1`` for the RBF), which is
    what makes the estimator unbiased rather than the biased V-statistic.
    """
    a: np.ndarray = np.asarray(indicators, dtype=float)
    single = a.ndim == 1
    if single:
        a = a[:, None]
    b: np.ndarray = 1.0 - a
    ka: np.ndarray = kernel @ a
    kb: np.ndarray = kernel @ b
    m: np.ndarray = a.sum(axis=0)
    n: np.ndarray = b.sum(axis=0)
    s_xx: np.ndarray = np.einsum("ij,ij->j", a, ka)
    s_yy: np.ndarray = np.einsum("ij,ij->j", b, kb)
    s_xy: np.ndarray = np.einsum("ij,ij->j", a, kb)
    diag: np.ndarray = np.diag(kernel)
    trace_x: np.ndarray = diag @ a
    trace_y: np.ndarray = diag @ b
    out: np.ndarray = (
        (s_xx - trace_x) / np.maximum(m * (m - 1.0), _EPS)
        + (s_yy - trace_y) / np.maximum(n * (n - 1.0), _EPS)
        - 2.0 * s_xy / np.maximum(m * n, _EPS)
    )
    return out[0] if single else out


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, *, gamma: float) -> float:
    """Unbiased MMD^2 between two samples (zero in expectation under the null)."""
    pooled: np.ndarray = np.vstack([x, y])
    kernel = rbf_kernel(pooled, pooled, gamma)
    mask = np.concatenate([np.ones(len(x)), np.zeros(len(y))])
    return float(mmd2_from_kernel(kernel, mask))


@dataclass
class MMDTest:
    """One permutation test: the statistic, its exact-null p-value, and what produced it."""

    statistic: float
    p_value: float
    gamma: float
    n_reference: int
    n_current: int
    permutations: int
    seconds: float

    def rejects(self, alpha: float) -> bool:
        """Whether the test fires at level ``alpha``."""
        return self.p_value <= alpha


def mmd_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    gamma: float | None = None,
    permutations: int = 200,
    bandwidth_points: int = 500,
) -> MMDTest:
    """Two-sample MMD test with a permutation null.

    MMD^2's asymptotic null is a weighted sum of chi-squares with data-dependent weights, so
    there is no closed form worth using; the permutation null is exact for any sample size,
    which is what a monitor needs on a window of a few hundred flows. The p-value uses the
    ``(1 + #{null >= observed}) / (1 + B)`` correction, so it can never be zero -- reporting
    ``p = 0`` from a finite permutation set is a real and common error.
    """
    start = time.perf_counter()
    if gamma is None:
        gamma = median_bandwidth(x, y, rng=rng, max_points=bandwidth_points)
    pooled: np.ndarray = np.vstack([x, y])
    kernel = rbf_kernel(pooled, pooled, gamma)
    total = len(pooled)
    mask = np.concatenate([np.ones(len(x)), np.zeros(len(y))])
    observed = float(mmd2_from_kernel(kernel, mask))

    draws = np.empty((total, permutations), dtype=float)
    for b in range(permutations):
        draws[:, b] = rng.permutation(mask)
    null = np.asarray(mmd2_from_kernel(kernel, draws), dtype=float)
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + permutations))
    return MMDTest(
        statistic=observed,
        p_value=p_value,
        gamma=float(gamma),
        n_reference=len(x),
        n_current=len(y),
        permutations=permutations,
        seconds=time.perf_counter() - start,
    )


@dataclass
class LinearMMDTest:
    """The streaming estimator: linear time and memory, normal null, less power."""

    statistic: float
    z_score: float
    p_value: float
    n_pairs: int
    seconds: float

    def rejects(self, alpha: float) -> bool:
        """Whether the test fires at level ``alpha``."""
        return self.p_value <= alpha


def linear_time_mmd(
    x: np.ndarray, y: np.ndarray, *, gamma: float, rng: np.random.Generator
) -> LinearMMDTest:
    """MMD_l: the linear-time estimator with an asymptotically normal null (Gretton 2012).

    Points are paired off and each pair contributes one independent term, so the estimator is
    an average of i.i.d. variables and its null is Gaussian -- no permutations, no kernel
    matrix, O(n) memory. The price is variance: it discards most of the pairwise information
    the quadratic estimator uses, and the power sweep in this report measures exactly how much
    that costs. Both samples are shuffled first, because pairing adjacent rows of a
    time-ordered window would couple the estimator to the arrival order.
    """
    start = time.perf_counter()
    n = min(len(x), len(y)) // 2 * 2
    if n < 4:
        return LinearMMDTest(0.0, 0.0, 1.0, 0, 0.0)
    xs: np.ndarray = x[rng.permutation(len(x))[:n]]
    ys: np.ndarray = y[rng.permutation(len(y))[:n]]
    x1, x2 = xs[0::2], xs[1::2]
    y1, y2 = ys[0::2], ys[1::2]

    def _k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2: np.ndarray = np.einsum("ij,ij->i", a - b, a - b)
        return np.exp(-gamma * d2)

    h: np.ndarray = _k(x1, x2) + _k(y1, y2) - _k(x1, y2) - _k(x2, y1)
    mean = float(np.mean(h))
    std = float(np.std(h, ddof=1))
    n_pairs = len(h)
    z = mean / max(std / np.sqrt(n_pairs), _EPS)
    # One-sided normal tail; MMD is non-negative in expectation so only large values matter.
    p_value = float(0.5 * math_erfc(z / np.sqrt(2.0)))
    return LinearMMDTest(
        statistic=mean,
        z_score=z,
        p_value=min(1.0, max(p_value, 0.0)),
        n_pairs=n_pairs,
        seconds=time.perf_counter() - start,
    )


def math_erfc(value: float) -> float:
    """``erfc`` without importing scipy -- the normal tail is the only special function needed."""
    from math import erfc

    return float(erfc(value))


def per_feature_mmd(
    x: np.ndarray, y: np.ndarray, *, rng: np.random.Generator, max_points: int = 400
) -> np.ndarray:
    """Univariate MMD^2 per feature -- the marginal view, for attribution against the joint one.

    This is the honest comparator for the joint statistic: if the joint test fires and every
    univariate one is at its null value, the change lives in the dependence structure and no
    per-feature monitor can reach it.
    """
    xs = x[rng.choice(len(x), min(len(x), max_points), replace=False)]
    ys = y[rng.choice(len(y), min(len(y), max_points), replace=False)]
    out = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        col_x = xs[:, j : j + 1]
        col_y = ys[:, j : j + 1]
        gamma = median_bandwidth(col_x, col_y, rng=rng, max_points=max_points)
        out[j] = mmd2_unbiased(col_x, col_y, gamma=gamma)
    return out


# --------------------------------------------------------------------------------------
# The faults. Each is a controlled corruption of a window, with a documented invariant.
# --------------------------------------------------------------------------------------


def marginal_shift(
    window: np.ndarray, columns: np.ndarray, *, sigmas: float, rng: np.random.Generator
) -> np.ndarray:
    """Shift a few features' means -- the change a marginal monitor is designed to catch."""
    out = window.copy()
    scale = np.std(out[:, columns], axis=0)
    scale = np.where(scale > _EPS, scale, 1.0)
    out[:, columns] = out[:, columns] + sigmas * scale
    return out


def dependence_fault(
    window: np.ndarray, columns: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    """Permute a block of columns across rows: every marginal is preserved *exactly*.

    This is the sensor-failure study's mis-assembly fault at the feature level -- a collector
    that pairs one flow's forward counters with another flow's backward counters. The multiset
    of values in each column is unchanged, so every per-feature statistic (KS, PSI, mean,
    variance, any quantile) takes precisely the value it would have taken without the fault.
    The joint distribution, which is what the model actually consumes, is destroyed.
    """
    out = window.copy()
    order = rng.permutation(len(out))
    out[:, columns] = out[order][:, columns]
    return out


def correlated_window(n: int, d: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """An equicorrelated Gaussian window: identical marginals at every ``rho``.

    ``X = sqrt(rho) Z + sqrt(1 - rho) E`` with one shared factor ``Z`` gives every pair
    correlation ``rho`` while every marginal stays standard normal. That invariance is the
    point: sweeping ``rho`` changes *only* the dependence structure, so the resulting power
    curve measures exactly one thing -- how much joint structure there was to destroy.
    """
    shared: np.ndarray = rng.standard_normal((n, 1))
    idiosyncratic: np.ndarray = rng.standard_normal((n, d))
    window: np.ndarray = np.sqrt(rho) * shared + np.sqrt(max(1.0 - rho, 0.0)) * idiosyncratic
    return window


def dependence_strength(matrix: np.ndarray) -> tuple[float, float]:
    """Mean and max absolute off-diagonal correlation -- how much dependence a window carries."""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr: np.ndarray = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    off = np.abs(corr)
    n = corr.shape[0]
    if n < 2:
        return 0.0, 0.0
    return float(off.sum() / (n * (n - 1))), float(off.max())


# --------------------------------------------------------------------------------------
# The study: the null first, then power against each fault, then cost.
# --------------------------------------------------------------------------------------


@dataclass
class MarginalVerdict:
    """What the deployed per-feature monitors say about one window pair."""

    ks_flagged: int
    ks_max_statistic: float
    ks_min_p: float
    psi_max: float


@dataclass
class FaultOutcome:
    """One controlled corruption, judged by every monitor on the same window pair."""

    fault: str
    description: str
    mmd: MMDTest
    linear: LinearMMDTest
    marginal: MarginalVerdict
    ks_identical_to_null: bool


@dataclass
class PowerRow:
    """Detection rate of each monitor for one fault at one window size."""

    fault: str
    window: int
    rates: dict[str, float]


@dataclass
class DependenceRow:
    """Detection rate against a dependence-only fault, by how much dependence exists."""

    rho: float
    window: int
    rates: dict[str, float]


@dataclass
class CostRow:
    """What each estimator costs at one window size."""

    n: int
    quadratic_seconds: float
    linear_seconds: float
    kernel_mb: float


@dataclass
class MMDStudy:
    """Everything the report renders."""

    n_features: int
    window_rows: int
    permutations: int
    alpha: float
    repeats: int
    psi_threshold: float
    calibration: dict[str, float]
    faults: list[FaultOutcome]
    real: FaultOutcome
    marginal_power: list[PowerRow]
    dependence_power: list[DependenceRow]
    cost: list[CostRow]
    attribution: dict[str, list[tuple[str, float]]]
    faulted_features: list[str]
    real_mean_corr: float
    real_max_corr: float
    stream_features: int


DETECTORS = ("MMD (permutation)", "MMD (linear)", "KS + BH", "PSI")


def _marginal_verdict(
    reference: np.ndarray,
    current: np.ndarray,
    names: list[str],
    *,
    alpha: float,
    psi_bins: int,
) -> tuple[MarginalVerdict, np.ndarray]:
    """Run the deployed monitors, returning their verdict and the raw KS statistics.

    The statistics come back so the report can *prove* the blindness rather than assert it:
    under a dependence-only fault they are bit-identical to the unfaulted run.
    """
    ref_frame = pd.DataFrame(reference, columns=names)
    cur_frame = pd.DataFrame(current, columns=names)
    ks = ks_feature_tests(ref_frame, cur_frame, names, alpha=alpha)
    stats = np.array([k.statistic for k in ks], dtype=float)
    psi = [
        population_stability_index(reference[:, j], current[:, j], bins=psi_bins)
        for j in range(reference.shape[1])
    ]
    verdict = MarginalVerdict(
        ks_flagged=int(sum(1 for k in ks if k.significant)),
        ks_max_statistic=float(stats.max()) if len(stats) else 0.0,
        ks_min_p=float(min((k.p_value for k in ks), default=1.0)),
        psi_max=float(max(psi, default=0.0)),
    )
    return verdict, stats


def _judge(
    reference: np.ndarray,
    current: np.ndarray,
    names: list[str],
    cfg: MMDConfig,
    rng: np.random.Generator,
    *,
    permutations: int,
) -> tuple[MMDTest, LinearMMDTest, MarginalVerdict, np.ndarray]:
    """Every monitor's opinion of one window pair, on identical data."""
    test = mmd_permutation_test(
        reference,
        current,
        rng=rng,
        permutations=permutations,
        bandwidth_points=cfg.bandwidth_points,
    )
    linear = linear_time_mmd(reference, current, gamma=test.gamma, rng=rng)
    verdict, stats = _marginal_verdict(
        reference, current, names, alpha=cfg.alpha, psi_bins=cfg.psi_bins
    )
    return test, linear, verdict, stats


def _fired(
    test: MMDTest, linear: LinearMMDTest, verdict: MarginalVerdict, cfg: MMDConfig
) -> dict[str, bool]:
    """The four monitors' fire/no-fire calls at the operator's configured levels."""
    return {
        DETECTORS[0]: test.rejects(cfg.alpha),
        DETECTORS[1]: linear.rejects(cfg.alpha),
        DETECTORS[2]: verdict.ks_flagged > 0,
        DETECTORS[3]: verdict.psi_max >= cfg.psi_threshold,
    }


def _draw(pool: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint windows from one pool -- the only honest way to build a null."""
    idx = rng.choice(len(pool), size=min(2 * n, len(pool)), replace=False)
    half = len(idx) // 2
    return pool[idx[:half]], pool[idx[half : 2 * half]]


def _rate(counts: dict[str, int], repeats: int) -> dict[str, float]:
    return {k: v / repeats for k, v in counts.items()}


def run_mmd_study(settings: Settings) -> MMDStudy:
    """Calibrate the test on stationary traffic, then measure its power and its cost."""
    cfg: MMDConfig = settings.mmd
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test_frame = load_split(variant, "temporal", "test")
    pipeline = build_pipeline(variant)
    train_matrix: np.ndarray = np.asarray(pipeline.fit_transform(train), dtype=float)
    test_matrix: np.ndarray = np.asarray(pipeline.transform(test_frame), dtype=float)
    names = list(numeric_features())
    if len(names) != train_matrix.shape[1]:  # a categorical branch is enabled; name generically
        names = [f"feature_{i}" for i in range(train_matrix.shape[1])]
    n_features = train_matrix.shape[1]
    real_mean_corr, real_max_corr = dependence_strength(train_matrix)

    # The faulted block is the most strongly coupled one available: a mis-assembling collector
    # scrambles *related* fields together, and a block with no coupling to the rest is a fault
    # with nothing to expose.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(train_matrix, rowvar=False))
    np.fill_diagonal(corr, 0.0)
    coupling = np.abs(corr).mean(axis=1)
    faulted = np.argsort(coupling)[::-1][: min(cfg.n_faulted_features, n_features)]
    faulted_names = [names[j] for j in sorted(faulted.tolist())]

    # 1. The null. Two windows from the same distribution, repeated: this is the false-alarm
    #    rate an operator would live with, and it is measured before any power claim.
    fired_counts = dict.fromkeys(DETECTORS, 0)
    for _ in range(cfg.repeats):
        ref_draw, cur_draw = _draw(train_matrix, cfg.window_rows, rng)
        test, linear, verdict, _ = _judge(
            ref_draw, cur_draw, names, cfg, rng, permutations=cfg.power_permutations
        )
        for name, fired in _fired(test, linear, verdict, cfg).items():
            fired_counts[name] += int(fired)
    calibration = _rate(fired_counts, cfg.repeats)

    # 2. The faults, at the operator's window size, with the full permutation budget.
    ref, cur = _draw(train_matrix, cfg.window_rows, rng)
    null_test, null_linear, null_verdict, null_ks = _judge(
        ref, cur, names, cfg, rng, permutations=cfg.permutations
    )
    outcomes = [
        FaultOutcome(
            "none (stationary)",
            "two windows from the same distribution -- nothing should fire",
            null_test,
            null_linear,
            null_verdict,
            ks_identical_to_null=True,
        )
    ]
    shifted = marginal_shift(cur, faulted, sigmas=cfg.shift_sigmas, rng=rng)
    m_test, m_linear, m_verdict, m_ks = _judge(
        ref, shifted, names, cfg, rng, permutations=cfg.permutations
    )
    outcomes.append(
        FaultOutcome(
            "marginal shift",
            f"{len(faulted)} features shifted by {cfg.shift_sigmas:g} standard deviations",
            m_test,
            m_linear,
            m_verdict,
            ks_identical_to_null=bool(np.allclose(m_ks, null_ks)),
        )
    )
    scrambled = dependence_fault(cur, faulted, rng=rng)
    d_test, d_linear, d_verdict, d_ks = _judge(
        ref, scrambled, names, cfg, rng, permutations=cfg.permutations
    )
    outcomes.append(
        FaultOutcome(
            "dependence only",
            f"the same {len(faulted)} features permuted across rows -- every marginal preserved",
            d_test,
            d_linear,
            d_verdict,
            ks_identical_to_null=bool(np.allclose(d_ks, null_ks)),
        )
    )

    # 3. The real thing: the temporal shift this project already knows it has.
    real_ref = train_matrix[rng.choice(len(train_matrix), cfg.window_rows, replace=False)]
    real_cur = test_matrix[
        rng.choice(len(test_matrix), min(cfg.window_rows, len(test_matrix)), replace=False)
    ]
    r_test, r_linear, r_verdict, _ = _judge(
        real_ref, real_cur, names, cfg, rng, permutations=cfg.permutations
    )
    real = FaultOutcome(
        "real temporal shift",
        "training days vs the later capture days the model is actually deployed against",
        r_test,
        r_linear,
        r_verdict,
        ks_identical_to_null=False,
    )

    # 4. Power against the marginal shift, as a function of the window an operator has to buy.
    marginal_power: list[PowerRow] = []
    for window in cfg.window_sweep:
        counts = dict.fromkeys(DETECTORS, 0)
        for _ in range(cfg.power_repeats):
            a, b = _draw(train_matrix, window, rng)
            corrupted = marginal_shift(b, faulted, sigmas=cfg.shift_sigmas, rng=rng)
            t, lin, v, _ = _judge(
                a, corrupted, names, cfg, rng, permutations=cfg.power_permutations
            )
            for name, fired in _fired(t, lin, v, cfg).items():
                counts[name] += int(fired)
        marginal_power.append(PowerRow("marginal shift", window, _rate(counts, cfg.power_repeats)))

    # 5. The dependence-only fault, swept over how much dependence the data carries. The real
    #    features carry almost none (see the diagnostic above), so a single point on them would
    #    report the stand-in's independence rather than the monitor's reach.
    stream_names = [f"stream_{i}" for i in range(cfg.stream_features)]
    stream_columns = np.arange(cfg.stream_features // 2)
    dependence_power: list[DependenceRow] = []
    for rho in cfg.dependence_rhos:
        for window in cfg.window_sweep:
            counts = dict.fromkeys(DETECTORS, 0)
            for _ in range(cfg.power_repeats):
                a = correlated_window(window, cfg.stream_features, rho, rng)
                b = correlated_window(window, cfg.stream_features, rho, rng)
                corrupted = dependence_fault(b, stream_columns, rng=rng)
                t, lin, v, _ = _judge(
                    a, corrupted, stream_names, cfg, rng, permutations=cfg.power_permutations
                )
                for name, fired in _fired(t, lin, v, cfg).items():
                    counts[name] += int(fired)
            dependence_power.append(DependenceRow(rho, window, _rate(counts, cfg.power_repeats)))

    # 6. Cost. The quadratic estimator's memory is the operational constraint, not its time.
    cost: list[CostRow] = []
    for n in cfg.cost_sweep:
        a, b = _draw(train_matrix, n, rng)
        quad = mmd_permutation_test(
            a, b, rng=rng, permutations=cfg.permutations, bandwidth_points=cfg.bandwidth_points
        )
        lin = linear_time_mmd(a, b, gamma=quad.gamma, rng=rng)
        pooled = len(a) + len(b)
        cost.append(
            CostRow(
                n=len(a),
                quadratic_seconds=quad.seconds,
                linear_seconds=lin.seconds,
                kernel_mb=pooled * pooled * 8 / 1e6,
            )
        )

    # 7. Attribution: the marginal view of each fault, for the same windows judged above.
    attribution: dict[str, list[tuple[str, float]]] = {}
    for label, window_matrix in (("marginal shift", shifted), ("dependence only", scrambled)):
        per_feature = per_feature_mmd(
            ref, window_matrix, rng=rng, max_points=cfg.attribution_points
        )
        order = np.argsort(per_feature)[::-1][:5]
        attribution[label] = [(names[j], float(per_feature[j])) for j in order]

    return MMDStudy(
        n_features=n_features,
        window_rows=cfg.window_rows,
        permutations=cfg.permutations,
        alpha=cfg.alpha,
        repeats=cfg.repeats,
        psi_threshold=cfg.psi_threshold,
        calibration=calibration,
        faults=outcomes,
        real=real,
        marginal_power=marginal_power,
        dependence_power=dependence_power,
        cost=cost,
        attribution=attribution,
        faulted_features=faulted_names,
        real_mean_corr=real_mean_corr,
        real_max_corr=real_max_corr,
        stream_features=cfg.stream_features,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_mmd_report(settings: Settings) -> Path:
    """Run the multivariate-drift study and write the report + figures."""
    study = run_mmd_study(settings)
    windows = np.array(sorted({row.window for row in study.dependence_power}), dtype=float)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rho in sorted({row.rho for row in study.dependence_power}):
        rates = [
            next(
                r.rates[DETECTORS[0]]
                for r in study.dependence_power
                if r.window == w and r.rho == rho
            )
            for w in windows
        ]
        series[f"MMD, dependence {rho:g}"] = (windows, np.array(rates, dtype=float))
    top_rho = max(row.rho for row in study.dependence_power)
    series["KS + BH (any dependence)"] = (
        windows,
        np.array(
            [
                next(
                    r.rates[DETECTORS[2]]
                    for r in study.dependence_power
                    if r.window == w and r.rho == top_rho
                )
                for w in windows
            ],
            dtype=float,
        ),
    )
    power_fig = plots.plot_lines(
        series,
        xlabel="window size (flows per side)",
        ylabel=f"detection rate at alpha = {study.alpha:g}",
        title="Power against a dependence-only fault, by how much dependence exists",
        out_path=settings.paths.figures_dir / POWER_FIGURE_NAME,
    )
    sizes = np.array([row.n for row in study.cost], dtype=float)
    cost_fig = plots.plot_lines(
        {
            "quadratic (permutation)": (
                sizes,
                np.array([row.quadratic_seconds for row in study.cost]),
            ),
            "linear (streaming)": (sizes, np.array([row.linear_seconds for row in study.cost])),
        },
        xlabel="window size (flows per side)",
        ylabel="seconds per test",
        title="What each estimator costs",
        out_path=settings.paths.figures_dir / COST_FIGURE_NAME,
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, power_fig, cost_fig), encoding="utf-8")
    logger.info("Wrote MMD drift report", extra={"path": str(out_path)})

    with track_run(settings, "mmd") as run:
        run.log_params({"window_rows": study.window_rows, "permutations": study.permutations})
        run.log_metrics(
            {
                "null_fire_rate_mmd": study.calibration[DETECTORS[0]],
                "null_fire_rate_ks": study.calibration[DETECTORS[2]],
                "real_feature_mean_abs_corr": study.real_mean_corr,
                "dependence_mmd_p": study.faults[2].mmd.p_value,
                "real_mmd_statistic": study.real.mmd.statistic,
            }
        )
        run.log_artifact(power_fig)
        run.log_artifact(cost_fig)
        run.log_artifact(out_path)
    return out_path


def _calibration_table(study: MMDStudy) -> str:
    rows = ["| monitor | fires on stationary traffic | target |", "|---|---|---|"]
    for detector in DETECTORS:
        rate = study.calibration[detector]
        target = f"{study.alpha:.0%}" if detector.startswith("MMD") else "n/a (no level)"
        rows.append(f"| {detector} | {rate:.0%} | {target} |")
    return "\n".join(rows)


def _fault_table(study: MMDStudy) -> str:
    rows = [
        "| window pair | MMD^2 | MMD p | linear-MMD p | KS features flagged | max PSI | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for outcome in [*study.faults, study.real]:
        marginal_fires = (
            outcome.marginal.ks_flagged > 0 or outcome.marginal.psi_max >= study.psi_threshold
        )
        mmd_fires = outcome.mmd.p_value <= study.alpha
        if mmd_fires and marginal_fires:
            verdict = "both fire"
        elif mmd_fires:
            verdict = "**only MMD fires**"
        elif marginal_fires:
            verdict = "only the marginals fire"
        else:
            verdict = "neither fires"
        rows.append(
            f"| {outcome.fault} | {outcome.mmd.statistic:.4f} | {outcome.mmd.p_value:.3f} | "
            f"{outcome.linear.p_value:.3f} | {outcome.marginal.ks_flagged} / {study.n_features} | "
            f"{outcome.marginal.psi_max:.3f} | {verdict} |"
        )
    return "\n".join(rows)


def _dependence_table(study: MMDStudy) -> str:
    window = max(row.window for row in study.dependence_power)
    header = "| pairwise dependence | " + " | ".join(DETECTORS) + " |"
    rows = [header, "|" + "---|" * (1 + len(DETECTORS))]
    for row in study.dependence_power:
        if row.window != window:
            continue
        cells = " | ".join(f"{row.rates[d]:.0%}" for d in DETECTORS)
        rows.append(f"| {row.rho:g} | {cells} |")
    return "\n".join(rows)


def _power_table(study: MMDStudy) -> str:
    header = "| window | " + " | ".join(DETECTORS) + " |"
    rows = [header, "|" + "---|" * (1 + len(DETECTORS))]
    for row in study.marginal_power:
        cells = " | ".join(f"{row.rates[d]:.0%}" for d in DETECTORS)
        rows.append(f"| {row.window:,} | {cells} |")
    return "\n".join(rows)


def _cost_table(study: MMDStudy) -> str:
    rows = [
        "| window (per side) | kernel memory | quadratic test | linear test | ratio |",
        "|---|---|---|---|---|",
    ]
    for row in study.cost:
        ratio = row.quadratic_seconds / max(row.linear_seconds, 1e-6)
        rows.append(
            f"| {row.n:,} | {row.kernel_mb:.1f} MB | {row.quadratic_seconds * 1000:.0f} ms | "
            f"{row.linear_seconds * 1000:.1f} ms | {ratio:.0f}x |"
        )
    return "\n".join(rows)


def _attribution_table(study: MMDStudy) -> str:
    rows = ["| fault | features with the largest *marginal* MMD |", "|---|---|"]
    for fault, entries in study.attribution.items():
        cells = ", ".join(f"`{name}` ({value:.4f})" for name, value in entries)
        rows.append(f"| {fault} | {cells} |")
    return "\n".join(rows)


def _blindness_read(study: MMDStudy) -> str:
    dependence = study.faults[2]
    marginal = study.faults[1]
    proof = (
        "The KS statistics under this fault are **bit-identical** to the ones from the "
        "unfaulted window pair -- `np.allclose` over the full vector of per-feature statistics "
        "returns `True`."
        if dependence.ks_identical_to_null
        else "The KS statistics under this fault differ slightly from the unfaulted ones because "
        "the two windows are not the same rows; the invariant that matters -- each column's "
        "multiset -- is preserved exactly."
    )
    return (
        f"The marginal shift is the easy case and both families see it: KS flags "
        f"{marginal.marginal.ks_flagged} of {study.n_features} features, PSI reads "
        f"{marginal.marginal.psi_max:.2f}, and MMD returns p = {marginal.mmd.p_value:.3f}. The "
        f"dependence-only fault permutes the *same* {len(study.faulted_features)} features across "
        "rows, so each column's multiset of values is unchanged and no per-feature statistic can "
        f"move. {proof} PSI reads {dependence.marginal.psi_max:.3f} against a "
        f"{study.psi_threshold:g} threshold and KS flags {dependence.marginal.ks_flagged} "
        "features -- not because the fault is small, but because the statistic they compute is "
        "mathematically constant under it. No threshold change closes that gap."
    )


def _stand_in_read(study: MMDStudy) -> str:
    dependence = study.faults[2]
    return (
        f"**And the joint test does not fire either** (p = {dependence.mmd.p_value:.3f}), which "
        "is the right answer and worth understanding before reading anything else here. Across "
        f"the {study.n_features} modelled features of the synthetic stand-in, the mean absolute "
        f"pairwise correlation is **{study.real_mean_corr:.3f}** and the strongest single pair "
        f"reaches **{study.real_max_corr:.3f}**. The features are very nearly independent, and "
        "under independence, re-pairing one block of columns with different rows produces a "
        "sample from *the same joint distribution* -- the fault is not merely invisible, it is a "
        "no-op. A test that fired here would be reporting an error. This is a property of the "
        "generator, not of CIC-IDS2017, where `Flow Duration`, the IAT statistics and the packet "
        "counts are mechanically coupled -- a duration is a sum of inter-arrival times. The same "
        "absence turned up when the feature store looked for repeat hosts and found one address "
        "per flow: the stand-in reproduces the dataset's *marginals*, not its structure. So the "
        "reach of the joint test is measured below on controlled windows whose dependence is a "
        "dial, which also answers a better question than a single fault would: not *can* it see "
        "this fault, but *how much structure must exist* before it can."
    )


def _dependence_read(study: MMDStudy) -> str:
    window = max(row.window for row in study.dependence_power)
    rows = [r for r in study.dependence_power if r.window == window]
    zero = min(rows, key=lambda r: r.rho)
    first_strong = next(
        (r for r in sorted(rows, key=lambda r: r.rho) if r.rates[DETECTORS[0]] >= 0.9), rows[-1]
    )
    ks_max = max(r.rates[DETECTORS[2]] for r in rows)
    psi_max = max(r.rates[DETECTORS[3]] for r in rows)
    return (
        f"At zero dependence the joint test fires {zero.rates[DETECTORS[0]]:.0%} of the time -- "
        "its false-alarm rate, correctly, because there the fault changes nothing. By a pairwise "
        f"dependence of {first_strong.rho:g} it reaches "
        f"{first_strong.rates[DETECTORS[0]]:.0%} detection on windows of {window:,} flows. The "
        f"marginal monitors stay at their own false-alarm rate throughout — KS peaks at "
        f"{ks_max:.0%} and PSI at {psi_max:.0%}, with no trend in the dependence they are being "
        "shown, because the numbers they compute do not depend on it. Those cells are noise on "
        "a null, not detection, and no amount of extra dependence will turn them into detection: "
        "the invariance is algebraic. The gap between the two curves is the entire argument for "
        "carrying a joint test alongside the per-feature ones."
    )


def _real_read(study: MMDStudy) -> str:
    real = study.real
    return (
        f"On the real temporal shift every monitor agrees -- MMD^2 = {real.mmd.statistic:.4f} at "
        f"p = {real.mmd.p_value:.3f}, {real.marginal.ks_flagged} of {study.n_features} features "
        f"BH-significant, max PSI {real.marginal.psi_max:.3f}. That agreement is worth stating "
        "plainly: the joint test is not a replacement for the deployed monitor on the shift the "
        "project already knows about. It is insurance against the class of change the deployed "
        "monitor cannot represent. And it is worth remembering what *any* input-distribution "
        "test can and cannot say -- the covariate-shift study diagnosed this same temporal gap "
        "as **concept** shift, where the inputs move far less than the input-output relationship "
        "does. A test on inputs alone will fire on harmless seasonality and stay silent while "
        "the labels rot. Drift detection buys an alarm, not a diagnosis."
    )


def _cost_read(study: MMDStudy) -> str:
    biggest = study.cost[-1]
    smallest = study.cost[0]
    top_rho = max(row.rho for row in study.dependence_power)
    strong = [r for r in study.dependence_power if r.rho == top_rho]
    best_linear = max((r.rates[DETECTORS[1]] for r in strong), default=0.0)
    best_quad = max((r.rates[DETECTORS[0]] for r in strong), default=0.0)
    return (
        f"The quadratic estimator holds an entire pooled kernel in memory: "
        f"{biggest.kernel_mb:.0f} MB at {biggest.n:,} flows per side against "
        f"{smallest.kernel_mb:.1f} MB at {smallest.n:,} -- it is the memory, not the "
        f"{biggest.quadratic_seconds * 1000:.0f} ms, that decides how large a window a monitor "
        "can afford. Batching the permutations into one matrix product is what keeps the time "
        "affordable at all: the null distribution costs a single GEMM against the pooled kernel "
        "instead of one kernel rebuild per permutation. The linear-time estimator removes the "
        f"quadratic term entirely ({biggest.linear_seconds * 1000:.1f} ms, O(n) memory) and pays "
        f"in power -- at the strongest dependence level it reaches {best_linear:.0%} detection "
        f"against the permutation test's {best_quad:.0%}. The operational reading: run the linear "
        "estimator continuously as a cheap tripwire and spend the quadratic test on the windows "
        "it flags, or on a schedule."
    )


def _render(study: MMDStudy, power_fig: Path, cost_fig: Path) -> str:
    return f"""# NetSentry — Multivariate Drift: the Change the Marginals Cannot See

_Kernel two-sample testing (MMD, Gretton et al. 2012) against the deployed per-feature monitors.
Windows of {study.window_rows:,} flows over {study.n_features} features, {study.permutations}
permutations, level alpha = {study.alpha:g}._

## Why this report exists

The deployed drift monitor is a **marginal** one. PSI bins each feature on its own; the KS suite
tests each feature on its own and controls the false-discovery rate across the family. Both
answer "did any single feature's distribution move?" — and that question has a blind spot with a
proof attached to it rather than a probability. A change that re-pairs values *between* rows
leaves every column's multiset exactly as it was, so every per-feature statistic takes exactly
the value it would have taken with no fault at all. The sensor-failure study met this already: a
collector that mis-assembles records moved PSI by nothing. That was recorded as a limitation.
This is the instrument that removes it.

The **maximum mean discrepancy** embeds each sample as a mean in a reproducing-kernel Hilbert
space and measures the distance between the two embeddings. With a characteristic kernel — the
Gaussian RBF used here — that distance is zero **if and only if** the distributions are equal,
so the test is consistent against any alternative, dependence structure included. Nothing in it
is specific to the moments a human thought to check.

## The null, before anything else

A monitor's false-alarm rate is the first thing an operator needs and the last thing most drift
reports measure. Two windows are drawn from the *same* distribution, {study.repeats} times, and
every monitor is asked whether it fires.

{_calibration_table(study)}

The permutation test is exact by construction, so its rate should sit at the level; the linear
estimator leans on a normal approximation and is the one that could misbehave. PSI and the KS
suite have no calibrated level at all — PSI's 0.1/0.2 thresholds are convention, not a test —
which is precisely why their rates belong in the same table as the tests that do.

## What each monitor sees

{_fault_table(study)}

{_blindness_read(study)}

## The stand-in has no dependence to destroy

{_stand_in_read(study)}

## How much structure does the joint test need?

![Detection rate vs window size, by dependence](../figures/{power_fig.name})

Controlled windows of {study.stream_features} features whose pairwise dependence is a dial and
whose marginals are **identical at every setting** (one shared factor plus idiosyncratic noise),
so the sweep varies exactly one thing. Half the columns are then re-paired across rows — the
same fault as above.

{_dependence_table(study)}

{_dependence_read(study)}

## Power against a marginal shift, on the real features

{_power_table(study)}

Window size is the operational variable: a monitor on hourly windows sees a few hundred flows,
one on daily windows sees tens of thousands, and the price of the second is latency — a day of
serving a model the data has moved away from. On the fault the deployed monitors *can* see, they
are not merely adequate but better than the joint test at small windows, which is what a
seventy-six-dimensional kernel test costs you: it spends its samples on the whole joint law
rather than concentrating them on the six coordinates that moved.

## The marginal view of each fault

{_attribution_table(study)}

Per-feature MMD, computed on the same windows the joint test judged. Under the marginal shift
the faulted features rank at the top, which is what attribution is for. Under the
dependence-only fault the largest marginal discrepancy is indistinguishable from the null —
there is nothing for a per-feature dashboard to rank.

## The real shift

{_real_read(study)}

## Cost

{_cost_table(study)}

![Cost per test](../figures/{cost_fig.name})

{_cost_read(study)}

## Scope and honest limits

- **A characteristic kernel is consistent, not omniscient at finite n.** MMD detects any
  difference *given enough samples*; on a window of a few hundred flows it detects the ones that
  are large relative to the kernel bandwidth. The median heuristic is a default, not an optimum
  — a kernel trained to maximise test power (Sutherland et al. 2017) would do better, and would
  need its own held-out split to stay honest.
- **It fires on harmless change too.** Consistency cuts both ways: a benign traffic-mix shift is
  a distributional change and will be flagged. This is a tripwire feeding triage, not an
  automatic retraining trigger — the retrain-policy study already showed what happens when a
  drift signal is wired straight to an action.
- **It says nothing about labels.** Input-only tests cannot see concept shift, which is what the
  covariate-shift study found this dataset's temporal gap actually is.
- **The dependence sweep is synthetic, deliberately.** Its value is that the invariant is exact:
  marginals fixed, dependence dialled, so the curve measures the monitor rather than the data.
  The real-feature diagnostic above is what says which regime CIC-IDS2017 would sit in."""
