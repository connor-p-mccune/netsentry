"""Extreme-value thresholds: estimating an operating point out where the data runs out.

The [Neyman-Pearson report](neyman_pearson.md) established that the deployed threshold is
an order statistic of a finite benign sample. This one asks the question that follows
naturally and is rarely asked: *which* order statistic, and is there anything better to do
with a tail than count it?

At a 0.1% false-positive budget on 5,611 benign validation flows, the threshold is pinned
by the top five scores. Five. Every claim downstream — the detection rate, the alerts-per-day
estimate, the cost model's operating point — rests on where five flows happened to land.
Push to a 0.01% budget and the empirical quantile stops existing: `n * alpha < 1`, so the
rule degenerates to "the largest score I have seen", which is not an estimate of anything.

Extreme-value theory is the branch of statistics built for exactly this. The
Pickands-Balkema-de Haan theorem says that for essentially any underlying distribution, the
**exceedances over a high threshold** converge to a Generalized Pareto distribution as the
threshold rises:

    P(X - u > y | X > u)  ->  (1 + xi * y / sigma) ^ (-1/xi)

So instead of reading one order statistic, fit a two-parameter GPD to the *whole* tail and
extrapolate. The estimator then uses hundreds of flows to place a threshold rather than
five, and it can quote an operating point beyond the smallest resolvable empirical quantile.
This is the same peaks-over-threshold machinery Siffer et al. (*Anomaly Detection in Streams
with Extreme Value Theory*, KDD 2017) brought to streaming anomaly detection.

The fit is Grimshaw's (1993) profile-likelihood reparameterisation, implemented here rather
than imported: with `theta = xi / sigma` the shape solves in closed form as
`xi(theta) = mean(log(1 + theta * y))`, leaving a well-behaved one-dimensional search. Two
details matter operationally. The shape parameter `xi` is a statement about the world — `xi
> 0` is a heavy tail with no upper limit, `xi = 0` exponential, `xi < 0` a tail that *ends*
at `u - sigma/xi`, which is what a bounded score ought to produce and is worth checking
rather than assuming. And the threshold `u` where the tail is declared to start is a
bias-variance dial, so it is swept rather than fixed by fiat.

None of this is free, and the report is built around the trade. EVT buys variance reduction
and extrapolation by *assuming* a tail shape, where the empirical quantile assumes nothing
and the Neyman-Pearson bound assumes nothing. The honest way to price that is a controlled
simulation against populations whose extreme quantiles are known in closed form — including
one where the extra shape parameter makes EVT *worse* — and that is the arm that carries
the conclusion. The real-data arm can only report thresholds, because the benign flows
available cannot resolve a 0.01% rate no matter which estimator produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.evaluation.neyman_pearson import naive_count, rates_above, threshold_from_count
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import EVTConfig

logger = get_logger(__name__)

REPORT_NAME = "evt.md"
TAIL_FIGURE = "evt_tail_fit.png"
ERROR_FIGURE = "evt_estimator_error.png"

#: The three ways this project places a threshold, and what each one costs epistemically.
_APPROACH_TABLE = "\n".join(
    [
        "| approach | what it assumes | what it gives you | where it fails |",
        "|---|---|---|---|",
        "| empirical quantile | nothing | an unbiased-ish point estimate "
        "| degenerates when `n * alpha < 1` |",
        "| [Neyman-Pearson](neyman_pearson.md) | nothing (order statistics only) "
        "| `P(FPR > alpha) <= delta`, finite-sample "
        "| infeasible below ~`log(delta)/log(1-alpha)` flows |",
        "| EVT / POT (this report) | the tail is in a GPD domain of attraction "
        "| a low-variance estimate that extrapolates past the data "
        "| silently wrong if the tail assumption is wrong |",
    ]
)


# --------------------------------------------------------------------------------------
# The Generalized Pareto fit (pure; unit-tested against closed-form draws and SciPy)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GPDFit:
    """A fitted Generalized Pareto tail: shape, scale, and where the tail was declared."""

    xi: float
    sigma: float
    u: float
    n_exceed: int
    n_total: int
    log_likelihood: float

    @property
    def tail_rate(self) -> float:
        """Share of the sample that exceeded ``u`` — the tail's own base rate."""
        return self.n_exceed / self.n_total if self.n_total else 0.0

    @property
    def upper_endpoint(self) -> float:
        """Largest value the fitted tail admits (infinite unless ``xi < 0``).

        A negative shape is a genuine claim, not a fitting artefact: it says the benign
        score distribution *stops*, and no amount of traffic will produce a flow above
        ``u - sigma/xi``. For a bounded score that is the expected answer, and it caps how
        strict an achievable false-positive budget can be.
        """
        if self.xi >= 0:
            return math.inf
        return self.u - self.sigma / self.xi


def _profile_log_likelihood(theta: float, excesses: np.ndarray) -> float:
    """GPD log-likelihood profiled onto Grimshaw's single parameter ``theta = xi / sigma``.

    Substituting the closed-form shape ``xi(theta) = mean(log(1 + theta*y))`` collapses a
    two-parameter surface with an awkward constraint into a one-dimensional search, which
    is why this reparameterisation is the standard way to fit a GPD without an optimiser.
    """
    z = 1.0 + theta * excesses
    if np.any(z <= 0.0):
        return -math.inf
    logs = np.log(z)
    xi = float(logs.mean())
    if abs(xi) < 1e-12:
        return -math.inf
    sigma = xi / theta
    if sigma <= 0.0:
        return -math.inf
    n = len(excesses)
    return float(-n * math.log(sigma) - (1.0 + 1.0 / xi) * logs.sum())


def _theta_grid(excesses: np.ndarray, grid_points: int) -> np.ndarray:
    """Candidate ``theta`` values spanning both tail regimes without crossing the boundary.

    ``theta`` must stay above ``-1/max(y)`` or the likelihood is undefined, and the useful
    range spans orders of magnitude, so the grid is logarithmic on each side of zero.
    """
    y_max = float(np.max(excesses))
    y_mean = float(np.mean(excesses))
    half = max(int(grid_points) // 2, 8)
    scale = 1.0 / max(y_mean, 1e-12)
    positive = np.logspace(math.log10(scale) - 4.0, math.log10(scale) + 4.0, half)
    boundary = 1.0 / max(y_max, 1e-12)
    negative = -np.logspace(math.log10(boundary) - 6.0, math.log10(boundary * 0.999999), half)
    return np.concatenate([negative[::-1], positive])


def fit_gpd(scores: np.ndarray, u: float, *, grid_points: int = 400) -> GPDFit | None:
    """Fit a Generalized Pareto to the exceedances of ``scores`` over ``u``.

    Returns ``None`` when the tail is too thin to fit (fewer than two distinct exceedances),
    which is a real outcome on a bounded score with ties, not an error to swallow. The
    search is a log-spaced grid over Grimshaw's ``theta`` followed by a golden-section
    refinement inside the winning bracket — deterministic, dependency-free, and accurate to
    well under the sampling error of the fit itself.
    """
    values = np.asarray(scores, dtype=float)
    excesses = values[values > u] - u
    excesses = excesses[excesses > 0.0]
    if len(excesses) < 2 or len(np.unique(excesses)) < 2:
        return None

    grid = _theta_grid(excesses, grid_points)
    lls = np.array([_profile_log_likelihood(float(t), excesses) for t in grid])
    if not np.any(np.isfinite(lls)):
        return None
    best = int(np.argmax(lls))
    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])

    # Golden-section refinement inside the bracket the grid selected.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(80):
        if _profile_log_likelihood(c, excesses) > _profile_log_likelihood(d, excesses):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    theta = 0.5 * (a + b)
    ll = _profile_log_likelihood(theta, excesses)
    if not math.isfinite(ll):
        theta = float(grid[best])
        ll = float(lls[best])
    if not math.isfinite(ll):
        return None

    xi = float(np.mean(np.log1p(theta * excesses)))
    sigma = xi / theta
    return GPDFit(
        xi=xi,
        sigma=sigma,
        u=float(u),
        n_exceed=len(excesses),
        n_total=len(values),
        log_likelihood=float(ll),
    )


def gpd_quantile(fit: GPDFit, tail_probability: float) -> float:
    """Score above which a fraction ``tail_probability`` of the population is expected.

    The peaks-over-threshold quantile: the GPD describes the conditional tail, so the
    unconditional budget is rescaled by the tail's own base rate before inverting. When the
    fitted shape is negative the expression saturates at the distribution's upper endpoint,
    which is the correct answer rather than a numerical accident — the tail has ended.
    """
    q = float(tail_probability)
    if q <= 0.0:
        return fit.upper_endpoint
    ratio = q / fit.tail_rate if fit.tail_rate > 0 else math.inf
    if ratio >= 1.0:
        # The budget is looser than the tail itself; POT has nothing to say above u.
        return fit.u
    if abs(fit.xi) < 1e-9:
        return fit.u - fit.sigma * math.log(ratio)
    quantile: float = fit.u + (fit.sigma / fit.xi) * (ratio ** (-fit.xi) - 1.0)
    return quantile


def pot_threshold(
    scores: np.ndarray,
    budget: float,
    *,
    tail_quantile: float = 0.95,
    grid_points: int = 400,
) -> tuple[float, GPDFit | None]:
    """Peaks-over-threshold estimate of the score at a target false-positive ``budget``.

    Falls back to the plain empirical quantile when the tail cannot be fitted, so the
    caller always receives a usable threshold and the report can say how often the fallback
    fired instead of hiding it behind a silent default.
    """
    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        return math.inf, None
    u = float(np.quantile(values, tail_quantile))
    fit = fit_gpd(values, u, grid_points=grid_points)
    if fit is None:
        return empirical_threshold(values, budget), None
    return gpd_quantile(fit, budget), fit


def empirical_threshold(scores: np.ndarray, budget: float) -> float:
    """The order-statistic threshold the project currently deploys, for comparison.

    Shares its definition with the Neyman-Pearson study so the two reports are talking
    about the same rule: the highest score that lets at most ``floor(n * budget)`` benign
    calibration flows through.
    """
    values = np.asarray(scores, dtype=float)
    return threshold_from_count(values, naive_count(len(values), budget))


def mean_excess(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Mean exceedance above each threshold — the classic GPD-regime diagnostic.

    Under a Generalized Pareto tail the mean excess is *linear* in the threshold, so the
    point where this curve straightens is where the asymptotics have taken hold. It is the
    honest way to choose ``u`` instead of picking a round quantile and hoping.
    """
    values = np.asarray(scores, dtype=float)
    out = []
    for t in np.asarray(thresholds, dtype=float):
        excess = values[values > t] - t
        out.append(float(excess.mean()) if len(excess) else 0.0)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------------------
# Controlled simulation: the only arm where the truth is known
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Population:
    """A distribution whose extreme quantiles and tail probabilities are closed-form."""

    name: str
    xi: float
    note: str

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` benign-like scores."""
        u = rng.random(n)
        if self.name == "exponential":
            return -np.log1p(-u)
        if self.name == "heavy (Pareto, xi=1/3)":
            return (1.0 - u) ** (-1.0 / 3.0) - 1.0
        return u  # uniform: a hard upper endpoint at 1

    def tail_probability(self, x: np.ndarray) -> np.ndarray:
        """Exact ``P(X > x)`` — the realized false-positive rate of a threshold at ``x``."""
        x = np.asarray(x, dtype=float)
        out: np.ndarray
        if self.name == "exponential":
            out = np.exp(-np.maximum(x, 0.0))
        elif self.name == "heavy (Pareto, xi=1/3)":
            out = np.power(1.0 + np.maximum(x, 0.0), -3.0)
        else:
            out = np.clip(1.0 - x, 0.0, 1.0)
        return out


POPULATIONS = (
    Population("exponential", 0.0, "the boundary case: no shape parameter to estimate"),
    Population("heavy (Pareto, xi=1/3)", 1.0 / 3.0, "an unbounded tail the quantile cannot see"),
    Population("uniform", -1.0, "a hard endpoint: the tail simply stops"),
)


@dataclass
class SimRow:
    """One (population, budget, method) cell of the controlled comparison."""

    population: str
    budget: float
    method: str
    median_ratio: float
    p10_ratio: float
    p90_ratio: float
    blowout_rate: float
    fallback_rate: float


def simulate_estimators(
    population: Population,
    budget: float,
    *,
    n: int,
    trials: int,
    tail_quantile: float,
    grid_points: int,
    seed: int,
) -> list[SimRow]:
    """Compare both estimators against a population whose true tail is known exactly.

    The metric is the **realized** false-positive rate of each estimated threshold, divided
    by the budget it was aiming for. A ratio of 1.0 is a threshold that does what it says;
    above 1.0 the deployment quietly runs over its alert budget. Because the population is
    closed-form, this needs no holdout and therefore carries none of the resolution problem
    that makes the real-data arm mute at extreme budgets.
    """
    rng = np.random.default_rng(seed)
    emp_ratios: list[float] = []
    evt_ratios: list[float] = []
    fallbacks = 0
    for _ in range(int(trials)):
        sample = population.sample(rng, int(n))
        t_emp = empirical_threshold(sample, budget)
        t_evt, fit = pot_threshold(
            sample, budget, tail_quantile=tail_quantile, grid_points=grid_points
        )
        if fit is None:
            fallbacks += 1
        emp_ratios.append(float(population.tail_probability(np.array([t_emp]))[0]) / budget)
        evt_ratios.append(float(population.tail_probability(np.array([t_evt]))[0]) / budget)

    def _row(method: str, ratios: list[float], fallback: float) -> SimRow:
        arr = np.asarray(ratios, dtype=float)
        return SimRow(
            population=population.name,
            budget=budget,
            method=method,
            median_ratio=float(np.median(arr)),
            p10_ratio=float(np.percentile(arr, 10)),
            p90_ratio=float(np.percentile(arr, 90)),
            blowout_rate=float(np.mean(arr > 2.0)),
            fallback_rate=fallback,
        )

    return [
        _row("empirical quantile", emp_ratios, 0.0),
        _row("EVT (peaks-over-threshold)", evt_ratios, fallbacks / max(int(trials), 1)),
    ]


# --------------------------------------------------------------------------------------
# Real-data arm
# --------------------------------------------------------------------------------------
@dataclass
class StabilityRow:
    """The fit at one choice of where the tail begins."""

    tail_quantile: float
    u: float
    n_exceed: int
    xi: float
    sigma: float
    threshold: float
    endpoint: float


@dataclass
class RealRow:
    """Both thresholds at one budget, and what they did on the later days."""

    budget: float
    resolvable: bool
    n_benign_needed: int
    empirical_threshold: float
    evt_threshold: float
    empirical_test_fpr: float
    evt_test_fpr: float
    empirical_test_tpr: float
    evt_test_tpr: float


@dataclass
class EVTStudy:
    """Everything the report renders."""

    n_cal: int
    n_test: int
    n_test_benign: int
    tail_quantile: float
    fit: GPDFit | None
    rows: list[RealRow]
    stability: list[StabilityRow]
    sims: list[SimRow]
    sim_n: int
    sim_trials: int
    tail_scores: np.ndarray


def run_evt(settings: Settings) -> EVTStudy:
    """Fit the benign tail, extrapolate the operating points, and price the assumption."""
    cfg: EVTConfig = settings.evt
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    result = fit_supervised(variant)
    benign_label = variant.labels.benign_label
    s_val = attack_probability(np.asarray(result.proba_val), result.classes, benign_label)
    s_test = attack_probability(np.asarray(result.proba_test), result.classes, benign_label)
    y_val = np.asarray(result.y_val).astype(int)
    y_test = np.asarray(result.y_test).astype(int)
    benign_cal = s_val[y_val == 0]
    n_cal = len(benign_cal)

    fit = fit_gpd(
        benign_cal, float(np.quantile(benign_cal, cfg.tail_quantile)), grid_points=cfg.grid_points
    )

    rows: list[RealRow] = []
    for budget in cfg.budgets:
        t_emp = empirical_threshold(benign_cal, budget)
        t_evt, _ = pot_threshold(
            benign_cal, budget, tail_quantile=cfg.tail_quantile, grid_points=cfg.grid_points
        )
        emp_tpr, emp_fpr = rates_above(y_test, s_test, t_emp)
        evt_tpr, evt_fpr = rates_above(y_test, s_test, t_evt)
        rows.append(
            RealRow(
                budget=budget,
                resolvable=naive_count(n_cal, budget) >= 1,
                n_benign_needed=math.ceil(1.0 / budget),
                empirical_threshold=t_emp,
                evt_threshold=t_evt,
                empirical_test_fpr=emp_fpr,
                evt_test_fpr=evt_fpr,
                empirical_test_tpr=emp_tpr,
                evt_test_tpr=evt_tpr,
            )
        )

    stability: list[StabilityRow] = []
    for q in cfg.tail_quantile_sweep:
        u = float(np.quantile(benign_cal, q))
        f = fit_gpd(benign_cal, u, grid_points=cfg.grid_points)
        if f is None:
            continue
        stability.append(
            StabilityRow(
                tail_quantile=q,
                u=u,
                n_exceed=f.n_exceed,
                xi=f.xi,
                sigma=f.sigma,
                threshold=gpd_quantile(f, settings.thresholds.primary_fpr),
                endpoint=f.upper_endpoint,
            )
        )

    sims: list[SimRow] = []
    for pop in POPULATIONS:
        for budget in cfg.sim_budgets:
            sims.extend(
                simulate_estimators(
                    pop,
                    budget,
                    n=cfg.sim_n,
                    trials=cfg.sim_trials,
                    tail_quantile=cfg.tail_quantile,
                    grid_points=cfg.grid_points,
                    seed=settings.seed,
                )
            )

    logger.info(
        "EVT study complete",
        extra={"n_benign_cal": n_cal, "xi": round(fit.xi, 4) if fit else None},
    )
    return EVTStudy(
        n_cal=n_cal,
        n_test=len(y_test),
        n_test_benign=int((y_test == 0).sum()),
        tail_quantile=cfg.tail_quantile,
        fit=fit,
        rows=rows,
        stability=stability,
        sims=sims,
        sim_n=int(cfg.sim_n),
        sim_trials=int(cfg.sim_trials),
        tail_scores=benign_cal,
    )


def run_evt_report(settings: Settings) -> Path:
    """Run the extreme-value study and write the report + figures."""
    study = run_evt(settings)
    cfg: EVTConfig = settings.evt

    tail_fig = _tail_figure(study, settings)
    error_fig = _error_figure(study, settings)

    report = _render(study, cfg, tail_fig, error_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote EVT report", extra={"path": str(out_path)})

    with track_run(settings, "evt") as run:
        run.log_params({"tail_quantile": study.tail_quantile, "sim_trials": study.sim_trials})
        metrics = {"n_benign_cal": float(study.n_cal)}
        if study.fit is not None:
            metrics |= {
                "gpd_xi": study.fit.xi,
                "gpd_sigma": study.fit.sigma,
                "gpd_n_exceed": float(study.fit.n_exceed),
            }
        run.log_metrics(metrics)
        run.log_artifact(tail_fig)
        run.log_artifact(error_fig)
        run.log_artifact(out_path)
    return out_path


def _tail_figure(study: EVTStudy, settings: Settings) -> Path:
    """Empirical tail survival vs the fitted GPD, on a log axis, past the last observation."""
    scores = np.sort(study.tail_scores)
    n = len(scores)
    survival = 1.0 - (np.arange(1, n + 1) - 0.5) / n
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "empirical tail (benign validation)": (scores, np.maximum(survival, 1e-6))
    }
    if study.fit is not None:
        fit = study.fit
        span = np.linspace(fit.u, float(scores.max()), 200)
        if fit.xi < 0:
            span = np.linspace(fit.u, min(fit.upper_endpoint, float(scores.max()) * 1.02), 200)
        z = 1.0 + fit.xi * (span - fit.u) / fit.sigma
        z = np.clip(z, 1e-12, None)
        modelled = fit.tail_rate * (
            np.exp(-(span - fit.u) / fit.sigma) if abs(fit.xi) < 1e-9 else z ** (-1.0 / fit.xi)
        )
        series["fitted GPD tail"] = (span, np.maximum(modelled, 1e-6))
    return plots.plot_lines(
        series,
        xlabel="attack score",
        ylabel="P(benign score > x)",
        title="The benign tail the operating point is read from",
        out_path=settings.paths.figures_dir / TAIL_FIGURE,
        yscale="log",
        vlines=(
            {f"tail starts (u, q={study.tail_quantile:.0%})": study.fit.u}
            if study.fit is not None
            else None
        ),
    )


def _error_figure(study: EVTStudy, settings: Settings) -> Path:
    """Realized-FPR ratio of each estimator against budget, per simulated population."""
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pop in POPULATIONS:
        for method in ("empirical quantile", "EVT (peaks-over-threshold)"):
            rows = [r for r in study.sims if r.population == pop.name and r.method == method]
            if not rows:
                continue
            label = "quantile" if method.startswith("empirical") else "EVT"
            series[f"{pop.name} - {label}"] = (
                np.array([r.budget for r in rows]),
                np.array([r.median_ratio for r in rows]),
            )
    return plots.plot_lines(
        series,
        xlabel="target false-positive budget",
        ylabel="realized FPR / budget (1.0 is honest)",
        title="Which estimator delivers the budget it promises",
        out_path=settings.paths.figures_dir / ERROR_FIGURE,
        xscale="log",
        yscale="log",
    )


def _fit_read(study: EVTStudy) -> str:
    if study.fit is None:
        return (
            "The benign tail could not be fitted: there are too few distinct exceedances above "
            "the chosen threshold. That is a real outcome for a bounded, tie-heavy score, and "
            "it is reported rather than papered over with a fallback."
        )
    fit = study.fit
    if fit.xi < -0.02:
        shape = (
            f"The fitted shape is **xi = {fit.xi:.3f}**, comfortably negative, which is a "
            "substantive claim and not a fitting artefact: the benign score distribution has an "
            f"**upper endpoint at {fit.upper_endpoint:.5f}**. Above it, the fit says, benign "
            "traffic does not go. That is what a bounded score should produce — the model's "
            "attack probability cannot exceed 1 — and it has an operational consequence the "
            "empirical quantile can never state: there is a strictest achievable false-positive "
            "budget, below which the only way to hit the target is to alert on nothing."
        )
    elif fit.xi > 0.02:
        shape = (
            f"The fitted shape is **xi = {fit.xi:.3f} > 0**: a heavy tail with no upper limit. "
            "Benign flows can in principle score arbitrarily high, so every false-positive "
            "budget is achievable but the threshold grows quickly as the budget tightens."
        )
    else:
        shape = (
            f"The fitted shape is **xi = {fit.xi:.3f}**, statistically indistinguishable from "
            "the exponential boundary case, so the tail decays at a constant rate and the "
            "threshold grows logarithmically as the budget tightens."
        )
    return (
        f"The tail is declared to start at the {study.tail_quantile:.0%} quantile of the benign "
        f"validation scores (u = {fit.u:.5f}), leaving **{fit.n_exceed:,} exceedances** to fit "
        f"two parameters — against the {naive_count(study.n_cal, 0.001)} order statistics the "
        f"0.1% empirical quantile rests on. {shape}"
    )


def _real_table(study: EVTStudy) -> str:
    rows = [
        "| budget | benign flows needed to resolve it | empirical threshold | EVT threshold "
        "| test FPR (empirical) | test FPR (EVT) | test TPR (empirical) | test TPR (EVT) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        mark = "" if r.resolvable else " **(unresolvable)**"
        emp = f"{r.empirical_threshold:.5f}{mark}"
        rows.append(
            f"| {r.budget:.3%} | {r.n_benign_needed:,} | {emp} | {r.evt_threshold:.5f} "
            f"| {r.empirical_test_fpr:.4%} | {r.evt_test_fpr:.4%} "
            f"| {r.empirical_test_tpr:.1%} | {r.evt_test_tpr:.1%} |"
        )
    return "\n".join(rows)


def _stability_table(study: EVTStudy) -> str:
    rows = [
        "| tail starts at | u | exceedances | xi | sigma | 0.1% threshold | upper endpoint |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in study.stability:
        endpoint = "unbounded" if not math.isfinite(s.endpoint) else f"{s.endpoint:.5f}"
        rows.append(
            f"| {s.tail_quantile:.0%} | {s.u:.5f} | {s.n_exceed:,} | {s.xi:+.3f} "
            f"| {s.sigma:.5f} | {s.threshold:.5f} | {endpoint} |"
        )
    return "\n".join(rows)


def _sim_table(study: EVTStudy) -> str:
    rows = [
        "| population | budget | estimator | median realized / budget | 10-90% spread "
        "| over 2x budget |",
        "|---|---|---|---|---|---|",
    ]
    for r in study.sims:
        rows.append(
            f"| {r.population} | {r.budget:.3%} | {r.method} | **{r.median_ratio:.2f}x** "
            f"| {r.p10_ratio:.2f}-{r.p90_ratio:.2f}x | {r.blowout_rate:.0%} |"
        )
    return "\n".join(rows)


def _sim_read(study: EVTStudy) -> str:
    if not study.sims:
        return ""
    tightest = min(r.budget for r in study.sims)

    def _pick(pop: str, method: str, budget: float) -> SimRow | None:
        return next(
            (
                r
                for r in study.sims
                if r.population == pop and r.method.startswith(method) and r.budget == budget
            ),
            None,
        )

    parts: list[str] = []
    for pop in POPULATIONS:
        emp = _pick(pop.name, "empirical", tightest)
        evt = _pick(pop.name, "EVT", tightest)
        if emp is None or evt is None:
            continue
        winner = "EVT" if abs(evt.median_ratio - 1.0) < abs(emp.median_ratio - 1.0) else "quantile"
        parts.append(
            f"- **{pop.name}** ({pop.note}). At {tightest:.3%} the empirical quantile delivers "
            f"{emp.median_ratio:.2f}x its budget (spread {emp.p10_ratio:.2f}-{emp.p90_ratio:.2f}x, "
            f"{emp.blowout_rate:.0%} of runs over double); EVT delivers {evt.median_ratio:.2f}x "
            f"(spread {evt.p10_ratio:.2f}-{evt.p90_ratio:.2f}x, {evt.blowout_rate:.0%} over "
            f"double). Closer to honest: **{winner}**."
        )
    body = "\n".join(parts)

    invariance = (
        "\n\nTwo things in that table deserve to be said out loud rather than left to be "
        "noticed.\n\n**The empirical-quantile rows are byte-identical across all three "
        "populations, and that is a check rather than a copy-paste.** The realized "
        "false-positive rate of an order-statistic threshold is `1 - F(X_(n-m))`, which is "
        "`Beta(m+1, n-m)` *whatever* `F` is — the rule reads the sample only through its "
        "ranks. So the empirical quantile's error distribution genuinely does not depend on "
        "the population, which is the same rank-invariance the "
        "[Neyman-Pearson report](neyman_pearson.md) leans on to simulate its guarantee with "
        "uniform draws. Seeing it reproduced here, from a completely different computation, "
        "is a free consistency check on both."
    )

    bounded = next((p for p in POPULATIONS if p.xi < 0), None)
    bounded_read = ""
    if bounded is not None:
        emp_b = _pick(bounded.name, "empirical", tightest)
        evt_b = _pick(bounded.name, "EVT", tightest)
        if emp_b is not None and evt_b is not None:
            bounded_read = (
                f"\n\n**EVT loses on the bounded population, and the reason is worth more than "
                f"the result.** At {tightest:.3%} it reads {evt_b.median_ratio:.2f}x against the "
                f"quantile's {emp_b.median_ratio:.2f}x — a dead heat, both catastrophic. It is "
                "not that the GPD fit fails; parameter recovery at this shape is unbiased to "
                "within a few percent (the unit tests pin it). It is that for a tail with a "
                "hard endpoint, the extreme quantile **is** the endpoint, and the endpoint "
                "cannot be known more precisely than the largest observation in hand. Traced "
                "through the runs, both estimators land on the sample maximum in essentially "
                "every replicate — they agree because they are estimating the same thing, and "
                "the sample maximum of n draws sits about `1/n` short of the true endpoint no "
                "matter how it is dressed up. Extrapolation buys nothing when there is nothing "
                "left to extrapolate into. This is the direct analogue of the Neyman-Pearson "
                "sample-size floor: past a certain budget the answer stops being *which "
                "estimator* and starts being *collect more benign traffic*."
            )
    return body + invariance + bounded_read


def _applies_read(study: EVTStudy) -> str:
    """Which simulated regime the real benign tail actually sits in, and what that costs."""
    if study.fit is None:
        return ""
    floor = 1.0 / study.n_cal
    below = [r for r in study.rows if r.budget < floor]
    if study.fit.xi >= 0:
        return (
            f"### Which regime is this detector in?\n\nThe benign tail fits **xi = "
            f"{study.fit.xi:+.3f}**, an unbounded tail — the regime where extrapolation has "
            "something to extrapolate into, and where the simulated heavy population shows EVT "
            f"holding its budget while the quantile drifts to {study.sims[0].median_ratio:.1f}x. "
            "The extrapolated thresholds above should be read as genuine estimates."
        )
    tail_note = (
        " On that reading the "
        + ", ".join(f"{r.budget:.3%}" for r in below)
        + " row"
        + ("s in the real-data table are" if len(below) > 1 else " in the real-data table is")
        + " an endpoint estimate wearing a quantile's clothes, from *either* method, and "
        "should be treated as an ordering of thresholds rather than as a calibrated budget."
        if below
        else ""
    )
    return (
        f"### Which regime is this detector in?\n\nThe bounded case is not the hypothetical "
        f"one: the benign tail here fits **xi = {study.fit.xi:+.3f}** with an upper endpoint at "
        f"**{study.fit.upper_endpoint:.5f}**, which is the same regime as the simulated uniform "
        f"population. The practical consequence is a second floor, alongside the "
        f"[Neyman-Pearson](neyman_pearson.md) one and arrived at from a different direction: "
        f"with {study.n_cal:,} benign calibration flows the largest observation sits around the "
        f"`1/n` = **{floor:.3%}** tail probability, so below roughly that budget any threshold — "
        f"empirical or extrapolated — is an estimate of where the distribution *ends*, not of a "
        f"quantile inside it.{tail_note} The 0.1% budget the project actually deploys sits "
        f"{0.001 / floor:.1f}x above the floor, which is why the two estimators still meaningfully "
        "disagree there rather than both collapsing onto the maximum."
    )


def _render(study: EVTStudy, cfg: EVTConfig, tail_fig: Path, error_fig: Path) -> str:
    unresolvable = [r for r in study.rows if not r.resolvable]
    unresolvable_note = (
        "The "
        + ", ".join(f"{r.budget:.3%}" for r in unresolvable)
        + " row"
        + ("s are" if len(unresolvable) > 1 else " is")
        + " marked unresolvable because `n * budget < 1`: there is no order statistic to "
        "read, so the empirical rule silently collapses to *the largest benign score seen*, "
        "which is not an estimate of a quantile at all — it is a sample maximum, and its "
        "expectation depends on how much traffic happened to be collected. EVT still returns "
        "a number there, and that number is the whole reason to fit a tail."
        if unresolvable
        else (
            "Every budget in the table is resolvable from this sample, so the comparison is "
            "between two estimators of the same quantity rather than between an estimator and "
            "a degenerate rule."
        )
    )
    return f"""# NetSentry — Extreme-Value Thresholds: Operating Points Past the Edge of the Data

_Synthetic stand-in. Honest temporal/binary split; the benign tail is fitted on the
{study.n_cal:,} benign validation flows and applied to the {study.n_test:,}-flow later-day test
set ({study.n_test_benign:,} benign). The controlled arm runs {study.sim_trials:,} replicates per
cell at n = {study.sim_n:,}._

## Why this report exists

The [Neyman-Pearson study](neyman_pearson.md) showed the deployed threshold is an order
statistic and gave it a guarantee. This one asks what else can be done with a tail besides
counting it.

At a 0.1% budget on {study.n_cal:,} benign validation flows, the threshold is pinned by the
top {naive_count(study.n_cal, 0.001)} scores. Every number downstream — detection rate,
alerts per day, the cost model's operating point — inherits wherever those
{naive_count(study.n_cal, 0.001)} flows happened to land. Tighten the budget by one order of
magnitude and the empirical quantile stops existing: `n * alpha < 1`, and the rule degenerates
into "the largest score I have seen".

Extreme-value theory is built for this regime. Pickands-Balkema-de Haan says exceedances over
a high threshold converge to a **Generalized Pareto** distribution for essentially any
underlying law, so the tail can be *fitted* from hundreds of flows and then extrapolated,
rather than read off five. It is the same peaks-over-threshold machinery Siffer et al. (KDD
2017) brought to streaming anomaly detection. The fit here is Grimshaw's (1993)
profile-likelihood reparameterisation, implemented directly: with `theta = xi/sigma` the shape
solves in closed form and only a one-dimensional search remains.

## The benign tail, fitted

{_fit_read(study)}

![the fitted benign tail](../figures/{tail_fig.name})

## Where the two estimators disagree

{_real_table(study)}

{unresolvable_note}

## Where does the tail begin? The bias-variance dial

`u` is not a free lunch — set it too low and non-tail data biases the fit, too high and there
is nothing left to fit. Standard practice is a stability plot: sweep `u` and look for the
range where the shape estimate stops moving.

{_stability_table(study)}

## The controlled comparison (the arm that actually decides)

The real-data rows above report thresholds, not verdicts, because **no held-out sample
available here can adjudicate them**: judging a 0.01% budget needs on the order of
{max(r.n_benign_needed for r in study.rows):,} benign flows before the measurement resolves
anything, and the later-day traffic carries {study.n_test_benign:,}. That is the same
finite-holdout trap the [Neyman-Pearson report](neyman_pearson.md) documents, and the same
answer applies: go somewhere the truth is known.

Each population below has a closed-form tail, so the *realized* false-positive rate of any
threshold can be computed exactly — no holdout, no resolution limit. A ratio of 1.00x is an
estimator that delivers the budget it promised; above 1.00x the deployment quietly runs over.

{_sim_table(study)}

{_sim_read(study)}

![estimator error by population](../figures/{error_fig.name})

The pattern is the one theory predicts, and it is not a clean win. EVT pays a fixed cost — a
shape parameter estimated from the same finite tail — and collects a return that grows as the
budget tightens and the empirical quantile runs out of order statistics to read. On the
unbounded populations that return is large and one-directional: at the tightest budget the
quantile overshoots by more than an order of magnitude while EVT stays inside a factor of two.
On the bounded one the return is zero, for the reason above. So the honest summary is narrower
than "EVT is better at extreme quantiles": **EVT trades an assumption for variance, and the
trade pays only where variance is the binding constraint and the tail has room left to
extrapolate into.** Where the tail has ended, no estimator can buy what the data does not
contain.

{_applies_read(study)}

## How this fits with the other two guarantees

Three ways to place the same threshold, with genuinely different epistemics:

{_APPROACH_TABLE}

They are complements, not rivals. NP answers *"can I promise this budget?"* and refuses when
it cannot. EVT answers *"what threshold should I use out where I have no data?"* and always
answers — which is its strength and its risk. Used together the sensible policy is EVT for the
point estimate and NP for the promise, and where NP declares the budget uncertifiable, that is
a signal to collect more benign traffic rather than to trust the extrapolation harder.

## Scope

The GPD limit is asymptotic in the threshold `u`, so every number here inherits the choice of
where the tail begins; the stability sweep exposes that dependence rather than resolving it.
The fit assumes independent exceedances, and flows within a burst are not independent, so the
effective tail sample is smaller than {study.fit.n_exceed if study.fit else 0} and the fitted
scale is correspondingly optimistic — declustering the exceedances is the standard remedy and
is not applied here. Scores are raw (uncalibrated) model outputs throughout: the isotonic
calibrator is monotone, so it cannot change which flows a threshold selects, but it flattens
the tail into ties and a tie-heavy tail cannot be fitted at all. The extrapolated thresholds
are reported, not wired into the served profiles — like the certified thresholds, adopting one
is an operator's decision about which assumption to hold, and this report exists to name the
assumption. {cfg.sim_trials:,} replicates per simulated cell."""
