"""Neyman-Pearson thresholds: a finite-sample guarantee on the false-positive budget.

Every operational number this project reports hangs off one sentence: *"the threshold is
chosen on validation at a 0.1% false-positive budget."* That sentence describes a
procedure, not a promise. The threshold is an empirical quantile of a finite benign
sample, so the rate it actually achieves on unseen benign traffic is a random variable —
and the empirical-quantile rule is not merely noisy, it is **biased above budget**. Its
population false-positive rate exceeds the target roughly half the time, which means half
of all such deployments quietly run over the alert budget they were sized for.

Neyman-Pearson classification (Cannon et al. 2002; Rigollet & Tong 2011; the umbrella
algorithm of Tong, Feng & Li, *JMLR* 2018) replaces the procedure with a guarantee: pick
the threshold as a specific **order statistic** of the benign calibration scores, chosen
so that

    P( true false-positive rate > alpha )  <=  delta

holds for a finite sample, with no assumption about the score distribution beyond
continuity. The mathematics is exact. If the threshold is the score that lets exactly `m`
of `n` benign calibration flows through, its population false-positive rate is
`Beta(m + 1, n - m)` distributed, and the tail of that Beta is a binomial CDF:

    P(FPR > alpha)  =  P( Binomial(n, alpha) <= m )

So the certified rule is simply the largest `m` whose binomial tail sits under `delta` —
a lower threshold means more detection, so we take the most permissive rule that still
certifies. Three consequences fall out, and this report measures all three:

1. **The naive rule's real promise.** Setting `m = floor(n * alpha)` (what an empirical
   quantile does) makes the binomial tail land near one half. The number is computed here
   rather than asserted.
2. **A hard floor on calibration data.** Even the most conservative rule (`m = 0`,
   threshold above *every* benign calibration score) has violation probability
   `(1 - alpha)^n`. Below `n = log(delta) / log(1 - alpha)` the budget cannot be certified
   at all, however the threshold is placed. At a 0.1% budget and 95% confidence that is
   ~3,000 benign flows — a sample-size requirement nobody states.
3. **The price, and how it decays.** The certified rule's expected false-positive rate is
   `(m + 1) / (n + 1)`, strictly below the budget, so the guarantee is bought with
   detection. That price shrinks like `1/sqrt(n)` relative to the budget, which turns
   "how much validation traffic do I need?" into an answerable engineering question.

The guarantee is over the distribution the calibration sample came from. NetSentry's
headline split is deliberately **not** exchangeable — validation is carved from Monday to
Wednesday, test is Thursday and Friday — so the report closes by applying the certified
threshold to the later days. Any excess there is not a sampling failure the guarantee was
supposed to cover; it is drift, and separating the two is the point of having a guarantee
in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import NeymanPearsonConfig

logger = get_logger(__name__)

REPORT_NAME = "neyman_pearson.md"
VIOLATION_FIGURE = "np_violation.png"
SIZE_FIGURE = "np_sample_size.png"


# --------------------------------------------------------------------------------------
# The finite-sample mathematics (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
def log_binom_cdf(n: int, p: float, m: int) -> float:
    """``log P(Binomial(n, p) <= m)``, accumulated in log space.

    The tails this report needs are astronomically small (``(1 - alpha)^n`` for
    ``n`` in the millions), so the sum is taken over log-probabilities with a
    running log-sum-exp rather than in linear space where it underflows to zero.
    No SciPy: the recurrence is three lines and keeping it explicit makes the
    guarantee auditable.
    """
    if m < 0:
        return -math.inf
    n = int(n)
    m = min(int(m), n)
    if n <= 0:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0 if m >= n else -math.inf
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_pmf = n * log_q  # j = 0
    log_cdf = log_pmf
    for j in range(1, m + 1):
        log_pmf += math.log(n - j + 1) - math.log(j) + log_p - log_q
        log_cdf = float(np.logaddexp(log_cdf, log_pmf))
    return log_cdf


def violation_probability(n: int, alpha: float, m: int) -> float:
    """``P(true FPR > alpha)`` for the rule that lets ``m`` of ``n`` benign flows through.

    The threshold is the ``(n - m)``-th smallest benign calibration score, so the fraction
    of the benign population above it is ``Beta(m + 1, n - m)`` distributed. The upper tail
    of that Beta is exactly ``P(Binomial(n, alpha) <= m)`` — which is what makes this a
    finite-sample, distribution-free statement rather than an asymptotic one.
    """
    if n <= 0:
        return 1.0
    m = int(np.clip(m, 0, n))
    return float(math.exp(log_binom_cdf(n, alpha, m)))


def np_admissible_count(n: int, alpha: float, delta: float) -> int | None:
    """Most permissive certified rule: the largest ``m`` with violation probability <= ``delta``.

    Larger ``m`` means a lower threshold and more detection, so the *largest* admissible
    count is the one to deploy. Returns ``None`` when the calibration sample is too small
    for any threshold to certify the budget (see :func:`min_calibration_size`).
    The scan is incremental — one binomial term per step — so it stays linear even when the
    answer is in the thousands.
    """
    n = int(n)
    if n <= 0 or not (0.0 < alpha < 1.0) or not (0.0 < delta < 1.0):
        return None
    log_delta = math.log(delta)
    log_p = math.log(alpha)
    log_q = math.log1p(-alpha)
    log_pmf = n * log_q
    log_cdf = log_pmf
    if log_cdf > log_delta:
        return None  # even the most conservative threshold cannot certify this budget
    for j in range(1, n + 1):
        log_pmf += math.log(n - j + 1) - math.log(j) + log_p - log_q
        nxt = float(np.logaddexp(log_cdf, log_pmf))
        if nxt > log_delta:
            return j - 1
        log_cdf = nxt
    return n


def min_calibration_size(alpha: float, delta: float) -> int:
    """Smallest benign calibration sample for which *any* certified threshold exists.

    The most conservative rule places the threshold above every calibration score
    (``m = 0``) and still violates the budget with probability ``(1 - alpha)^n``. Solving
    that for ``n`` gives a hard floor: at a 0.1% budget and 95% confidence, ~3,000 benign
    flows. Below it the honest answer is that the budget cannot be certified, not that the
    threshold should be nudged.
    """
    return math.ceil(math.log(delta) / math.log1p(-alpha))


def naive_count(n: int, alpha: float) -> int:
    """Benign calibration flows the plain empirical-quantile rule lets through."""
    return math.floor(int(n) * alpha)


def expected_fpr(n: int, m: int) -> float:
    """Mean of the ``Beta(m + 1, n - m)`` population FPR: ``(m + 1) / (n + 1)``.

    Reading it for the naive rule is instructive on its own — ``(floor(n*alpha) + 1)/(n+1)``
    sits *above* ``alpha``, so the empirical quantile is biased over budget by construction.
    """
    if n <= 0:
        return 1.0
    return (int(m) + 1) / (int(n) + 1)


def threshold_from_count(benign_scores: np.ndarray, m: int) -> float:
    """Score threshold that lets exactly ``m`` of the benign calibration flows through.

    Decisions use **strict** exceedance (``score > threshold``): the threshold is itself an
    observed benign score, so an inclusive rule would alert on it and let ``m + 1`` through,
    quietly invalidating the count the guarantee is stated in.
    """
    scores = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(scores)
    if n == 0:
        return math.inf
    m = int(np.clip(m, 0, n - 1))
    return float(scores[n - m - 1])


def rates_above(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """``(TPR, FPR)`` under the strict-exceedance rule the order-statistic guarantee describes."""
    y = np.asarray(y_true).astype(int)
    alerted = np.asarray(scores, dtype=float) > threshold
    attacks = y == 1
    benign = ~attacks
    tpr = float(alerted[attacks].mean()) if attacks.any() else 0.0
    fpr = float(alerted[benign].mean()) if benign.any() else 0.0
    return tpr, fpr


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class ThresholdRule:
    """One threshold-selection rule, its promise, and what it did on the later days."""

    name: str
    budget: float
    count: int
    threshold: float
    violation_prob: float
    expected_fpr: float
    test_tpr: float
    test_fpr: float


@dataclass
class DeltaPoint:
    """The certified rule at one confidence level."""

    delta: float
    feasible: bool
    count: int
    expected_fpr: float
    violation_prob: float
    test_tpr: float
    test_fpr: float


@dataclass
class SizePoint:
    """What a calibration sample of size ``n`` buys, computed from the theory alone."""

    n: int
    feasible: bool
    naive_count: int
    np_count: int
    naive_expected_fpr: float
    np_expected_fpr: float
    naive_violation: float


@dataclass
class EmpiricalArm:
    """One rule checked two ways: against the population truth, and against a finite holdout."""

    budget: float
    rule: str
    count: int
    n_cal: int
    n_hold: int
    expected_holdout_fp: float
    predicted: float
    exact_rate: float
    holdout_rate: float
    n_sims: int
    n_splits: int


@dataclass
class NPStudy:
    """Everything the report renders."""

    delta: float
    n_cal: int
    n_test: int
    n_test_benign: int
    primary_budget: float
    rules: list[ThresholdRule]
    deltas: list[DeltaPoint]
    sizes: list[SizePoint]
    empirical: list[EmpiricalArm]
    min_sizes: dict[float, int]


def _rule(
    name: str,
    budget: float,
    count: int,
    cal_scores: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    n_cal: int,
) -> ThresholdRule:
    """Materialise one counting rule into a threshold and score it on the test days."""
    threshold = threshold_from_count(cal_scores, count)
    tpr, fpr = rates_above(y_test, s_test, threshold)
    return ThresholdRule(
        name=name,
        budget=budget,
        count=count,
        threshold=threshold,
        violation_prob=violation_probability(n_cal, budget, count),
        expected_fpr=expected_fpr(n_cal, count),
        test_tpr=tpr,
        test_fpr=fpr,
    )


def simulate_violation_rate(n_cal: int, alpha: float, m: int, *, n_sims: int, seed: int) -> float:
    """Monte-Carlo the guarantee exactly, by simulating ranks rather than scores.

    The rule reads the calibration sample **only through its order statistics**, so its
    violation probability is invariant under any strictly increasing transform of the
    score. Uniform draws are therefore not a toy stand-in — they are the same experiment,
    which is precisely what "distribution-free" buys. Each replicate draws ``n_cal``
    uniforms, places the threshold that lets ``m`` through, and reads the *population*
    false-positive rate exactly (``1 - t`` for uniforms), so no finite holdout stands
    between the simulation and the quantity the bound is about.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random((int(n_sims), int(n_cal)))
    thresholds = np.sort(draws, axis=1)[:, int(n_cal) - int(m) - 1]
    return float(np.mean((1.0 - thresholds) > alpha))


def holdout_violation_rate(
    pool: np.ndarray, alpha: float, m: int, n_cal: int, *, n_splits: int, seed: int
) -> tuple[float, float]:
    """What a practitioner would measure: repeated cal/holdout splits of the real scores.

    Returns ``(violation rate, mean holdout FPR)``. This is deliberately the *wrong*
    instrument for the job — a finite holdout estimates the realized FPR with error, and
    that error leaks into the violation count — but it is the check almost anyone would
    run, so the report measures how far it misleads instead of asserting that it does.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(pool, dtype=float)
    n_total = len(scores)
    n_cal = int(min(n_cal, max(n_total - 1, 1)))
    fprs: list[float] = []
    for _ in range(int(n_splits)):
        perm = rng.permutation(n_total)
        cal = scores[perm[:n_cal]]
        hold = scores[perm[n_cal:]]
        if len(hold) == 0:
            continue
        fprs.append(float(np.mean(hold > threshold_from_count(cal, m))))
    if not fprs:
        return 0.0, 0.0
    arr = np.asarray(fprs)
    return float(np.mean(arr > alpha)), float(np.mean(arr))


def _empirical_arms(
    budget: float,
    benign_pool: np.ndarray,
    cfg: NeymanPearsonConfig,
    seed: int,
) -> list[EmpiricalArm]:
    """Check both rules at ``budget`` against the population truth and a finite holdout."""
    pool = np.asarray(benign_pool, dtype=float)
    n_total = len(pool)
    n_cal = min(int(cfg.split_calibration_size), max(n_total // 2, 1))
    n_hold = n_total - n_cal
    rules: list[tuple[str, int | None]] = [
        ("empirical quantile", naive_count(n_cal, budget)),
        ("certified", np_admissible_count(n_cal, budget, cfg.delta)),
    ]
    arms: list[EmpiricalArm] = []
    for name, m in rules:
        if m is None:
            continue
        exact = simulate_violation_rate(n_cal, budget, m, n_sims=cfg.n_sims, seed=seed)
        measured, _mean_fpr = holdout_violation_rate(
            pool, budget, m, n_cal, n_splits=cfg.n_splits, seed=seed
        )
        arms.append(
            EmpiricalArm(
                budget=budget,
                rule=name,
                count=m,
                n_cal=n_cal,
                n_hold=n_hold,
                expected_holdout_fp=n_hold * budget,
                predicted=violation_probability(n_cal, budget, m),
                exact_rate=exact,
                holdout_rate=measured,
                n_sims=int(cfg.n_sims),
                n_splits=int(cfg.n_splits),
            )
        )
    return arms


def run_neyman_pearson(settings: Settings) -> NPStudy:
    """Certify the operating point on the honest split and price the guarantee."""
    cfg: NeymanPearsonConfig = settings.neyman_pearson
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    result = fit_supervised(variant)
    benign_label = variant.labels.benign_label
    # Raw (uncalibrated) scores: the calibrator is monotone, so it cannot change which
    # flows a threshold selects, but it creates ties that would corrupt an order statistic.
    s_val = attack_probability(np.asarray(result.proba_val), result.classes, benign_label)
    s_test = attack_probability(np.asarray(result.proba_test), result.classes, benign_label)
    y_val = np.asarray(result.y_val).astype(int)
    y_test = np.asarray(result.y_test).astype(int)

    benign_cal = s_val[y_val == 0]
    n_cal = len(benign_cal)
    budgets = list(
        dict.fromkeys([*settings.thresholds.fpr_targets, settings.thresholds.primary_fpr])
    )
    primary = settings.thresholds.primary_fpr

    rules: list[ThresholdRule] = []
    for budget in budgets:
        naive_m = naive_count(n_cal, budget)
        rules.append(
            _rule(
                f"empirical quantile @ {budget:.1%}",
                budget,
                naive_m,
                benign_cal,
                y_test,
                s_test,
                n_cal,
            )
        )
        np_m = np_admissible_count(n_cal, budget, cfg.delta)
        if np_m is None:
            continue
        rules.append(
            _rule(f"certified @ {budget:.1%}", budget, np_m, benign_cal, y_test, s_test, n_cal)
        )

    deltas: list[DeltaPoint] = []
    for d in cfg.delta_sweep:
        m = np_admissible_count(n_cal, primary, d)
        if m is None:
            deltas.append(DeltaPoint(d, False, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        threshold = threshold_from_count(benign_cal, m)
        tpr, fpr = rates_above(y_test, s_test, threshold)
        deltas.append(
            DeltaPoint(
                delta=d,
                feasible=True,
                count=m,
                expected_fpr=expected_fpr(n_cal, m),
                violation_prob=violation_probability(n_cal, primary, m),
                test_tpr=tpr,
                test_fpr=fpr,
            )
        )

    sizes: list[SizePoint] = []
    for n in cfg.calibration_sizes:
        m = np_admissible_count(n, primary, cfg.delta)
        naive_m = naive_count(n, primary)
        sizes.append(
            SizePoint(
                n=int(n),
                feasible=m is not None,
                naive_count=naive_m,
                np_count=int(m) if m is not None else 0,
                naive_expected_fpr=expected_fpr(n, naive_m),
                np_expected_fpr=expected_fpr(n, int(m)) if m is not None else 0.0,
                naive_violation=violation_probability(n, primary, naive_m),
            )
        )

    empirical = [
        arm for budget in budgets for arm in _empirical_arms(budget, benign_cal, cfg, settings.seed)
    ]

    logger.info(
        "Neyman-Pearson calibration complete",
        extra={"n_benign_cal": n_cal, "delta": cfg.delta, "budgets": len(budgets)},
    )
    return NPStudy(
        delta=cfg.delta,
        n_cal=n_cal,
        n_test=len(y_test),
        n_test_benign=int((y_test == 0).sum()),
        primary_budget=primary,
        rules=rules,
        deltas=deltas,
        sizes=sizes,
        empirical=empirical,
        min_sizes={b: min_calibration_size(b, cfg.delta) for b in budgets},
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_neyman_pearson_report(settings: Settings) -> Path:
    """Run the Neyman-Pearson study and write the report + figures."""
    study = run_neyman_pearson(settings)
    cfg = settings.neyman_pearson
    primary = study.primary_budget

    naive_m = naive_count(study.n_cal, primary)
    np_m = np_admissible_count(study.n_cal, primary, study.delta)
    span = max(2 * max(naive_m, 1), (np_m or 0) + 2, 12)
    counts = np.arange(0, span + 1)
    curve = np.array([violation_probability(study.n_cal, primary, int(m)) for m in counts])
    vlines = {f"empirical quantile (m={naive_m})": float(naive_m)}
    if np_m is not None:
        vlines[f"certified (m={np_m})"] = float(np_m)
    violation_fig = plots.plot_lines(
        {f"P(true FPR > {primary:.1%})": (counts, curve)},
        xlabel="benign calibration flows allowed above the threshold (m)",
        ylabel="probability the budget is exceeded",
        title=f"What each threshold actually promises (n={study.n_cal:,} benign)",
        out_path=settings.paths.figures_dir / VIOLATION_FIGURE,
        vlines=vlines,
    )

    ns = np.array([p.n for p in study.sizes], dtype=float)
    size_fig = plots.plot_lines(
        {
            "certified rule (expected FPR / budget)": (
                ns,
                np.array([p.np_expected_fpr / primary if p.feasible else 0.0 for p in study.sizes]),
            ),
            "empirical quantile (expected FPR / budget)": (
                ns,
                np.array([p.naive_expected_fpr / primary for p in study.sizes]),
            ),
        },
        xlabel="benign calibration flows (n)",
        ylabel=f"expected FPR as a fraction of the {primary:.1%} budget",
        title="How much validation traffic a certified budget costs",
        out_path=settings.paths.figures_dir / SIZE_FIGURE,
        xscale="log",
        vlines={f"floor: n = {study.min_sizes[primary]:,}": float(study.min_sizes[primary])},
    )

    report = _render(study, cfg, violation_fig, size_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote Neyman-Pearson report", extra={"path": str(out_path)})

    with track_run(settings, "neyman_pearson") as run:
        run.log_params({"delta": study.delta, "primary_fpr": primary, "n_benign_cal": study.n_cal})
        certified = next((r for r in study.rules if r.name.startswith("certified")), None)
        naive = next((r for r in study.rules if r.name.startswith("empirical")), None)
        metrics = {"min_calibration_size": float(study.min_sizes[primary])}
        if naive is not None:
            metrics["naive_violation_prob"] = naive.violation_prob
            metrics["naive_test_tpr"] = naive.test_tpr
        if certified is not None:
            metrics["certified_violation_prob"] = certified.violation_prob
            metrics["certified_test_tpr"] = certified.test_tpr
        run.log_metrics(metrics)
        run.log_artifact(violation_fig)
        run.log_artifact(size_fig)
        run.log_artifact(out_path)
    return out_path


def _rule_table(study: NPStudy) -> str:
    rows = [
        "| rule | budget | m | threshold | P(FPR > budget) | expected FPR | test TPR | test FPR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in study.rules:
        rows.append(
            f"| {r.name} | {r.budget:.1%} | {r.count} | {r.threshold:.5f} "
            f"| **{r.violation_prob:.1%}** | {r.expected_fpr:.4%} | {r.test_tpr:.1%} "
            f"| {r.test_fpr:.4%} |"
        )
    return "\n".join(rows)


def _delta_table(study: NPStudy) -> str:
    rows = [
        "| delta (tolerated violation) | certified m | expected FPR | actual P(FPR > budget) "
        "| test TPR | test FPR |",
        "|---|---|---|---|---|---|",
    ]
    for d in study.deltas:
        if not d.feasible:
            rows.append(f"| {d.delta:.0%} | infeasible | — | — | — | — |")
            continue
        rows.append(
            f"| {d.delta:.0%} | {d.count} | {d.expected_fpr:.4%} | {d.violation_prob:.2%} "
            f"| {d.test_tpr:.1%} | {d.test_fpr:.4%} |"
        )
    return "\n".join(rows)


def _size_table(study: NPStudy) -> str:
    rows = [
        "| benign calibration flows | certified m | certified expected FPR | quantile expected FPR "
        "| quantile P(over budget) |",
        "|---|---|---|---|---|",
    ]
    for p in study.sizes:
        certified = (
            f"{p.np_expected_fpr:.4%} ({p.np_expected_fpr / study.primary_budget:.0%} of budget)"
            if p.feasible
            else "**cannot certify**"
        )
        m = str(p.np_count) if p.feasible else "—"
        rows.append(
            f"| {p.n:,} | {m} | {certified} | {p.naive_expected_fpr:.4%} "
            f"| {p.naive_violation:.1%} |"
        )
    return "\n".join(rows)


def _empirical_table(study: NPStudy) -> str:
    rows = [
        "| budget | rule | m | closed form | rank simulation (population FPR) "
        "| finite-holdout measurement |",
        "|---|---|---|---|---|---|",
    ]
    for arm in study.empirical:
        rows.append(
            f"| {arm.budget:.1%} | {arm.rule} | {arm.count} | {arm.predicted:.1%} "
            f"| {arm.exact_rate:.1%} | {arm.holdout_rate:.1%} |"
        )
    return "\n".join(rows)


def _headline(study: NPStudy) -> str:
    primary = study.primary_budget
    naive = next(
        (r for r in study.rules if r.budget == primary and r.name.startswith("empir")), None
    )
    cert = next((r for r in study.rules if r.budget == primary and r.name.startswith("cert")), None)
    if naive is None:
        return "No rule could be formed on the available benign calibration sample."
    lead = (
        f"On {study.n_cal:,} benign validation flows, the empirical-quantile threshold this "
        f"project reports its headline detection rate at lets {naive.count} of them through — "
        f"and the probability that its **true** false-positive rate exceeds the "
        f"{primary:.1%} budget is **{naive.violation_prob:.1%}**. That is not a rounding "
        f"concern. Its expected false-positive rate is {naive.expected_fpr:.4%}, which is "
        f"{naive.expected_fpr / primary:.2f}x the budget: the empirical quantile is biased "
        "over budget by construction, because the order statistic it lands on is the one that "
        "*just* fits the sample, and the population is larger than the sample."
    )
    if cert is None:
        return (
            lead + f" At {study.delta:.0%} confidence no threshold on this sample can certify the "
            f"budget at all — the floor is {study.min_sizes[primary]:,} benign flows and there "
            f"are {study.n_cal:,}. The honest conclusion is that this operating point is not "
            "certifiable with the validation traffic available, which is a finding about the "
            "deployment, not a defect in the method."
        )
    price = naive.test_tpr - cert.test_tpr
    return (
        lead
        + f" The certified rule admits only {cert.count} benign calibration flows, which pins the "
        f"violation probability at {cert.violation_prob:.2%} — inside the {study.delta:.0%} "
        f"promise — and holds an expected false-positive rate of {cert.expected_fpr:.4%}, "
        f"{cert.expected_fpr / primary:.0%} of budget. The guarantee is not free: detection on "
        f"the later days falls from {naive.test_tpr:.1%} to {cert.test_tpr:.1%}, a "
        f"{price:.1%}-point price for knowing the budget holds."
    )


def _empirical_read(study: NPStudy) -> str:
    if not study.empirical:
        return ""
    arms = study.empirical
    exact_gap = max(abs(a.exact_rate - a.predicted) for a in arms)
    certified = [a for a in arms if a.rule == "certified"]
    worst = max(certified, key=lambda a: a.holdout_rate - a.exact_rate) if certified else arms[0]
    inflation = worst.holdout_rate - worst.exact_rate
    budgets = {a.budget for a in arms}
    uncertifiable = sorted(budgets - {a.budget for a in certified})
    missing = (
        " The "
        + ", ".join(f"{b:.1%}" for b in uncertifiable)
        + f" budget has no certified row at all: this arm calibrates on {worst.n_cal:,} flows, "
        f"below the {min_calibration_size(uncertifiable[0], study.delta):,} the floor demands, "
        "so there is nothing to validate. The floor is not a formality."
        if uncertifiable
        else ""
    )
    verdict = (
        f"The rank simulation reproduces the closed form to within {exact_gap:.1%} on every "
        "row. That is the self-validation that counts: an off-by-one in the order statistic "
        "would appear as a systematic offset, not as Monte-Carlo noise."
        if exact_gap < 0.02
        else (
            f"The rank simulation and the closed form differ by up to {exact_gap:.1%}, which is "
            "more than Monte-Carlo noise should produce and would be the first place to look "
            "for an off-by-one in the order statistic."
        )
    )
    return (
        f"Two checks, deliberately different. The **rank simulation** draws {worst.n_sims:,} "
        f"replicate calibration samples of {worst.n_cal:,} flows and reads the *population* "
        "false-positive rate exactly. It is legitimate to simulate this with uniform draws "
        "rather than real scores because the rule touches the data only through its order "
        "statistics, so its violation probability is invariant to any strictly increasing "
        f'transform of the score — that invariance is exactly what "distribution-free" '
        f"means here. {verdict}\n\n"
        f"The **finite-holdout measurement** is the check a practitioner would actually run: "
        f"split the benign pool {worst.n_splits} times, calibrate on one half, and count how "
        f"often the other half's *measured* FPR lands over budget. It disagrees — and the "
        f"direction is the point. For the certified rule at {worst.budget:.1%} it reads "
        f"{worst.holdout_rate:.1%} against a true {worst.exact_rate:.1%}, an inflation of "
        f"{inflation:.1%} that would look like the guarantee failing its {study.delta:.0%} "
        f"promise. It is not failing. A {worst.n_hold:,}-flow holdout carries only "
        f"{worst.expected_holdout_fp:.1f} expected false positives, so its FPR estimate is "
        "noisy, and a certified rule sits *below* budget by design — which means holdout noise "
        "can only push the estimate across the line, never back. **A finite holdout cannot "
        "validate a finite-sample bound**, because it is the same finite-sample regime the "
        "bound exists to handle. The empirical-quantile rows show the mirror image: their true "
        "rate straddles the budget, so the same noise is roughly symmetric and the measurement "
        "reads close to the truth. This is worth stating plainly because the wrong check is the "
        "natural one to run, and it condemns the correct method." + missing
    )


def _drift_read(study: NPStudy) -> str:
    primary = study.primary_budget
    cert = next((r for r in study.rules if r.budget == primary and r.name.startswith("cert")), None)
    naive = next(
        (r for r in study.rules if r.budget == primary and r.name.startswith("empir")), None
    )
    if cert is None or naive is None:
        return (
            "With no certifiable rule at this sample size there is no guarantee to test against "
            "the later days."
        )
    over = cert.test_fpr > primary
    if over:
        return (
            f"The certified threshold was built to hold {primary:.1%} with "
            f"{1 - study.delta:.0%} confidence, and on Thursday-Friday it realizes "
            f"{cert.test_fpr:.4%} — **{cert.test_fpr / primary:.1f}x the budget**. The guarantee "
            "is not broken; it was never a guarantee about these days. It covers sampling error "
            "within the distribution the calibration flows came from, and Monday-Wednesday is a "
            "different distribution from Thursday-Friday. That is exactly the value of stating "
            "the guarantee explicitly: the excess can no longer be blamed on an unlucky "
            "validation draw, so it has to be attributed to drift — which is what the "
            "[drift](drift.md), [covariate-shift](covariate_shift.md) and "
            "[refresh](refresh.md) studies then go and measure. A threshold with no stated "
            "guarantee cannot make that distinction, which is how over-budget deployments get "
            "explained away as noise."
        )
    return (
        f"Applied to Thursday-Friday the certified threshold realizes {cert.test_fpr:.4%} against "
        f"a {primary:.1%} budget, so the guarantee survives the day boundary here — the benign "
        "score distribution is stable enough across the temporal gap that a bound proved on "
        "Monday-Wednesday still binds. That is a stronger statement than the guarantee itself "
        "makes, and it is contingent on the data, not on the method: the "
        "[covariate-shift](covariate_shift.md) study measures how far the benign distribution "
        "actually moves, and a deployment whose benign traffic moves more would need the "
        "threshold re-certified on fresher flows ([refresh](refresh.md) prices exactly that)."
    )


def _render(study: NPStudy, cfg: NeymanPearsonConfig, violation_fig: Path, size_fig: Path) -> str:
    primary = study.primary_budget
    floors = ", ".join(
        f"{b:.1%} needs {n:,} benign flows" for b, n in sorted(study.min_sizes.items())
    )
    return f"""# NetSentry — Neyman-Pearson Thresholds: Certifying the False-Positive Budget

_Synthetic stand-in. Honest temporal/binary split; thresholds calibrated on the
{study.n_cal:,} benign validation flows and applied to the {study.n_test:,}-flow later-day test
set ({study.n_test_benign:,} benign). Confidence level delta = {study.delta:.0%}: the certified
rule may exceed its budget with probability at most {study.delta:.0%}._

## Why this report exists

Every operational claim in this project rests on one sentence — *"the threshold is chosen on
validation at a {primary:.1%} false-positive budget."* That sentence describes a **procedure**,
not a promise. The threshold is an empirical quantile of a finite benign sample, so the rate it
achieves on traffic it has not seen is a random variable. Worse, it is a *biased* one: the
quantile lands on the order statistic that just fits the sample, and the population is bigger
than the sample.

Neyman-Pearson classification (Cannon et al. 2002; Rigollet & Tong 2011; Tong, Feng & Li,
JMLR 2018) replaces the procedure with a guarantee. If the threshold lets exactly `m` of `n`
benign calibration flows through, the fraction of the benign *population* above it is
`Beta(m + 1, n - m)` distributed, whose upper tail is exactly a binomial CDF:

```
P( true FPR > alpha )  =  P( Binomial(n, alpha) <= m )
```

Choose the largest `m` whose tail sits under `delta` and the resulting classifier satisfies
`P(FPR > alpha) <= delta` for a finite sample, with no distributional assumption beyond a
continuous score. Everything below follows from that one identity.

## What the deployed threshold actually promises

{_rule_table(study)}

{_headline(study)}

![violation probability vs threshold rank](../figures/{violation_fig.name})

The curve is the whole argument in one line: violation probability rises steeply with how many
benign calibration flows the threshold admits, the empirical quantile sits near the middle of
that rise, and the certified rule is the last point under `delta`.

## Buying confidence: the delta sweep

{_delta_table(study)}

Confidence is not free and it is not linear. Tightening `delta` moves the threshold up a small
number of order statistics, and each step costs detection on the later days. The table is the
honest menu — an operator who needs 99% confidence that the alert budget holds can read the
detection they are giving up to get it, rather than discovering it after the queue overflows.

## The floor nobody states: how much benign traffic a guarantee needs

Even the most conservative rule — a threshold above *every* benign calibration score, `m = 0` —
still exceeds the budget with probability `(1 - alpha)^n`. Setting that equal to `delta` gives a
hard sample-size floor below which the budget **cannot be certified at any threshold**:
{floors}.

{_size_table(study)}

![the price of a certified budget vs calibration size](../figures/{size_fig.name})

Both rules converge on the budget from opposite sides as `n` grows — the quantile from above
(it is biased over budget) and the certified rule from below (it pays for its confidence). The
gap between them is the price of the guarantee, and it closes like `1/sqrt(n)`. This turns a
vague instinct ("more validation data is better") into a sizing requirement: to certify a
{primary:.1%} budget at {study.delta:.0%} confidence while giving up less than a tenth of the
budget in detection, read across the table for the `n` where the certified column reaches ~90%
of budget.

## Does the closed form survive contact with a measurement?

{_empirical_table(study)}

{_empirical_read(study)}

## The guarantee meets the temporal gap

{_drift_read(study)}

## Scope

The guarantee is **distribution-free but not assumption-free**: it needs the benign calibration
scores to be i.i.d. draws from the same distribution as the benign traffic being judged, and a
continuous score so no two flows tie on the threshold (raw model scores are used throughout for
this reason — the isotonic calibrator is monotone, so it cannot change *which* flows a threshold
selects, but it creates ties that would corrupt an order statistic). Flows within one attack
burst are not independent, so the effective sample size is below the nominal one and the true
violation probability is somewhat higher than the number printed here — a caveat this dataset
shares with every finite-sample bound applied to network traffic. The rule certifies the
false-positive rate only; detection is whatever falls out, which is the correct asymmetry for a
SOC (the budget is the binding constraint) and the wrong one for a setting where misses dominate
— [cost.md](cost.md) takes the other side and optimises the decision economics directly, and
[conformal.md](conformal.md) gives the complementary guarantee on the *label set* rather than on
the error rate. The certified thresholds here are computed, reported and priced, but not wired
into the served bundle's threshold profiles: swapping the deployed operating point is a decision
about how much detection the operator will trade for a guarantee, and the point of this report
is to put that trade on the table with a number attached rather than to make it silently.
{cfg.n_sims:,} rank-space replicates and {cfg.n_splits} calibration/holdout re-draws stand behind
the validation section."""
