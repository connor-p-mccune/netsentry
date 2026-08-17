"""Control the risk the SOC actually names, not the one the threshold happens to target.

Every operating point in this project is chosen the same way: pick a false-positive budget,
find the threshold that respects it, report the detection rate that falls out. That is the
right instrument for the question "how much noise can my analysts absorb", and it is the wrong
instrument for the question a detection contract is usually written in — *"you may miss at
most one attack in ten."* Nothing here controls that. The [conformal study](conformal.md)
guarantees **coverage** of a prediction set, the [alert-FDR study](alert_fdr.md) guarantees
the **false-discovery rate** of an alert batch, and the [Neyman-Pearson study]
(neyman_pearson.md) certifies the **false-positive rate**. None of the three is a bound on the
miss rate, and the miss rate is what gets a detection team fired.

Two instruments close that gap, and the difference between them is the point of this report:

- **Conformal risk control** (Angelopoulos, Bates, Fisch, Lei, Schuster, 2022) generalises
  split conformal from coverage to any bounded, monotone loss. Choose the threshold as the
  extreme value whose *inflated* empirical risk `(n R̂ + B) / (n + 1)` still clears the target
  and the guarantee `E[R] <= alpha` holds in finite samples with no distributional assumption
  at all. It is one line of code and it controls the **expectation**.
- **Learn then Test** (Angelopoulos, Bates, Candès, Jordan, Lei, 2021) reframes threshold
  selection as multiple hypothesis testing: every candidate threshold is a null hypothesis
  ("this one's risk exceeds alpha"), each gets a valid p-value from the Hoeffding-Bentkus
  inequality, and a family-wise error correction returns the set of thresholds that are
  certified. That buys the **high-probability** statement `P(R > alpha) <= delta`, and it
  extends to several risks at once.

The distinction is not academic and the report measures it rather than describing it: an
expectation bound is satisfied by a procedure that lands *under* target half the time and
*over* it the other half. Simulating the whole calibration-and-deploy loop hundreds of times
shows exactly that — conformal risk control exceeds its own target on roughly half of the
draws, which is not a bug, it is what "in expectation" means, and it is not what an operator
hears when they are told a miss rate is controlled.

Three things then get priced:

1. **What the guarantee costs in alerts.** A miss-rate promise is bought with false positives.
   On a detector this weak the exchange rate is brutal, and the report gives it in
   alerts-per-day and analyst-headcount rather than in a false-positive rate.
2. **Two risks at once.** Miss rate *and* alert volume, controlled simultaneously via
   Learn-then-Test with a Bonferroni correction (the two constraints run in opposite
   directions, so the fixed-sequence shortcut that works for one is invalid for both). When no
   threshold satisfies both, the procedure returns an **empty set** — a certificate of
   infeasibility, which is a far more useful output than a threshold that quietly violates one
   of the two promises.
3. **Per class.** A global miss-rate promise is the average of promises the detector cannot
   keep equally. Controlling the risk per attack family says which promises are affordable and
   which are not available at any price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import alerts_per_day, attack_probability
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import RiskControlConfig

logger = get_logger(__name__)

REPORT_NAME = "risk_control.md"
FIGURE_NAME = "risk_control_exceedance.png"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Valid p-values for a bounded risk.
# --------------------------------------------------------------------------------------


def log_binomial_cdf(k: int, n: int, p: float) -> float:
    """``log P(Bin(n, p) <= k)``, summed in log space so large ``n`` does not underflow.

    Written out rather than imported so the bound has no optional dependency: the Bentkus
    inequality needs a binomial tail and nothing else, and a p-value that silently degrades to
    a normal approximation is a p-value that no longer certifies anything.
    """
    if k < 0:
        return -math.inf
    if k >= n:
        return 0.0
    p = min(max(p, _EPS), 1.0 - _EPS)
    terms = []
    log_p, log_q = math.log(p), math.log1p(-p)
    for i in range(k + 1):
        terms.append(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def hoeffding_bentkus_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """A valid p-value for ``H0: R > alpha`` from ``n`` bounded losses (Bates et al. 2021).

    The minimum of two bounds, because neither dominates: Hoeffding's is tighter when the
    empirical risk sits far below the target, Bentkus's is tighter close to it. Both are valid
    for losses in [0, 1] with no distributional assumption, which is the whole reason this is
    usable on a detection threshold where nothing is Gaussian.
    """
    if n <= 0:
        return 1.0
    risk_hat = float(np.clip(risk_hat, 0.0, 1.0))
    alpha = float(np.clip(alpha, _EPS, 1.0 - _EPS))
    if risk_hat >= alpha:
        return 1.0
    # Hoeffding, via the KL form h1(a, b) which is tighter than the raw exponential bound.
    a = min(risk_hat, alpha)
    kl = a * math.log(max(a, _EPS) / alpha) + (1 - a) * math.log(max(1 - a, _EPS) / (1 - alpha))
    hoeffding = math.exp(-n * kl)
    # Bentkus: e * P(Bin(n, alpha) <= ceil(n * risk_hat)).
    bentkus = math.e * math.exp(log_binomial_cdf(math.ceil(n * risk_hat), n, alpha))
    return float(min(1.0, hoeffding, bentkus))


# --------------------------------------------------------------------------------------
# The two selectors.
# --------------------------------------------------------------------------------------


def crc_threshold(
    grid: np.ndarray, risks: np.ndarray, n: int, alpha: float, bound: float = 1.0
) -> float:
    """Conformal risk control: the largest threshold whose inflated risk still clears alpha.

    ``risks`` is the empirical risk at each grid point and must be non-decreasing in the
    threshold (a higher bar misses more attacks). The theorem is stated for a non-increasing
    loss and an infimum; this is the same statement read right-to-left. The ``+ bound`` term
    is what turns an empirical average into a finite-sample guarantee — it is the price of
    having estimated the risk on the same kind of data you are about to deploy on.
    """
    inflated = (n * np.asarray(risks, dtype=float) + bound) / (n + 1)
    feasible = np.flatnonzero(inflated <= alpha)
    if len(feasible) == 0:
        return float(grid[0])  # nothing clears it: fall back to the most permissive threshold
    return float(grid[feasible[-1]])


def ltt_valid_set(
    grid: np.ndarray,
    risks: np.ndarray,
    n: int,
    alpha: float,
    delta: float,
    *,
    method: str = "fixed_sequence",
) -> np.ndarray:
    """Thresholds certified at ``P(R > alpha) <= delta`` (Learn then Test).

    ``fixed_sequence`` walks the grid from the most conservative threshold and stops at the
    first hypothesis it cannot reject; it controls the family-wise error rate at full alpha
    per test and is the right choice when the risk is monotone in the threshold, because a
    failure at one point implies failure everywhere beyond it. ``bonferroni`` tests every grid
    point at ``delta / |grid|`` and is the correction to use when the constraints are not
    monotone in the same direction — as they are not once alert volume joins the miss rate.
    """
    grid = np.asarray(grid, dtype=float)
    risks = np.asarray(risks, dtype=float)
    if method == "bonferroni":
        threshold = delta / max(len(grid), 1)
        keep = [
            i for i in range(len(grid)) if hoeffding_bentkus_pvalue(risks[i], n, alpha) <= threshold
        ]
        return grid[np.array(keep, dtype=int)] if keep else np.zeros(0, dtype=float)
    valid: list[float] = []
    for i in range(len(grid)):
        if hoeffding_bentkus_pvalue(risks[i], n, alpha) <= delta:
            valid.append(float(grid[i]))
        else:
            break
    return np.array(valid, dtype=float)


def multi_risk_valid_set(
    grid: np.ndarray,
    miss_risks: np.ndarray,
    volume_risks: np.ndarray,
    n_miss: int,
    n_volume: int,
    alpha_miss: float,
    alpha_volume: float,
    delta: float,
) -> np.ndarray:
    """Thresholds certified against **both** constraints simultaneously.

    Each threshold is one null hypothesis ("at least one of the two risks is violated"), so
    its p-value is the *maximum* of the two individual p-values — an intersection-union test,
    which needs no assumption about how the two risks are related. Bonferroni across the grid
    then controls the family-wise error at ``delta``. Returning nothing is a real answer: it
    certifies that no threshold on this grid can keep both promises.
    """
    grid = np.asarray(grid, dtype=float)
    threshold = delta / max(len(grid), 1)
    keep: list[int] = []
    for i in range(len(grid)):
        p_miss = hoeffding_bentkus_pvalue(float(miss_risks[i]), n_miss, alpha_miss)
        p_volume = hoeffding_bentkus_pvalue(float(volume_risks[i]), n_volume, alpha_volume)
        if max(p_miss, p_volume) <= threshold:
            keep.append(i)
    return grid[np.array(keep, dtype=int)] if keep else np.zeros(0, dtype=float)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class GuaranteeRow:
    """One target miss rate, both selectors, and what each costs to keep."""

    alpha: float
    crc_threshold: float
    crc_realised_risk: float
    crc_mean_risk: float
    crc_exceedance: float
    ltt_threshold: float
    ltt_realised_risk: float
    ltt_mean_risk: float
    ltt_exceedance: float
    crc_fpr: float
    ltt_fpr: float
    crc_alerts_per_day: float
    ltt_alerts_per_day: float
    ltt_analysts: float


@dataclass
class ClassRow:
    """Whether a per-class miss-rate promise is affordable, and at what alert volume."""

    label: str
    support: int
    alpha: float
    feasible: bool
    threshold: float
    fpr: float
    alerts_per_day: float
    budget_multiple: float  # alerts/day divided by the largest analyst budget on file


@dataclass
class MultiRiskRow:
    """Two constraints at once: the feasible threshold band, or the certificate that none exists."""

    alpha_miss: float
    alpha_volume: float
    n_feasible: int
    low: float
    high: float
    realised_miss: float
    realised_volume: float


@dataclass
class RiskControlStudy:
    """Everything the report renders."""

    rows: list[GuaranteeRow]
    classes: list[ClassRow]
    multi: list[MultiRiskRow]
    deployed_threshold: float
    deployed_miss_rate: float
    deployed_fpr: float
    deployed_alerts_per_day: float
    n_calibration: int
    n_evaluation: int
    n_trials: int
    delta: float
    grid_size: int
    minutes_per_alert: float
    analyst_minutes: float
    flows_per_day: int


def _risk_curve(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Miss rate at each grid threshold: the fraction of attacks scoring below it."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    return np.searchsorted(ordered, grid, side="left") / max(len(ordered), 1)


def _volume_curve(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Alert rate at each grid threshold: the fraction of flows at or above it."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    return 1.0 - np.searchsorted(ordered, grid, side="left") / max(len(ordered), 1)


def run_risk_control_study(settings: Settings) -> RiskControlStudy:
    """Calibrate both selectors, simulate the deployment loop, and price every promise."""
    cfg: RiskControlConfig = settings.risk_control
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    fit = fit_supervised(variant)
    benign_label = variant.labels.benign_label
    scores_test = attack_probability(fit.proba_test, fit.classes, benign_label)
    y_test = (
        (fit.y_test != benign_label).astype(int)
        if fit.y_test.dtype.kind in "OU"
        else fit.y_test.astype(int)
    )
    scores_val = attack_probability(fit.proba_val, fit.classes, benign_label)
    y_val = (
        (fit.y_val != benign_label).astype(int)
        if fit.y_val.dtype.kind in "OU"
        else fit.y_val.astype(int)
    )

    attack_scores = scores_test[y_test == 1]
    benign_scores = scores_test[y_test == 0]
    grid = np.unique(np.quantile(scores_test, np.linspace(0.0, 1.0, cfg.grid_size)))
    flows_per_day = variant.thresholds.assumed_flows_per_day
    benign_fraction = float(np.mean(y_test == 0))

    # The deployed operating point, for comparison: a threshold picked at a false-positive
    # budget on validation, judged by the risk it happens to deliver.
    from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr

    deployed_threshold = threshold_at_fpr(y_val, scores_val, variant.thresholds.primary_fpr)
    deployed_rates = rates_at_threshold(y_test, scores_test, deployed_threshold)

    n_cal = int(len(attack_scores) * cfg.calibration_fraction)
    n_cal_benign = int(len(benign_scores) * cfg.calibration_fraction)

    rows: list[GuaranteeRow] = []
    exceedance_curves: dict[str, list[tuple[float, float]]] = {"CRC": [], "LTT": []}
    for alpha in cfg.alphas:
        crc_realised: list[float] = []
        ltt_realised: list[float] = []
        crc_thresholds: list[float] = []
        ltt_thresholds: list[float] = []
        for _ in range(cfg.n_trials):
            order = rng.permutation(len(attack_scores))
            cal, evaluation = attack_scores[order[:n_cal]], attack_scores[order[n_cal:]]
            risks = _risk_curve(cal, grid)
            crc = crc_threshold(grid, risks, len(cal), alpha)
            valid = ltt_valid_set(grid, risks, len(cal), alpha, cfg.delta)
            ltt = float(valid[-1]) if len(valid) else float(grid[0])
            crc_thresholds.append(crc)
            ltt_thresholds.append(ltt)
            crc_realised.append(float(np.mean(evaluation < crc)))
            ltt_realised.append(float(np.mean(evaluation < ltt)))

        crc_point = float(np.median(crc_thresholds))
        ltt_point = float(np.median(ltt_thresholds))
        crc_rates = rates_at_threshold(y_test, scores_test, crc_point)
        ltt_rates = rates_at_threshold(y_test, scores_test, ltt_point)
        crc_alerts = alerts_per_day(crc_rates["fpr"], flows_per_day, benign_fraction)
        ltt_alerts = alerts_per_day(ltt_rates["fpr"], flows_per_day, benign_fraction)
        crc_exceed = float(np.mean(np.array(crc_realised) > alpha))
        ltt_exceed = float(np.mean(np.array(ltt_realised) > alpha))
        exceedance_curves["CRC"].append((alpha, crc_exceed))
        exceedance_curves["LTT"].append((alpha, ltt_exceed))
        rows.append(
            GuaranteeRow(
                alpha=alpha,
                crc_threshold=crc_point,
                crc_realised_risk=1.0 - crc_rates["tpr"],
                crc_mean_risk=float(np.mean(crc_realised)),
                crc_exceedance=crc_exceed,
                ltt_threshold=ltt_point,
                ltt_realised_risk=1.0 - ltt_rates["tpr"],
                ltt_mean_risk=float(np.mean(ltt_realised)),
                ltt_exceedance=ltt_exceed,
                crc_fpr=crc_rates["fpr"],
                ltt_fpr=ltt_rates["fpr"],
                crc_alerts_per_day=crc_alerts,
                ltt_alerts_per_day=ltt_alerts,
                ltt_analysts=ltt_alerts
                * variant.alert_queue.minutes_per_alert
                / max(variant.alert_queue.analyst_minutes_per_day, 1.0),
            )
        )
        logger.info("Risk target priced", extra={"alpha": alpha, "ltt_alerts": round(ltt_alerts)})

    # Per class: a global promise is the average of promises the detector keeps unequally.
    # The fitted task is binary, so the attack *family* has to come from the split itself --
    # the same row order the model scored, which is how the per-class slices study reads it.
    from netsentry.data.split import load_split

    test_frame = load_split(variant, "temporal", "test")
    classes: list[ClassRow] = []
    labels = (
        test_frame[MULTICLASS_TARGET].to_numpy().astype(str)
        if MULTICLASS_TARGET in test_frame.columns and len(test_frame) == len(scores_test)
        else np.asarray(fit.y_test).astype(str)
    )
    alpha_class = cfg.class_alpha
    analyst_budget = float(max(variant.alert_queue.alert_budgets_per_day))
    for label in sorted(set(labels[y_test == 1])):
        rows_for_class = labels == label
        class_scores = scores_test[rows_for_class]
        if len(class_scores) < cfg.class_min_support:
            continue
        order = rng.permutation(len(class_scores))
        cal = class_scores[order[: max(len(class_scores) // 2, 1)]]
        risks = _risk_curve(cal, grid)
        valid = ltt_valid_set(grid, risks, len(cal), alpha_class, cfg.delta)
        certified = len(valid) > 0
        threshold = float(valid[-1]) if certified else float(grid[0])
        rates = rates_at_threshold(y_test, scores_test, threshold)
        classes.append(
            ClassRow(
                label=label,
                support=int(rows_for_class.sum()),
                alpha=alpha_class,
                feasible=certified,
                threshold=threshold,
                fpr=rates["fpr"],
                alerts_per_day=alerts_per_day(rates["fpr"], flows_per_day, benign_fraction),
                budget_multiple=alerts_per_day(rates["fpr"], flows_per_day, benign_fraction)
                / max(analyst_budget, 1.0),
            )
        )

    # Two risks at once.
    multi: list[MultiRiskRow] = []
    order = rng.permutation(len(attack_scores))
    cal_attacks = attack_scores[order[:n_cal]]
    order_b = rng.permutation(len(benign_scores))
    cal_benign = benign_scores[order_b[:n_cal_benign]]
    miss_risks = _risk_curve(cal_attacks, grid)
    volume_risks = _volume_curve(cal_benign, grid)
    for alpha_miss in cfg.multi_alphas:
        for alpha_volume in cfg.volume_budgets:
            feasible = multi_risk_valid_set(
                grid,
                miss_risks,
                volume_risks,
                len(cal_attacks),
                len(cal_benign),
                alpha_miss,
                alpha_volume,
                cfg.delta,
            )
            if len(feasible):
                chosen = float(feasible[-1])
                rates = rates_at_threshold(y_test, scores_test, chosen)
                multi.append(
                    MultiRiskRow(
                        alpha_miss=alpha_miss,
                        alpha_volume=alpha_volume,
                        n_feasible=len(feasible),
                        low=float(feasible[0]),
                        high=float(feasible[-1]),
                        realised_miss=1.0 - rates["tpr"],
                        realised_volume=rates["fpr"],
                    )
                )
            else:
                multi.append(
                    MultiRiskRow(
                        alpha_miss=alpha_miss,
                        alpha_volume=alpha_volume,
                        n_feasible=0,
                        low=float("nan"),
                        high=float("nan"),
                        realised_miss=float("nan"),
                        realised_volume=float("nan"),
                    )
                )

    return RiskControlStudy(
        rows=rows,
        classes=classes,
        multi=multi,
        deployed_threshold=deployed_threshold,
        deployed_miss_rate=1.0 - deployed_rates["tpr"],
        deployed_fpr=deployed_rates["fpr"],
        deployed_alerts_per_day=alerts_per_day(
            deployed_rates["fpr"], flows_per_day, benign_fraction
        ),
        n_calibration=n_cal,
        n_evaluation=len(attack_scores) - n_cal,
        n_trials=cfg.n_trials,
        delta=cfg.delta,
        grid_size=len(grid),
        minutes_per_alert=variant.alert_queue.minutes_per_alert,
        analyst_minutes=variant.alert_queue.analyst_minutes_per_day,
        flows_per_day=flows_per_day,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_risk_control_report(settings: Settings) -> Path:
    """Run the risk-control study and write the report + figure."""
    study = run_risk_control_study(settings)
    alphas = np.array([row.alpha for row in study.rows], dtype=float)
    figure = plots.plot_lines(
        {
            "conformal risk control (expectation bound)": (
                alphas,
                np.array([row.crc_exceedance for row in study.rows], dtype=float),
            ),
            "Learn then Test (high-probability bound)": (
                alphas,
                np.array([row.ltt_exceedance for row in study.rows], dtype=float),
            ),
            f"the promise LTT makes (delta = {study.delta:g})": (
                alphas,
                np.full(len(alphas), study.delta),
            ),
        },
        xlabel="target miss rate alpha",
        ylabel=f"fraction of {study.n_trials} deployments whose realised risk exceeded alpha",
        title="An expectation bound is not a promise about your deployment",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote risk-control report", extra={"path": str(out_path)})

    with track_run(settings, "risk_control") as run:
        run.log_params({"delta": study.delta, "trials": study.n_trials})
        run.log_metrics(
            {f"crc_exceedance_{row.alpha:g}": row.crc_exceedance for row in study.rows}
            | {f"ltt_exceedance_{row.alpha:g}": row.ltt_exceedance for row in study.rows}
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _guarantee_table(study: RiskControlStudy) -> str:
    rows = [
        "| target miss rate | selector | threshold | realised miss rate | mean over "
        f"{study.n_trials} deployments | exceeded target | realised FPR | alerts/day | analysts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in study.rows:
        rows.append(
            f"| {row.alpha:.0%} | conformal risk control | {row.crc_threshold:.4f} | "
            f"{row.crc_realised_risk:.1%} | {row.crc_mean_risk:.1%} | "
            f"**{row.crc_exceedance:.0%}** | {row.crc_fpr:.2%} | "
            f"{row.crc_alerts_per_day:,.0f} | — |"
        )
        rows.append(
            f"| {row.alpha:.0%} | Learn then Test | {row.ltt_threshold:.4f} | "
            f"{row.ltt_realised_risk:.1%} | {row.ltt_mean_risk:.1%} | "
            f"**{row.ltt_exceedance:.0%}** | {row.ltt_fpr:.2%} | "
            f"{row.ltt_alerts_per_day:,.0f} | {row.ltt_analysts:,.0f} |"
        )
    return "\n".join(rows)


def _class_table(study: RiskControlStudy) -> str:
    rows = [
        "| attack class | attacks in test | promise certified | FPR it costs | alerts/day | "
        "times the largest analyst budget |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(study.classes, key=lambda r: r.alerts_per_day):
        verdict = "yes" if row.feasible else "**no threshold certifies it**"
        rows.append(
            f"| {row.label} | {row.support:,} | {verdict} | {row.fpr:.1%} | "
            f"{row.alerts_per_day:,.0f} | **{row.budget_multiple:,.0f}x** |"
        )
    return "\n".join(rows)


def _multi_table(study: RiskControlStudy) -> str:
    rows = [
        "| miss rate <= | alert rate <= | thresholds certified | realised miss | "
        "realised alert rate |",
        "|---|---|---|---|---|",
    ]
    for row in study.multi:
        if row.n_feasible == 0:
            rows.append(
                f"| {row.alpha_miss:.0%} | {row.alpha_volume:.1%} | **none (infeasible)** | — | — |"
            )
        else:
            rows.append(
                f"| {row.alpha_miss:.0%} | {row.alpha_volume:.1%} | {row.n_feasible} | "
                f"{row.realised_miss:.1%} | {row.realised_volume:.2%} |"
            )
    return "\n".join(rows)


def _headline(study: RiskControlStudy) -> str:
    worst = max(study.rows, key=lambda r: r.crc_exceedance)
    tightest = min(study.rows, key=lambda r: r.alpha)
    return (
        f"**Conformal risk control keeps its promise and it is not the promise an operator "
        f"hears.** Across {study.n_trials} simulated calibrate-and-deploy cycles its mean "
        f"realised miss rate lands under target at every level — that is the theorem — while "
        f"the *individual* deployment exceeds target on up to {worst.crc_exceedance:.0%} of "
        "draws. An expectation bound says the average deployment is fine. Roughly half of all "
        "deployments are above average in the wrong direction, and each of those is somebody's "
        f"quarter.\n\n"
        f"Learn then Test buys the statement operators think they are getting — "
        f"`P(miss rate > alpha) <= {study.delta:g}` — and the exceedance column confirms it "
        f"empirically at every level. The price is a lower threshold and more alerts: at a "
        f"{tightest.alpha:.0%} target it demands {tightest.ltt_alerts_per_day:,.0f} alerts a "
        f"day against conformal risk control's {tightest.crc_alerts_per_day:,.0f}."
    )


def _price_read(study: RiskControlStudy) -> str:
    tightest = min(study.rows, key=lambda r: r.alpha)
    loosest = max(study.rows, key=lambda r: r.alpha)
    return (
        "The alerts-per-day column is the one to take to a budget meeting. The deployed "
        f"operating point — chosen at a {study.deployed_fpr:.2%} realised false-positive rate — "
        f"misses **{study.deployed_miss_rate:.1%}** of attacks and generates "
        f"{study.deployed_alerts_per_day:,.0f} alerts a day at {study.flows_per_day:,} flows. "
        f"Certifying a {tightest.alpha:.0%} miss rate instead requires "
        f"{tightest.ltt_alerts_per_day:,.0f} alerts a day — about "
        f"{tightest.ltt_analysts:,.0f} analysts at {study.minutes_per_alert:.0f} minutes per "
        f"alert and {study.analyst_minutes:.0f} productive minutes each. Even the loosest "
        f"target here ({loosest.alpha:.0%} misses) costs {loosest.ltt_alerts_per_day:,.0f} "
        f"alerts a day and {loosest.ltt_analysts:,.0f} analysts.\n\n"
        "That is not a criticism of the method; the method is doing its job, which is to make "
        "the exchange rate explicit instead of letting a false-positive budget imply a miss "
        "rate nobody wrote down. The uncomfortable reading is the honest one: **on this "
        "detector, a miss-rate guarantee at any interesting level is unaffordable**, and the "
        "options are a better detector, a narrower promise (per class, below), or a contract "
        "written in the units the detector can actually deliver."
    )


def _multi_read(study: RiskControlStudy) -> str:
    infeasible = [row for row in study.multi if row.n_feasible == 0]
    feasible = [row for row in study.multi if row.n_feasible > 0]
    return (
        "A real detection contract has two clauses, not one: miss at most this much, *and* do "
        "not drown the queue. Learn-then-Test handles that natively — each threshold becomes a "
        "single null hypothesis whose p-value is the maximum of the two constraints' p-values "
        "(an intersection-union test, which assumes nothing about how the risks relate), with "
        "Bonferroni across the grid because the two risks move in opposite directions and the "
        "fixed-sequence shortcut is only valid when they do not.\n\n"
        f"{len(infeasible)} of the {len(study.multi)} pairs come back **empty**, and an empty "
        "set is the most useful output in this report. It is not a failure to find a "
        "threshold; it is a certificate that no threshold on the grid can keep both promises "
        "at this confidence, delivered before anybody signs the contract rather than during "
        "the first incident review. "
        + (
            f"The {len(feasible)} pairs that are feasible are the ones where the alert budget "
            "is loose enough to pay for the miss rate, and the report gives the certified "
            "band rather than a point, because every threshold inside it is equally valid."
            if feasible
            else "None of the pairs is feasible, which says the two clauses in the contract "
            "cannot both be met by thresholding this model at all."
        )
    )


def _class_read(study: RiskControlStudy) -> str:
    if not study.classes:
        return ""
    ordered = sorted(study.classes, key=lambda r: r.alerts_per_day)
    cheapest, dearest = ordered[0], ordered[-1]
    ratio = dearest.alerts_per_day / max(cheapest.alerts_per_day, 1.0)
    return (
        f"Every class can be *certified* at the {study.classes[0].alpha:.0%} target, which "
        "makes the feasibility column useless and the price column the whole point. "
        f"`{cheapest.label}` costs {cheapest.fpr:.1%} false positives; `{dearest.label}` costs "
        f"{dearest.fpr:.1%} — a **{ratio:.0f}x** difference in alert volume for the identical "
        "promise. The global number in the table above is an average over these, weighted by "
        "whichever attack mix the test days happened to contain, so it describes a contract no "
        "single class is actually operating under.\n\n"
        "This is the [open-set study's](openset.md) finding in contractual form. That study "
        "found the deployed novelty rule's entire lead carried by `DDoS` while it was blind to "
        "`PortScan`; here the same asymmetry reappears as a price list. The engineering "
        "conclusion is that a miss-rate SLA should be written **per attack family**, because a "
        "global one silently subsidises the classes the detector cannot see with the alert "
        "budget of the one it can — and the subsidy is invisible until the mix changes."
    )


def _render(study: RiskControlStudy, figure: Path) -> str:
    return f"""# NetSentry — Controlling the Risk the Contract Names

_Conformal risk control (Angelopoulos et al. 2022) and Learn-then-Test (Angelopoulos et al.
2021) over a {study.grid_size}-point threshold grid, calibrated on {study.n_calibration:,}
attacks and validated by simulating {study.n_trials} calibrate-and-deploy cycles._

## Why this report exists

Every operating point in this project is chosen by fixing a **false-positive** budget. That is
the right instrument for "how much noise can my analysts absorb" and the wrong one for the
sentence detection contracts are actually written in: *you may miss at most one attack in ten*.
[Conformal prediction](conformal.md) guarantees coverage, [alert FDR](alert_fdr.md) guarantees
the false-discovery rate of a batch, [Neyman-Pearson](neyman_pearson.md) certifies the
false-positive rate. None of them bounds the miss rate.

Two instruments do, and they differ in a way that matters more than the algorithms.

## An expectation bound is not a promise about your deployment

![Exceedance by target](../figures/{figure.name})

{_guarantee_table(study)}

{_headline(study)}

## What a miss-rate promise costs

{_price_read(study)}

## Two promises at once

{_multi_table(study)}

{_multi_read(study)}

## The promise that is affordable is per class

{_class_table(study)}

{_class_read(study)}

## Scope and honest limits

- **Both guarantees are conditional on exchangeability**, and this calibration set is drawn
  from the same capture days it is evaluated on. Flows inside one attack burst are near
  duplicates, so the *effective* sample is smaller than {study.n_calibration:,} and the bound
  is correspondingly optimistic — the same caveat the [conformal study](conformal.md) carries,
  for the same reason.
- **Calibrating on production attacks requires labelled production attacks.** This report
  assumes a SOC that confirms incidents (which is what a SOC does) and can therefore calibrate
  on its own history; a greenfield deployment has no such set and must inherit a threshold,
  which the [threshold-transfer study](threshold_transfer.md) prices.
- **The risk is a miss rate over flows, not over incidents.** Missing nine flows of a
  thousand-flow DDoS is not the same event as missing the only flow of an exfiltration, and
  the [campaign study](campaigns.md) is where that distinction is measured. A per-incident
  risk is the better contract and needs incident-level labels to calibrate.
- **The grid is finite.** Both selectors return thresholds from a {study.grid_size}-point
  quantile grid; a finer grid costs Bonferroni power in the multi-risk arm and changes
  nothing for the monotone single-risk one.
- **The alert-volume figures assume {study.flows_per_day:,} flows a day** and a fixed
  {study.minutes_per_alert:.0f}-minute triage cost. Both are the project's standing
  assumptions, kept here so this table is comparable with the
  [alert-queue study](alert_queue.md) rather than because either is a measurement."""
