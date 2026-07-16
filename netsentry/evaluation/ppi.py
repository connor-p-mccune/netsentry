"""Prediction-powered inference: a tight, valid estimate of prevalence from few labels.

Every evaluation so far assumes a fully labelled test set. A SOC never has one. It has a
firehose of flows the model has scored and a tiny hand-labelled audit sample — and it
still needs a defensible answer to "what fraction of today's traffic is actually
malicious?", with an honest confidence interval, not a point guess. There are two obvious
ways to answer, and both are wrong on their own:

- **Classical** (label the audit sample, ignore the model): unbiased and valid, but the
  interval is only as tight as a few hundred labels allow — wide.
- **Naive imputation** (let the model label everything, average its scores): tight,
  because it uses every flow — but biased by the model's own error, and its interval
  treats model outputs as ground truth, so it *understates* uncertainty and its coverage
  guarantee is unearned.

**Prediction-powered inference** (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic,
*Science* 2023) keeps the model's tightness *and* the classical validity. It starts from
the model's estimate over all the unlabelled flows and then subtracts the model's measured
bias on the labelled audit — the **rectifier** ``mean(f - y)`` over the labels:

    theta_PP = mean(f over unlabelled)  -  mean(f - y over labelled)

The rectifier is what makes it honest: whatever the model gets wrong, on average, is
corrected using real labels, so ``theta_PP`` is unbiased for the true prevalence *whether
or not the model is calibrated*. Its variance is ``var(f)/N + var(f - y)/n``; because a
useful model makes the residual ``f - y`` lower-variance than the label ``y`` itself, the
interval is **narrower than classical at the same coverage** — the paper's central result,
here priced as "labels saved". This study sweeps the audit budget, and at each budget
measures every interval's half-width and its *empirical coverage* of the true test
prevalence over many random audit draws, so validity is demonstrated, not asserted.

Runs on the exchangeable **stratified/binary** split: PPI's guarantee assumes the labelled
audit is a random sample of the scored population, exactly the exchangeability the temporal
split is designed to break — so this is the honest home for a validity claim, the same
reason the membership and data-valuation studies live there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

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

REPORT_NAME = "ppi.md"
FIGURE_NAME = "ppi.png"


@dataclass
class Interval:
    """A point estimate and its two-sided confidence interval."""

    point: float
    lo: float
    hi: float

    @property
    def halfwidth(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def covers(self, theta: float) -> bool:
        return self.lo <= theta <= self.hi


def _z(alpha: float) -> float:
    """Two-sided normal critical value for a 1 - ``alpha`` interval."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def _sample_var(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.var(v, ddof=1)) if len(v) > 1 else 0.0


def classical_mean_ci(y_labeled: np.ndarray, alpha: float) -> Interval:
    """Label-only estimate of a mean: the sample mean and its normal CI.

    Valid by construction (it is just the labelled sample mean), but its width is set
    entirely by the audit budget ``n`` — the interval PPI has to beat.
    """
    y = np.asarray(y_labeled, dtype=float)
    n = len(y)
    if n == 0:
        return Interval(float("nan"), -float("inf"), float("inf"))
    mean = float(y.mean())
    half = _z(alpha) * float(np.sqrt(_sample_var(y) / n))
    return Interval(mean, mean - half, mean + half)


def naive_mean_ci(f_unlabeled: np.ndarray, alpha: float) -> Interval:
    """Model-only ("imputation") estimate: average the model scores and pretend they are labels.

    Tight, because it uses every scored flow, but biased by the model's error and blind to
    label uncertainty — its coverage is an artefact of calibration, not a guarantee. Shown
    as the tempting-but-invalid baseline PPI corrects.
    """
    f = np.asarray(f_unlabeled, dtype=float)
    n = len(f)
    if n == 0:
        return Interval(float("nan"), -float("inf"), float("inf"))
    mean = float(f.mean())
    half = _z(alpha) * float(np.sqrt(_sample_var(f) / n))
    return Interval(mean, mean - half, mean + half)


def ppi_mean_ci(
    y_labeled: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    alpha: float,
) -> Interval:
    """Prediction-powered CI for a mean (Angelopoulos et al. 2023).

    ``theta_PP = mean(f_unlabeled) - mean(f_labeled - y_labeled)``: the model's estimate
    over the unlabelled flows, de-biased by the rectifier measured on the labelled audit.
    The variance ``var(f)/N + var(f - y)/n`` combines the (large-N, tiny) model-mean term
    with the (small-n) rectifier term; because a useful model makes ``f - y`` lower-variance
    than ``y``, the result is tighter than classical while staying unbiased.
    """
    y = np.asarray(y_labeled, dtype=float)
    f_lab = np.asarray(f_labeled, dtype=float)
    f_unl = np.asarray(f_unlabeled, dtype=float)
    n, big_n = len(y), len(f_unl)
    if n == 0 or big_n == 0:
        return Interval(float("nan"), -float("inf"), float("inf"))
    rectifier = f_lab - y
    point = float(f_unl.mean()) - float(rectifier.mean())
    var = _sample_var(f_unl) / big_n + _sample_var(rectifier) / n
    half = _z(alpha) * float(np.sqrt(max(var, 0.0)))
    return Interval(point, point - half, point + half)


def effective_sample_size_gain(y_labeled: np.ndarray, f_labeled: np.ndarray) -> float:
    """How many more labels classical would need to match PPI's width: ``var(y) / var(f - y)``.

    1.0 means the model adds nothing (an uninformative or constant score); larger means the
    model's residual is tighter than the raw label, so PPI is that many times more
    label-efficient. Infinite when the model is exactly right on the audit.
    """
    y = np.asarray(y_labeled, dtype=float)
    rect = np.asarray(f_labeled, dtype=float) - y
    var_rect = _sample_var(rect)
    var_y = _sample_var(y)
    if var_rect <= 0.0:
        return float("inf")
    return float(var_y / var_rect)


@dataclass
class BudgetRow:
    """Interval widths and empirical coverage at one audit-label budget."""

    n_labels: int
    hw_classical: float
    hw_ppi: float
    hw_naive: float
    cover_classical: float
    cover_ppi: float
    cover_naive: float
    ess_gain: float  # median var(y)/var(f-y) over the trials


@dataclass
class PPIStudy:
    """The full prediction-powered-inference study over the audit-budget sweep."""

    alpha: float
    n_trials: int
    pool_size: int
    true_prevalence: float
    model_prevalence: float  # mean(f) — the naive point estimate
    naive: Interval
    rows: list[BudgetRow]


def run_ppi(settings: Settings) -> PPIStudy:
    """Fit the honest exchangeable model, then sweep the audit budget for the three estimators."""
    cfg = settings.ppi
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test").reset_index(drop=True)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.model.classes_)

    x_test = np.asarray(pipeline.transform(test))
    f_test = positive_scores(model.predict_proba(x_test), classes)  # raw model attack prob.
    y_test = test[BINARY_TARGET].to_numpy().astype(float)

    pool = len(y_test)
    theta_star = float(y_test.mean())
    naive = naive_mean_ci(f_test, cfg.alpha)  # deterministic: it ignores the labels

    rng = np.random.default_rng(variant.seed)
    rows: list[BudgetRow] = []
    for n in cfg.label_budgets:
        if n >= pool:
            continue
        hw_c, hw_p, ess = [], [], []
        cover_c = cover_p = 0
        for _ in range(cfg.n_trials):
            idx = rng.choice(pool, size=n, replace=False)
            y_lab, f_lab = y_test[idx], f_test[idx]
            ci_c = classical_mean_ci(y_lab, cfg.alpha)
            ci_p = ppi_mean_ci(y_lab, f_lab, f_test, cfg.alpha)
            cover_c += int(ci_c.covers(theta_star))
            cover_p += int(ci_p.covers(theta_star))
            hw_c.append(ci_c.halfwidth)
            hw_p.append(ci_p.halfwidth)
            ess.append(effective_sample_size_gain(y_lab, f_lab))
        finite = [e for e in ess if np.isfinite(e)]
        rows.append(
            BudgetRow(
                n_labels=n,
                hw_classical=float(np.mean(hw_c)),
                hw_ppi=float(np.mean(hw_p)),
                hw_naive=naive.halfwidth,
                cover_classical=cover_c / cfg.n_trials,
                cover_ppi=cover_p / cfg.n_trials,
                cover_naive=float(naive.covers(theta_star)),
                ess_gain=float(np.median(finite)) if finite else float("inf"),
            )
        )
        logger.info(
            "PPI budget",
            extra={
                "n": n,
                "hw_classical": round(rows[-1].hw_classical, 4),
                "hw_ppi": round(rows[-1].hw_ppi, 4),
                "cover_ppi": round(rows[-1].cover_ppi, 3),
            },
        )
    return PPIStudy(
        alpha=cfg.alpha,
        n_trials=cfg.n_trials,
        pool_size=pool,
        true_prevalence=theta_star,
        model_prevalence=float(f_test.mean()),
        naive=naive,
        rows=rows,
    )


def run_ppi_report(settings: Settings) -> Path:
    """Run the prediction-powered-inference study and write the report + figure."""
    study = run_ppi(settings)

    budgets = np.array([r.n_labels for r in study.rows], dtype=float)
    series = {
        "classical (labels only)": (budgets, np.array([r.hw_classical for r in study.rows])),
        "prediction-powered": (budgets, np.array([r.hw_ppi for r in study.rows])),
        "naive imputation (invalid)": (budgets, np.array([r.hw_naive for r in study.rows])),
    }
    fig = plots.plot_lines(
        series,
        xlabel="audit labels (n)",
        ylabel=f"CI half-width ({1 - study.alpha:.0%} interval)",
        title="Prediction-powered inference: model-tight, label-valid prevalence",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote PPI report", extra={"path": str(out_path)})

    with track_run(settings, "ppi") as run:
        run.log_metrics(
            {
                "true_prevalence": study.true_prevalence,
                "model_prevalence": study.model_prevalence,
                "cover_ppi_max_budget": study.rows[-1].cover_ppi if study.rows else 0.0,
                "width_ratio_max_budget": (
                    study.rows[-1].hw_ppi / study.rows[-1].hw_classical
                    if study.rows and study.rows[-1].hw_classical
                    else 0.0
                ),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _table(study: PPIStudy) -> str:
    rows = [
        "| audit labels | classical HW | PPI HW | naive HW | classical cov. | PPI cov. "
        "| labels saved (x) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        ess = "∞" if not np.isfinite(r.ess_gain) else f"{r.ess_gain:.1f}"
        rows.append(
            f"| {r.n_labels:,} | {r.hw_classical:.4f} | {r.hw_ppi:.4f} | {r.hw_naive:.4f} "
            f"| {r.cover_classical:.0%} | {r.cover_ppi:.0%} | {ess} |"
        )
    return "\n".join(rows)


def _read(study: PPIStudy) -> str:
    if not study.rows:
        return "No budget below the pool size was configured."
    last = study.rows[-1]
    ratio = last.hw_ppi / last.hw_classical if last.hw_classical else 1.0
    tighter = ratio < 0.97
    nominal = 1.0 - study.alpha
    valid = abs(last.cover_ppi - nominal) <= 0.05
    bias = study.model_prevalence - study.true_prevalence
    width_clause = (
        f"At {last.n_labels:,} audit labels the prediction-powered interval is "
        f"**{1 - ratio:.0%} narrower** than the classical one ({last.hw_ppi:.4f} vs "
        f"{last.hw_classical:.4f}) — the model's residual carries information the raw label "
        f"does not, worth about **{last.ess_gain:.1f}x the labels** at this budget. "
        if tighter
        else (
            f"At {last.n_labels:,} labels the PPI interval ({last.hw_ppi:.4f}) is close to "
            f"classical ({last.hw_classical:.4f}); on this stand-in the model's residual is "
            "barely tighter than the label itself, so there is little to gain — reported as "
            "it fell. "
        )
    )
    cover_clause = (
        f"Both classical and PPI cover the true prevalence at roughly the nominal "
        f"{nominal:.0%} ({last.cover_ppi:.0%} for PPI) — the tightening is free of validity cost. "
        if valid
        else (
            f"PPI's empirical coverage is {last.cover_ppi:.0%} against a {nominal:.0%} target on "
            "this small stand-in; the normal approximation loosens at tiny budgets, reported "
            "plainly. "
        )
    )
    naive_clause = (
        f"Naive imputation is the cautionary column. Its point estimate is the model's own "
        f"mean score, {study.model_prevalence:.3f} against a true prevalence of "
        f"{study.true_prevalence:.3f} (a {bias:+.3f} bias), and its interval — the same every "
        f"audit because it never looks at a label — is far too narrow at "
        f"{study.naive.halfwidth:.4f}, so it "
        + ("misses" if not study.naive.covers(study.true_prevalence) else "only happens to cover")
        + " the truth. It treats model outputs as ground truth; PPI treats them as a lead to be "
        "checked against real labels, which is exactly why only PPI's confidence is earned."
    )
    return width_clause + cover_clause + naive_clause


def _render(study: PPIStudy, fig: Path) -> str:
    return f"""# NetSentry — Prediction-Powered Inference (attack prevalence)

_Synthetic stand-in. Stratified/binary model; the {study.pool_size:,}-flow test set is the
scored population, a random audit of it is labelled, and every interval is a
{1 - study.alpha:.0%} CI. Widths and coverage are averaged over {study.n_trials:,} random
audit draws per budget. True test prevalence: **{study.true_prevalence:.3f}**._

## Why this report exists

A SOC scores every flow but labels almost none. It still needs a defensible estimate of how
much of today's traffic is malicious, with an honest interval. Labelling a small audit and
ignoring the model (classical) is valid but wide; letting the model label everything (naive
imputation) is tight but biased and blind to label uncertainty. Prediction-powered inference
(Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Science* 2023) keeps the model's tightness
and the classical validity by correcting the model's estimate with its **measured bias on the
labelled audit** — the rectifier ``mean(f - y)``. The estimate is unbiased whether or not the
model is calibrated, and tighter than classical because a useful model's residual is
lower-variance than the label itself.

## Interval width and coverage vs the audit budget

{_table(study)}

![PPI vs classical vs naive interval width](../figures/{fig.name})

{_read(study)}

## Scope

The estimand here is a **mean** (prevalence); PPI extends to quantiles, regression
coefficients and other convex M-estimation problems by the same rectifier construction, but
the mean is the SOC-relevant one and keeps the demonstration exact. The study runs on the
exchangeable stratified split on purpose: PPI's validity assumes the labelled audit is a
random sample of the scored population, so under the temporal split — where the audit day and
the scored day differ — the rectifier estimated on one distribution would not correct the bias
on another, and the guarantee would not hold. That is not a limitation of PPI so much as a
restatement of this project's thesis: the honest interval is only as honest as the
exchangeability behind it."""
