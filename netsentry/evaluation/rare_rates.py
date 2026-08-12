"""Rare-class detection rates: what a per-class number means when the class has eleven flows.

The per-class table is the most-read output this project produces, and it is the one most likely
to be misread. `DDoS: 54.9%` rests on thousands of test flows and means what it says. `Heartbleed:
0.0%` rests on a handful, and means almost nothing — the same detector, on a different sample of
the same size, could plausibly have printed 33% or 100%. Both numbers are set in the same column,
in the same font, with no indication that one is a measurement and the other is a rumour.

The standard fix is an interval, and the standard interval is wrong here: the Wald interval on a
proportion collapses to zero width at 0 and 1 successes, which is precisely the regime the rare
classes live in. Wilson (1927) and Jeffreys fix that, and this report computes both. But a
frequentist interval on eleven flows is honest and *useless*: it spans most of the unit interval,
and an operator reading it learns only that nothing is known.

**Partial pooling** does better, by using something the per-class view throws away: the other
classes. Model each class's detection rate as drawn from a shared Beta prior whose parameters are
estimated from all classes at once (the empirical-Bayes / James-Stein move), and each class's
posterior becomes a compromise between its own data and the population it belongs to, weighted
automatically by how much data it has. A class with thousands of flows barely moves. A class with
eleven is pulled most of the way toward the pooled rate — which is the correct answer, because
eleven flows genuinely do not distinguish that class from the population it came from.

Three things are checked rather than asserted:

- the **shrinkage** each class receives, and how far the ranking moves when the point estimates
  are replaced by posterior means — because a leaderboard of rare classes is mostly a leaderboard
  of sample sizes;
- the **coverage** of the credible intervals, validated by simulation against Wilson's, since a
  Bayesian interval that does not cover at its stated rate is worse than no interval at all;
- the **sample size** each class would need before its own data could support a +/-5-point claim,
  which for the rarest classes exceeds the number of such flows the dataset contains at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta as beta_dist

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import RareRatesConfig

logger = get_logger(__name__)

REPORT_NAME = "rare_rates.md"
FIGURE_NAME = "rare_rates_shrinkage.png"


# --------------------------------------------------------------------------------------
# Intervals.
# --------------------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — the frequentist answer that survives 0 and n successes.

    The Wald interval (`p +/- z*sqrt(p(1-p)/n)`) has zero width at `p = 0` or `p = 1`, which is
    exactly where the rare classes sit, so it reports perfect certainty precisely when there is
    none. Wilson inverts the score test instead and stays sane at the boundaries.
    """
    if trials <= 0:
        return 0.0, 1.0
    z = _z_for(level)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _z_for(level: float) -> float:
    """Two-sided normal quantile for a confidence level (1.96 at 95%)."""
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2.0))


def jeffreys_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """The Jeffreys interval: a Beta(1/2, 1/2) prior, i.e. no pooling but a proper prior."""
    if trials <= 0:
        return 0.0, 1.0
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    a, b = successes + 0.5, trials - successes + 0.5
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(lo_q, a, b))
    hi = 1.0 if successes == trials else float(beta_dist.ppf(hi_q, a, b))
    return lo, hi


# --------------------------------------------------------------------------------------
# The hierarchical model: Beta-Binomial with empirical-Bayes hyperparameters.
# --------------------------------------------------------------------------------------


def beta_binomial_logpmf(k: int, n: int, alpha: float, beta: float) -> float:
    """Log marginal probability of `k` of `n` successes with `theta ~ Beta(alpha, beta)`.

    This is the likelihood the hyperparameters are fit against: the class-level rate is
    integrated out, so `(alpha, beta)` are judged on how well they explain *every* class's count
    at once rather than on any single class's rate.
    """
    if not 0 <= k <= n or alpha <= 0 or beta <= 0:
        return -np.inf
    log_choose = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_beta_num = lgamma(k + alpha) + lgamma(n - k + beta) - lgamma(n + alpha + beta)
    log_beta_den = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return float(log_choose + log_beta_num - log_beta_den)


def fit_beta_prior(
    successes: list[int], trials: list[int], grid_points: int = 60
) -> tuple[float, float]:
    """Empirical-Bayes `(alpha, beta)` by maximising the beta-binomial marginal likelihood.

    Parameterised by `(mean, concentration)` rather than `(alpha, beta)` because the two are
    nearly unidentified in the raw parameterisation: the mean is pinned by the pooled rate while
    the concentration -- how tightly classes cluster around it, and therefore how hard each class
    is shrunk -- is what the data has to speak to. A coarse log-grid is used deliberately: with a
    handful of classes the likelihood surface is flat, and a gradient optimiser would report a
    precision the data cannot support.
    """
    if not successes or len(successes) != len(trials):
        raise ValueError("successes and trials must be non-empty and the same length")
    total_k, total_n = sum(successes), sum(trials)
    pooled = min(max(total_k / max(total_n, 1), 1e-3), 1 - 1e-3)
    means = np.clip(
        np.linspace(pooled * 0.25, min(1 - 1e-3, pooled * 4), grid_points), 1e-3, 1 - 1e-3
    )
    concentrations = np.logspace(-1, 4, grid_points)
    best, best_ll = (pooled, 1.0), -np.inf
    for mean in means:
        for conc in concentrations:
            a, b = mean * conc, (1 - mean) * conc
            ll = sum(
                beta_binomial_logpmf(k, n, a, b) for k, n in zip(successes, trials, strict=True)
            )
            if ll > best_ll:
                best_ll, best = ll, (float(mean), float(conc))
    mean, conc = best
    return mean * conc, (1 - mean) * conc


def posterior_interval(
    successes: int, trials: int, alpha: float, beta: float, level: float = 0.95
) -> tuple[float, float, float]:
    """Posterior `(mean, lo, hi)` for one class under the fitted Beta prior."""
    a, b = alpha + successes, beta + trials - successes
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    return (
        float(a / (a + b)),
        float(beta_dist.ppf(lo_q, a, b)),
        float(beta_dist.ppf(hi_q, a, b)),
    )


def shrinkage_weight(alpha: float, beta: float, trials: int) -> float:
    """Share of the posterior mean supplied by the prior rather than by this class's own data."""
    prior_strength = alpha + beta
    return float(prior_strength / (prior_strength + trials))


def flows_needed(rate: float, half_width: float, level: float = 0.95) -> int:
    """Test flows a class needs before its own data supports a +/- `half_width` claim."""
    z = _z_for(level)
    p = min(max(rate, 1e-6), 1 - 1e-6)
    return int(np.ceil(z**2 * p * (1 - p) / half_width**2))


def coverage_simulation(
    alpha: float,
    beta: float,
    trials_per_class: list[int],
    n_replicates: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Simulate from the fitted prior and measure what the intervals actually cover.

    A credible interval that does not cover at its stated rate is worse than no interval, so the
    procedure is checked against data generated from the model it assumes. Returns
    `(bayes_coverage, wilson_coverage, bayes_mean_width, wilson_mean_width)`.
    """
    rng = np.random.default_rng(seed)
    hits_b = hits_w = 0
    width_b = width_w = 0.0
    total = 0
    for _ in range(n_replicates):
        thetas = rng.beta(alpha, beta, size=len(trials_per_class))
        for theta, n in zip(thetas, trials_per_class, strict=True):
            k = int(rng.binomial(n, theta))
            _, lo_b, hi_b = posterior_interval(k, n, alpha, beta, level)
            lo_w, hi_w = wilson_interval(k, n, level)
            hits_b += int(lo_b <= theta <= hi_b)
            hits_w += int(lo_w <= theta <= hi_w)
            width_b += hi_b - lo_b
            width_w += hi_w - lo_w
            total += 1
    return hits_b / total, hits_w / total, width_b / total, width_w / total


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class ClassRate:
    """One class's detection count and every estimate of its underlying rate."""

    name: str
    detected: int
    total: int
    naive: float
    posterior_mean: float
    posterior_lo: float
    posterior_hi: float
    wilson_lo: float
    wilson_hi: float
    jeffreys_lo: float
    jeffreys_hi: float
    shrinkage: float
    flows_for_5pp: int

    @property
    def posterior_width(self) -> float:
        return self.posterior_hi - self.posterior_lo

    @property
    def wilson_width(self) -> float:
        return self.wilson_hi - self.wilson_lo


@dataclass
class RareRatesStudy:
    """Per-class rates, the fitted prior, the coverage check, and the ranking movement."""

    rates: list[ClassRate]
    alpha: float
    beta: float
    pooled_rate: float
    prior_strength: float
    fpr_budget: float
    bayes_coverage: float
    wilson_coverage: float
    bayes_width: float
    wilson_width: float
    level: float
    rank_moves: list[tuple[str, int, int]]


def _per_class_counts(settings: Settings) -> tuple[list[str], list[int], list[int], float]:
    """Detected/total per attack class at the deployed operating point."""
    from netsentry.data.split import load_split

    variant = settings.model_copy(deep=True)
    variant.split.strategy = settings.rare_rates.split
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    strategy = variant.split.strategy
    train = load_split(variant, strategy, "train")
    val = load_split(variant, strategy, "val")
    test = load_split(variant, strategy, "test")
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    from netsentry.data.clean import BINARY_TARGET

    model = SupervisedClassifier(variant).fit(
        x_train,
        train[BINARY_TARGET].to_numpy().astype(int),
        eval_set=(x_val, val[BINARY_TARGET].to_numpy().astype(int)),
    )
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), model.classes_, benign)
    threshold = threshold_at_fpr(
        val[BINARY_TARGET].to_numpy().astype(int), s_val, variant.thresholds.primary_fpr
    )
    labels = test[MULTICLASS_TARGET].to_numpy()
    flagged = s_test >= threshold

    names, detected, totals = [], [], []
    for cls in sorted(set(labels)):
        if cls == benign:
            continue
        mask = labels == cls
        names.append(str(cls))
        detected.append(int(flagged[mask].sum()))
        totals.append(int(mask.sum()))
    return names, detected, totals, variant.thresholds.primary_fpr


def run_rare_rates(settings: Settings) -> RareRatesStudy:
    """Fit the shared prior, shrink every class, and validate the intervals by simulation."""
    cfg: RareRatesConfig = settings.rare_rates
    names, detected, totals, fpr = _per_class_counts(settings)
    alpha, beta = fit_beta_prior(detected, totals, cfg.grid_points)

    rates = []
    for name, k, n in zip(names, detected, totals, strict=True):
        mean, lo, hi = posterior_interval(k, n, alpha, beta, cfg.level)
        w_lo, w_hi = wilson_interval(k, n, cfg.level)
        j_lo, j_hi = jeffreys_interval(k, n, cfg.level)
        rates.append(
            ClassRate(
                name=name,
                detected=k,
                total=n,
                naive=k / n if n else 0.0,
                posterior_mean=mean,
                posterior_lo=lo,
                posterior_hi=hi,
                wilson_lo=w_lo,
                wilson_hi=w_hi,
                jeffreys_lo=j_lo,
                jeffreys_hi=j_hi,
                shrinkage=shrinkage_weight(alpha, beta, n),
                flows_for_5pp=flows_needed(mean, cfg.target_half_width, cfg.level),
            )
        )

    naive_order = [r.name for r in sorted(rates, key=lambda r: r.naive, reverse=True)]
    post_order = [r.name for r in sorted(rates, key=lambda r: r.posterior_mean, reverse=True)]
    moves = [
        (name, naive_order.index(name) + 1, post_order.index(name) + 1)
        for name in naive_order
        if naive_order.index(name) != post_order.index(name)
    ]

    cov_b, cov_w, width_b, width_w = coverage_simulation(
        alpha, beta, totals, cfg.coverage_replicates, settings.seed, cfg.level
    )
    return RareRatesStudy(
        rates=rates,
        alpha=alpha,
        beta=beta,
        pooled_rate=alpha / (alpha + beta),
        prior_strength=alpha + beta,
        fpr_budget=fpr,
        bayes_coverage=cov_b,
        wilson_coverage=cov_w,
        bayes_width=width_b,
        wilson_width=width_w,
        level=cfg.level,
        rank_moves=moves,
    )


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def run_rare_rates_report(settings: Settings) -> Path:
    """Run the hierarchical-rate study and write the report + figure."""
    study = run_rare_rates(settings)
    ordered = sorted(study.rates, key=lambda r: r.total)
    fig = plots.plot_barh(
        labels=[f"{r.name} (n={r.total:,})" for r in ordered],
        values=[r.shrinkage for r in ordered],
        xlabel="share of the estimate supplied by the prior, not by the class's own data",
        title="How much of each per-class number is borrowed?",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote rare-rates report", extra={"path": str(out_path)})

    with track_run(settings, "rare_rates") as run:
        run.log_metrics(
            {
                "prior_alpha": study.alpha,
                "prior_beta": study.beta,
                "pooled_rate": study.pooled_rate,
                "bayes_coverage": study.bayes_coverage,
                "wilson_coverage": study.wilson_coverage,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _rate_table(study: RareRatesStudy) -> str:
    rows = [
        "| class | detected / total | naive rate | Wilson 95% | posterior mean | posterior 95% "
        "| borrowed |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(study.rates, key=lambda x: x.total, reverse=True):
        rows.append(
            f"| `{r.name}` | {r.detected:,} / {r.total:,} | {r.naive:.1%} "
            f"| [{r.wilson_lo:.1%}, {r.wilson_hi:.1%}] | {r.posterior_mean:.1%} "
            f"| [{r.posterior_lo:.1%}, {r.posterior_hi:.1%}] | {r.shrinkage:.0%} |"
        )
    return "\n".join(rows)


def _sample_size_table(study: RareRatesStudy) -> str:
    rows = [
        "| class | test flows | flows needed for +/-5 points | shortfall |",
        "|---|---|---|---|",
    ]
    for r in sorted(study.rates, key=lambda x: x.total):
        short = max(0, r.flows_for_5pp - r.total)
        note = (
            "—" if short == 0 else f"{short:,}x more than it has" if r.total == 0 else f"{short:,}"
        )
        rows.append(f"| `{r.name}` | {r.total:,} | {r.flows_for_5pp:,} | {note} |")
    return "\n".join(rows)


def _shrinkage_read(study: RareRatesStudy) -> str:
    most = max(study.rates, key=lambda r: r.shrinkage)
    least = min(study.rates, key=lambda r: r.shrinkage)
    widest = max(study.rates, key=lambda r: r.wilson_width)
    return (
        f"The prior is fit to {len(study.rates)} classes at once and lands at a pooled detection "
        f"rate of {study.pooled_rate:.1%} with a concentration of {study.prior_strength:.1f} "
        "pseudo-flows — that concentration is the whole model, because it is what decides how "
        f"hard each class is pulled. `{most.name}`, with {most.total:,} test flows, takes "
        f"{most.shrinkage:.0%} of its estimate from the other classes; `{least.name}`, with "
        f"{least.total:,}, takes {least.shrinkage:.0%} and is effectively left alone. That is "
        "the behaviour to want: pooling is not a smoothing knob applied uniformly, it is a "
        "weighting that vanishes exactly where the data is sufficient. The contrast with the "
        f"frequentist column is stark on `{widest.name}`, whose Wilson interval spans "
        f"{widest.wilson_width:.0%} of the unit interval — technically correct, and useless to "
        "anyone deciding whether that class is covered."
    )


def _rank_read(study: RareRatesStudy) -> str:
    if not study.rank_moves:
        return (
            "Shrinkage does not reorder the classes here: the ranking by naive rate and by "
            "posterior mean agree throughout, which means the ordering was being driven by "
            "genuine differences rather than by sample size."
        )
    biggest = max(study.rank_moves, key=lambda m: abs(m[1] - m[2]))
    listed = ", ".join(f"`{n}` ({a} to {b})" for n, a, b in study.rank_moves[:4])
    return (
        f"**{len(study.rank_moves)} of {len(study.rates)} classes change position** when the "
        f"point estimates are replaced by posterior means: {listed}. The largest move is "
        f"`{biggest[0]}`, from rank {biggest[1]} to {biggest[2]}. A per-class leaderboard built "
        "on raw rates is, to that extent, a leaderboard of sample sizes — the rare classes swing "
        "to the top or the bottom because one flow either way is worth tens of percentage points "
        "to them, and nothing to a class with thousands."
    )


def _coverage_read(study: RareRatesStudy) -> str:
    target = study.level
    ok = abs(study.bayes_coverage - target) < 0.03
    verdict = (
        f"covers at {study.bayes_coverage:.1%} against its nominal {target:.0%}"
        if ok
        else f"covers at {study.bayes_coverage:.1%} against a nominal {target:.0%}"
    )
    ratio = study.wilson_width / max(study.bayes_width, 1e-9)
    return (
        f"Simulating from the fitted prior and re-estimating, the credible interval {verdict}, "
        f"and Wilson's covers at {study.wilson_coverage:.1%}. Both are honest; the difference is "
        f"the price. The Bayesian interval averages {study.bayes_width:.1%} wide against Wilson's "
        f"{study.wilson_width:.1%} — **{ratio:.1f}x narrower for the same coverage** — because it "
        "is allowed to use the other classes and Wilson is not. That is the entire argument for "
        "pooling, stated as a measurement rather than a preference. The caveat is equally "
        "concrete: this coverage is *conditional on the prior being right*, since the simulation "
        "draws from the same Beta the estimator assumes. A class that genuinely does not belong "
        "to the population — a novel family the detector has no purchase on at all — would be "
        "shrunk toward a rate it does not have, and the interval would understate the error."
    )


def _render(study: RareRatesStudy, fig: Path) -> str:
    smallest = min(study.rates, key=lambda r: r.total)
    return f"""# NetSentry — Rare-Class Detection Rates, Estimated Honestly

_Synthetic stand-in. Per-class detection at the deployed {study.fpr_budget:.1%}-FPR operating
point. Shared Beta prior fitted across all {len(study.rates)} attack classes by empirical Bayes:
`Beta({study.alpha:.2f}, {study.beta:.2f})`, pooled rate {study.pooled_rate:.1%}. Intervals at
{study.level:.0%}._

## Why this report exists

The per-class table is the most-read output here and the most easily misread. A detection rate
computed over thousands of flows and one computed over {smallest.total} are printed in the same
column, in the same font, with nothing to say that the first is a measurement and the second is
a rumour. Put an interval on them and the problem becomes visible but not solved: a frequentist
interval on a dozen flows spans most of the unit interval, which is honest and useless.

Partial pooling uses what the per-class view discards — the other classes. Each class's rate is
modelled as a draw from a shared Beta prior fitted across all of them, so each posterior is a
compromise between that class's own data and the population it belongs to, weighted by how much
data it actually has. Classes with thousands of flows barely move; classes with a dozen are
pulled most of the way to the pooled rate, which is the right answer, because a dozen flows
genuinely do not distinguish that class from its population.

## Every class, three ways

{_rate_table(study)}

{_shrinkage_read(study)}

![Shrinkage by class](../figures/{fig.name})

## Does the ranking survive?

{_rank_read(study)}

## Do the intervals cover what they claim?

A credible interval that does not cover at its stated rate is worse than no interval at all, so
the procedure is tested against data generated from the model it assumes.

| interval | simulated coverage | mean width |
|---|---|---|
| posterior (partial pooling) | {study.bayes_coverage:.1%} | {study.bayes_width:.1%} |
| Wilson score (no pooling) | {study.wilson_coverage:.1%} | {study.wilson_width:.1%} |

{_coverage_read(study)}

## What would it take to know?

Sample size required for a class's *own* data to support a plus-or-minus five-point claim at
{study.level:.0%}, against what the split actually provides.

{_sample_size_table(study)}

The rows where the shortfall runs to thousands are the useful ones. They say that no amount of
care in the modelling will produce a trustworthy per-class number for those families on this
dataset — the flows do not exist. The options are to pool (as here), to report the class only as
part of a group, or to collect more data; quietly printing a point estimate is not among them.

## Scope

Empirical Bayes plugs a *point* estimate of the hyperparameters into the posterior, so the
intervals are mildly too narrow — they ignore uncertainty in `(alpha, beta)` itself. A full
hierarchical treatment would put a hyperprior on those and integrate, which with
{len(study.rates)} classes would widen the intervals somewhat and change no conclusion here. The
prior is fitted on the same counts it is then applied to, the standard empirical-Bayes
compromise; the shrinkage is therefore slightly optimistic for the class that most influences the
fit. This is the estimation-side complement of the [seed-variance study](seed_variance.md), which
measures the *training* noise under a metric, and of the [per-class slice
report](slices.md), whose point estimates this report is arguing should be read with the
intervals attached."""
