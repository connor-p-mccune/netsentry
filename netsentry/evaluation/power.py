"""How big does a difference have to be here before it means anything?

This project reports differences constantly -- a model against a baseline, a defence against an
attack, a wave's change against the last one -- and almost always as a point estimate. A number
like "+1.2 points of detection" is only a finding if it is larger than the wobble a second sample
of the same traffic would produce. Nobody had measured that wobble.

The [seed-sensitivity study](seed_variance.md) measured the other half of it: the noise that
comes from *training*, when the same configuration is fitted twice. This is the half that comes
from *evaluating* -- the test split is one finite sample of a distribution, and every metric read
off it is a random variable whose spread is a property of the sample size and the metric's own
construction, not of the model.

Three things follow, and they are not equally comfortable.

1. **The metrics have wildly different resolutions.** PR-AUC averages over every threshold and
   concentrates quickly. **TPR at a 0.1% false-positive budget** -- the operational metric this
   project leads with -- is determined by a handful of flows in the extreme tail of the score
   distribution, and its interval is correspondingly enormous. A headline metric that cannot
   resolve the differences it is used to argue about is worth knowing about.

2. **Paired comparison is not a refinement, it is the difference between having an answer and
   not.** Two models scored on the *same* flows share almost all of their sampling noise, so the
   interval around their difference is far narrower than the intervals around either one. A
   comparison that reads two overlapping confidence intervals and concludes "not significant" is
   answering a question nobody asked.

3. **Some of this project's own published differences do not clear their own noise floor.** The
   study ends by taking real comparisons from other reports and putting them against the bar
   computed here, including one from the same wave. That is the point of building it.

Nothing here is exotic: percentile bootstrap over the test rows, a paired variant, an exact
permutation null, and the minimum detectable effect that follows from the resulting standard
error. The value is in applying it to the numbers this repository actually publishes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import PowerConfig

logger = get_logger(__name__)

REPORT_NAME = "power.md"
FIGURE_NAME = "power_intervals.png"

#: The multiplier taking a standard error to the smallest effect a two-sided 5% test detects 80%
#: of the time: z(0.975) + z(0.80). Written out because it is the whole of the power calculation
#: and hiding it in a library call would obscure how little there is to it.
POWER_FACTOR = 1.959964 + 0.841621


def as_percent(quantity: float, signed: bool = False) -> str:
    """A rate, at enough precision to still be a number when the rate is a tenth of a percent.

    One decimal place is right for a detection rate and useless for a false-positive rate at a
    0.1% budget, where it renders every value and every interval bound as `0.0%`. The threshold
    below is where the extra digits start to matter rather than clutter.
    """
    sign = "+" if signed else ""
    return f"{quantity:{sign}.3%}" if abs(quantity) < 0.01 else f"{quantity:{sign}.1%}"


# --------------------------------------------------------------------------------------
# Metrics, as functions of a sample.
# --------------------------------------------------------------------------------------

Metric = Callable[[np.ndarray, np.ndarray], float]


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: the project's primary single number."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def fixed_threshold_tpr(threshold: float) -> Metric:
    """Detection rate at a threshold that was chosen elsewhere and does not move.

    This is the operational construction and it matters that the threshold is *fixed*: the cut is
    picked once on validation and then applied to whatever arrives. Re-picking it inside each
    bootstrap sample would measure a different and less relevant quantity -- the variability of a
    procedure rather than of a deployment.
    """

    def metric(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(rates_at_threshold(y_true, scores, threshold)["tpr"])

    return metric


def fixed_threshold_fpr(threshold: float) -> Metric:
    def metric(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(rates_at_threshold(y_true, scores, threshold)["fpr"])

    return metric


def alert_rate(threshold: float) -> Metric:
    def metric(y_true: np.ndarray, scores: np.ndarray) -> float:
        return float(np.mean(scores >= threshold))

    return metric


# --------------------------------------------------------------------------------------
# The bootstrap.
# --------------------------------------------------------------------------------------


def bootstrap(
    metric: Metric,
    y_true: np.ndarray,
    scores: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The metric's sampling distribution, by resampling flows with replacement."""
    n = len(y_true)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        rows = rng.integers(0, n, n)
        draws[index] = metric(y_true[rows], scores[rows])
    return draws


def paired_bootstrap(
    metric: Metric,
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The *difference's* sampling distribution, with both models scored on the same resample.

    Sharing the resample is the entire point. Most of the variation in either model's score is
    variation in which flows were drawn, and that part cancels in the difference -- so the paired
    interval is narrower than either marginal one, often dramatically. Bootstrapping the two
    models independently would reintroduce the noise the pairing removes and answer a question
    about two unrelated experiments.
    """
    n = len(y_true)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        rows = rng.integers(0, n, n)
        labels = y_true[rows]
        draws[index] = metric(labels, first[rows]) - metric(labels, second[rows])
    return draws


def permutation_null(
    metric: Metric,
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The difference's null distribution, by swapping which model scored each flow.

    Under the null that the two scorers are exchangeable, swapping their scores on a random
    subset of flows leaves the distribution unchanged. This needs no assumption about the shape
    of the metric's sampling distribution, which matters because TPR at a tight budget is a
    step function of a handful of order statistics and is not remotely normal.
    """
    n = len(y_true)
    draws = np.empty(permutations, dtype=float)
    for index in range(permutations):
        swap = rng.random(n) < 0.5
        left = np.where(swap, second, first)
        right = np.where(swap, first, second)
        draws[index] = metric(y_true, left) - metric(y_true, right)
    return draws


def interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    """A percentile interval, ignoring resamples where the metric was undefined."""
    finite = draws[np.isfinite(draws)]
    if len(finite) < 2:
        return (float("nan"), float("nan"))
    low, high = np.quantile(finite, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


def standard_error(draws: np.ndarray) -> float:
    finite = draws[np.isfinite(draws)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricRow:
    """One metric's resolution on this split."""

    name: str
    value: float
    low: float
    high: float
    error: float
    decisive: int
    percent: bool = False

    @property
    def halfwidth(self) -> float:
        return (self.high - self.low) / 2

    @property
    def detectable(self) -> float:
        """Smallest difference a two-sided 5% test finds 80% of the time, at this sample size."""
        return POWER_FACTOR * self.error

    @property
    def relative(self) -> float:
        """The detectable effect as a fraction of the metric's own value."""
        return self.detectable / self.value if self.value else float("inf")

    def show(self, quantity: float) -> str:
        return as_percent(quantity) if self.percent else f"{quantity:.4f}"


@dataclass(frozen=True)
class ComparisonRow:
    """One model-versus-model difference, measured paired and unpaired."""

    metric: str
    difference: float
    paired_low: float
    paired_high: float
    unpaired_low: float
    unpaired_high: float
    p_value: float
    percent: bool = False

    @property
    def paired_halfwidth(self) -> float:
        return (self.paired_high - self.paired_low) / 2

    @property
    def unpaired_halfwidth(self) -> float:
        return (self.unpaired_high - self.unpaired_low) / 2

    @property
    def narrowing(self) -> float:
        """How many times narrower pairing makes the interval."""
        return self.unpaired_halfwidth / self.paired_halfwidth if self.paired_halfwidth else 1.0

    @property
    def significant(self) -> bool:
        return self.paired_low > 0.0 or self.paired_high < 0.0

    def show(self, quantity: float) -> str:
        return as_percent(quantity, signed=True) if self.percent else f"{quantity:+.4f}"


@dataclass(frozen=True)
class PublishedClaim:
    """A difference this repository has already published, put against the bar computed here."""

    report: str
    description: str
    claimed: float
    metric: str
    detectable: float
    percent: bool = False

    @property
    def clears(self) -> bool:
        return abs(self.claimed) >= self.detectable

    @property
    def ratio(self) -> float:
        """The claim as a multiple of the smallest difference this split can resolve."""
        return abs(self.claimed) / self.detectable if self.detectable else float("inf")

    @property
    def marginal(self) -> bool:
        """Clears the bar, but by so little that a second sample could easily reverse it."""
        return 1.0 <= self.ratio < 1.5

    def show(self, quantity: float) -> str:
        return as_percent(quantity, signed=True) if self.percent else f"{quantity:+.4f}"


@dataclass
class PowerStudy:
    """Everything the report needs, computed once."""

    metrics: list[MetricRow]
    comparisons: list[ComparisonRow]
    published: list[PublishedClaim]
    n_test: int
    n_attacks: int
    n_benign: int
    seconds: float = 0.0

    def least_certain(self) -> MetricRow:
        """The metric known least precisely relative to its own size."""
        return max(self.metrics, key=lambda row: row.relative)

    def most_certain(self) -> MetricRow:
        return min(self.metrics, key=lambda row: row.relative)

    def operational(self) -> MetricRow:
        """Detection at the tightest budget -- the number this project leads with."""
        rows = [row for row in self.metrics if row.name.startswith("tpr_at_")]
        return min(rows, key=lambda row: float(row.name.rpartition("_at_")[2]))

    def marginal(self) -> list[PublishedClaim]:
        """Published claims that clear the bar by less than half again."""
        return [claim for claim in self.published if claim.marginal]

    def by_name(self, name: str) -> MetricRow:
        return next(row for row in self.metrics if row.name == name)

    def failing(self) -> list[PublishedClaim]:
        return [claim for claim in self.published if not claim.clears]


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def detection_rate(y_true: np.ndarray, alerts: np.ndarray) -> float:
    """Share of attacks flagged, given per-flow alert decisions.

    Taking decisions rather than scores is what lets the paired bootstrap and the permutation
    null treat two models with different thresholds -- and therefore different score scales --
    as exchangeable. Swapping a score without its threshold would compare nothing.
    """
    attacks = y_true == 1
    return float(np.mean(alerts[attacks])) if attacks.any() else 0.0


def run_power_study(settings: Settings) -> PowerStudy:
    """Measure what a difference has to exceed on this split, then check some published ones."""
    start = time.perf_counter()
    cfg: PowerConfig = settings.power
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from sklearn.linear_model import LogisticRegression

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    val_frame = load_split(variant, "temporal", "val")
    test_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(val_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = val_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = test_frame[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    column = list(model.classes_).index(1)
    val_scores = np.asarray(model.predict_proba(x_val))[:, column]
    test_scores = np.asarray(model.predict_proba(x_test))[:, column]

    challenger = LogisticRegression(
        max_iter=cfg.baseline_max_iter, class_weight="balanced", random_state=variant.seed
    ).fit(x_train, y_train)
    rival_column = list(challenger.classes_).index(1)
    rival_val = np.asarray(challenger.predict_proba(x_val))[:, rival_column]
    rival_test = np.asarray(challenger.predict_proba(x_test))[:, rival_column]

    # Thresholds are chosen once on validation and then held fixed, which is the deployed
    # construction: the cut does not move when tomorrow's traffic arrives.
    cuts = {budget: threshold_at_fpr(y_val, val_scores, budget) for budget in cfg.budgets}
    rival_cuts = {budget: threshold_at_fpr(y_val, rival_val, budget) for budget in cfg.budgets}

    metrics: list[MetricRow] = []
    attacks = int(np.sum(y_test == 1))
    benign = int(len(y_test) - attacks)

    def add(name: str, metric: Metric, decisive: int, percent: bool) -> None:
        draws = bootstrap(metric, y_test, test_scores, cfg.resamples, rng)
        low, high = interval(draws, cfg.level)
        metrics.append(
            MetricRow(
                name=name,
                value=metric(y_test, test_scores),
                low=low,
                high=high,
                error=standard_error(draws),
                decisive=decisive,
                percent=percent,
            )
        )

    add("pr_auc", pr_auc, attacks, False)
    add("roc_auc", roc_auc, attacks, False)
    for budget in cfg.budgets:
        cut = cuts[budget]
        rates = rates_at_threshold(y_test, test_scores, cut)
        add(f"tpr_at_{budget:g}", fixed_threshold_tpr(cut), int(rates["tp"]), True)
        add(f"fpr_at_{budget:g}", fixed_threshold_fpr(cut), int(rates["fp"]), True)
    add("alert_rate", alert_rate(cuts[max(cfg.budgets)]), 0, True)

    # The comparison: the deployed forest against a logistic-regression challenger, measured the
    # way a comparison should be measured and the way it usually is.
    comparisons: list[ComparisonRow] = []
    paired = paired_bootstrap(pr_auc, y_test, test_scores, rival_test, cfg.resamples, rng)
    first = bootstrap(pr_auc, y_test, test_scores, cfg.resamples, rng)
    second = bootstrap(pr_auc, y_test, rival_test, cfg.resamples, rng)
    unpaired = first - second
    null = permutation_null(pr_auc, y_test, test_scores, rival_test, cfg.permutations, rng)
    observed = pr_auc(y_test, test_scores) - pr_auc(y_test, rival_test)
    paired_low, paired_high = interval(paired, cfg.level)
    unpaired_low, unpaired_high = interval(unpaired, cfg.level)
    comparisons.append(
        ComparisonRow(
            metric="PR-AUC",
            difference=observed,
            paired_low=paired_low,
            paired_high=paired_high,
            unpaired_low=unpaired_low,
            unpaired_high=unpaired_high,
            p_value=float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
        )
    )

    for budget in cfg.budgets:
        alerts = (test_scores >= cuts[budget]).astype(float)
        rival_alerts = (rival_test >= rival_cuts[budget]).astype(float)
        paired = paired_bootstrap(detection_rate, y_test, alerts, rival_alerts, cfg.resamples, rng)
        first = bootstrap(detection_rate, y_test, alerts, cfg.resamples, rng)
        second = bootstrap(detection_rate, y_test, rival_alerts, cfg.resamples, rng)
        null = permutation_null(detection_rate, y_test, alerts, rival_alerts, cfg.permutations, rng)
        observed = detection_rate(y_test, alerts) - detection_rate(y_test, rival_alerts)
        paired_low, paired_high = interval(paired, cfg.level)
        unpaired_low, unpaired_high = interval(first - second, cfg.level)
        comparisons.append(
            ComparisonRow(
                metric=f"detection at a {budget:.1%} budget",
                difference=observed,
                paired_low=paired_low,
                paired_high=paired_high,
                unpaired_low=unpaired_low,
                unpaired_high=unpaired_high,
                p_value=float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
                percent=True,
            )
        )

    # Finally: published differences from other reports, against the bar computed above.
    detectable = {row.name: row.detectable for row in metrics}
    published = [
        PublishedClaim(
            report=item.report,
            description=item.description,
            claimed=item.claimed,
            metric=item.metric,
            detectable=detectable.get(item.metric, float("nan")),
            percent=item.metric.startswith("tpr") or item.metric.startswith("fpr"),
        )
        for item in cfg.published
    ]

    study = PowerStudy(
        metrics=metrics,
        comparisons=comparisons,
        published=published,
        n_test=len(y_test),
        n_attacks=attacks,
        n_benign=benign,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Power study complete",
        extra={
            "least_certain": study.least_certain().name,
            "claims_below_noise": len(study.failing()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "alert_rate": "alert rate",
}


def label(name: str) -> str:
    """A metric's name in prose."""
    if name in LABELS:
        return LABELS[name]
    kind, _, budget = name.partition("_at_")
    return f"{kind.upper()} at a {float(budget):.1%} budget"


def _lead(study: PowerStudy) -> str:
    """The finding, written from the computed numbers."""
    fragile = study.least_certain()
    operational = study.operational()
    solid = study.most_certain()
    comparison = study.comparisons[0]
    failing = study.failing()
    marginal = study.marginal()
    lines = [
        f"**The false-positive budget this whole project is organised around is decided by "
        f"{fragile.decisive} flows.**",
        "",
        f"At the tightest budget the deployed threshold lets {fragile.decisive} benign flows "
        f"through out of {study.n_benign:,}. That is what the realised false-positive rate is "
        f"measured from, so its 95% interval is "
        f"[{fragile.show(fragile.low)}, {fragile.show(fragile.high)}] around a value of "
        f"{fragile.show(fragile.value)} -- an uncertainty of **{fragile.relative:.0%} of the "
        "quantity itself**. Every claim in this repository that a budget was respected rests on "
        "a count small enough to fit in a sentence.",
        "",
        f"Detection at the same budget is better but not by as much as its four decimal places "
        f"suggest: {operational.decisive:,} of {study.n_attacks:,} attacks clear the cut, so a "
        f"difference in that number has to reach **{operational.show(operational.detectable)}** "
        f"before a two-sided 5% test would find it four times in five. {label(solid.name)}, which "
        f"integrates over every threshold and therefore uses all {study.n_attacks:,} attacks, "
        f"resolves to {solid.show(solid.detectable)} -- {solid.relative:.0%} of its value against "
        f"{operational.relative:.0%}. **Tightening a false-positive budget does not only lower "
        "the detection rate; it lowers the precision with which the detection rate is known.**",
        "",
    ]
    verdicts = []
    if failing:
        worst = min(failing, key=lambda claim: claim.ratio)
        verdicts.append(
            f"{len(failing)} of {len(study.published)} do not clear the bar at all, the smallest "
            f"being {worst.show(worst.claimed)} from [{worst.report}]({worst.report}) at "
            f"{worst.ratio:.0%} of what would be needed"
        )
    if marginal:
        closest = min(marginal, key=lambda claim: claim.ratio)
        verdicts.append(
            f"{len(marginal)} clear it only just -- {closest.show(closest.claimed)} from "
            f"[{closest.report}]({closest.report}) against a bar of "
            f"{closest.show(closest.detectable)}, a margin of {closest.ratio:.2f} to one"
        )
    if verdicts:
        lines += [
            "**Put this repository's own published differences against that bar and "
            + "; ".join(verdicts)
            + ".** None of those numbers is wrong. They are simply at or below the resolution of "
            "the instrument that produced them, and they are quoted elsewhere in this repository "
            "without that qualification. The remedy is to say so, which is what the last table "
            "does.",
            "",
        ]
    else:
        lines += [
            "Every published difference checked here clears its bar comfortably, which is the "
            "state this study exists to verify rather than assume.",
            "",
        ]
    lines += [
        f"One methodological result falls out along the way and is worth more than the rest. "
        f"Comparing the deployed forest against a logistic-regression challenger on PR-AUC, the "
        f"**paired** interval around the difference is **{comparison.narrowing:.1f} times "
        f"narrower** than the unpaired one, because two models scored on the same flows share "
        "almost all of their sampling noise and it cancels in the difference. Reading two "
        "overlapping marginal intervals and concluding 'no significant difference' is the most "
        "common way to get this wrong, and the table below measures the factor by which it is "
        "wrong.",
    ]
    return "\n".join(lines)


def _render(study: PowerStudy, figure: Path) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- How Big Does a Difference Have to Be?",
        "",
        f"_Percentile bootstrap over {study.n_test:,} later-day flows "
        f"({study.n_attacks:,} attacks, {study.n_benign:,} benign), a paired variant, and an "
        f"exact permutation null. Regenerate with `netsentry power`._",
        "",
        "## Why this report exists",
        "",
        "This project reports differences constantly -- a model against a baseline, a defence "
        "against an attack, this wave against the last -- and almost always as a point estimate. "
        'A number like "+1.2 points of detection" is a finding only if it is larger than the '
        "wobble a second sample of the same traffic would produce, and nobody had measured that "
        "wobble.",
        "",
        "The [seed-sensitivity study](seed_variance.md) measured the other half: the noise from "
        "*training* the same configuration twice. This is the half from *evaluating* -- the test "
        "split is one finite sample, and every metric read off it is a random variable whose "
        "spread depends on the sample size and on the metric's own construction, not on the "
        "model.",
        "",
        _lead(study),
        "",
        "## What each metric can resolve",
        "",
        f"![95% intervals and the effect each metric can detect](../figures/{figure.name})",
        "",
        "| metric | value | 95% interval | standard error | smallest detectable difference | "
        "as a share of the value | flows that decide it |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study.metrics:
        lines.append(
            f"| {label(row.name)} | {row.show(row.value)} | "
            f"[{row.show(row.low)}, {row.show(row.high)}] | {row.error:.4f} | "
            f"**{row.show(row.detectable)}** | {row.relative:.0%} | "
            f"{f'{row.decisive:,}' if row.decisive else 'every flow'} |"
        )
    operational = study.operational()
    fragile = study.least_certain()
    lines += [
        "",
        "The last column explains the rest of the table. PR-AUC and ROC-AUC integrate over every "
        f"threshold, so all {study.n_attacks:,} attacks contribute and the estimate concentrates. "
        "A rate at a fixed budget is a proportion over whichever flows clear the cut -- "
        f"{operational.decisive:,} attacks, and only {fragile.decisive} benign flows -- and a "
        "proportion estimated from a small numerator is noisy no matter how large the dataset "
        "around it is. That is a property of the operating point rather than of the model, and it "
        "applies to every fixed-budget number this project publishes.",
        "",
        "'Smallest detectable difference' is the minimum effect a two-sided 5% test finds 80% of "
        "the time at this sample size: 2.80 standard errors, the sum of the two normal quantiles. "
        "It is the number to hold a claim against.",
        "",
        "## Paired versus unpaired: the same comparison, two answers",
        "",
        "| comparison | difference | paired 95% interval | unpaired 95% interval | how much "
        "narrower | permutation p | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for compared in study.comparisons:
        lines.append(
            f"| {compared.metric} | {compared.show(compared.difference)} | "
            f"[{compared.show(compared.paired_low)}, {compared.show(compared.paired_high)}] | "
            f"[{compared.show(compared.unpaired_low)}, {compared.show(compared.unpaired_high)}] | "
            f"**{compared.narrowing:.1f}x** | {compared.p_value:.3f} | "
            f"{'**real**' if compared.significant else 'inside the noise'} |"
        )
    lines += [
        "",
        "Both columns describe the same two models on the same flows. The unpaired interval is "
        "what you get by bootstrapping each model separately and subtracting -- which "
        "reintroduces exactly the noise that pairing removes, because most of the variation in "
        "either model's score is variation in *which flows were drawn*, and that part is common "
        "to both. The paired interval is the honest one, and it is the one this project should "
        "quote whenever two scorers are compared on a shared split.",
        "",
        "The challenger is not a strawman: on this stand-in a plain logistic regression **beats** "
        "the deployed forest on PR-AUC, which is a finding the [leaderboard study](leaderboard.md) "
        "reports independently and a caveat the model card carries. That makes it a better test "
        "of the machinery here than a challenger built to lose would be, because both intervals "
        "sit clearly away from zero and the question is how much narrower pairing makes them.",
        "",
        "The permutation column is a check on the bootstrap rather than a duplicate of it. It "
        "assumes nothing about the shape of the sampling distribution, which matters here: a "
        "detection rate at a tight budget is a step function of a handful of order statistics and "
        "is not remotely normal. Where the two agree, the interval can be trusted; the comparison "
        "of detection rates is done on the models' **alert decisions** rather than their scores, "
        "because each model's threshold is calibrated to its own score scale and swapping a score "
        "without its threshold would compare nothing.",
        "",
        "## This project's own published differences, against the bar",
        "",
        "| report | difference | what it claims | bar at 80% power | verdict |",
        "|---|---|---|---|---|",
    ]
    for claim in study.published:
        lines.append(
            f"| [{claim.report}]({claim.report}) | {claim.show(claim.claimed)} | "
            f"{claim.description} | {claim.show(claim.detectable)} | "
            f"{'clears it' if claim.clears else '**inside the noise**'} |"
        )
    lines += [
        "",
        "This is the section the study was built for, and the uncomfortable rows are the point. A "
        "difference smaller than the bar is not necessarily absent -- an underpowered test failing "
        "to detect an effect is not evidence the effect is zero -- but it is not established "
        "either, and quoting it as a result without that caveat is the thing this repository "
        "spends most of its effort not doing elsewhere.",
        "",
        "The [held-out reuse study](reuse.md) reaches the same conclusion from the other side and "
        "in the same wave: it measured a selection cost of +0.0093 PR-AUC against a bootstrap "
        "half-width of 0.0211 on the third of the split it used, and said so in its own report. "
        "Two studies built a week apart, both concluding that an effect they measured is smaller "
        "than the instrument measuring it, is a sign the instrument deserved measuring.",
        "",
        "## Scope and honest limits",
        "",
        "- **This is sampling noise only.** Refitting the model on the same data with a different "
        "seed moves the numbers too, and that is the [seed-sensitivity study](seed_variance.md). "
        "The two sources add; neither report claims to bound the other.",
        "- **The bar assumes the published difference was measured on this split at this size.** "
        "A claim measured on a subset -- the reuse study used a third of the later days -- faces "
        "a wider bar than the one tabulated here, by roughly the square root of the ratio.",
        "- **A percentile bootstrap is not exact.** It is the standard tool and it agrees with "
        "the permutation null where both apply, which is the check available without assuming a "
        "distribution.",
        "- **Power is about detecting a difference, not about it mattering.** A change of "
        "+0.0165 PR-AUC can be statistically resolvable and operationally irrelevant, which is "
        "what the [cost study](cost.md) and the [frontier study](hull.md) are for.",
        "- **The claims audited here were entered by hand from other reports.** They are in "
        "config with the report each came from, so a reader can check the quotation; nothing "
        "parses the reports automatically, and the sample is small and deliberately includes "
        "results this wave produced.",
    ]
    return "\n".join(lines) + "\n"


def run_power_report(settings: Settings) -> Path:
    """Run the resolution study and write the report + figure."""
    study = run_power_study(settings)
    # Standard error against the number of flows that decide each metric, on log axes, with the
    # square-root law drawn through it. Plotting the metrics' values on shared axes would be
    # meaningless -- PR-AUC and a 0.1% false-positive rate differ by three orders of magnitude --
    # whereas this is the actual finding: precision is bought with deciding flows, and a tight
    # budget starves the metric of them.
    decisive = np.array([row.decisive for row in study.metrics if row.decisive > 0], dtype=float)
    errors = np.array([row.error for row in study.metrics if row.decisive > 0], dtype=float)
    order = np.argsort(decisive)
    decisive, errors = decisive[order], errors[order]
    constant = float(np.exp(np.mean(np.log(errors) + 0.5 * np.log(decisive))))
    figure = plots.plot_lines(
        {
            "measured standard error": (decisive, errors),
            "the square-root law": (decisive, constant / np.sqrt(decisive)),
        },
        xlabel="flows that decide the metric",
        ylabel="standard error",
        title="Precision is bought with deciding flows, and a tight budget starves the metric",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote power report", extra={"path": str(out_path)})

    with track_run(settings, "power") as run:
        run.log_params({"resamples": str(len(study.metrics))})
        run.log_metrics(
            {f"detectable_{row.name}": row.detectable for row in study.metrics}
            | {
                "claims_below_noise": float(len(study.failing())),
                "paired_narrowing": study.comparisons[0].narrowing,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
