"""Decide what to score when you cannot score everything — and estimate what you skipped.

Every measurement in this project assumes the model sees every flow. At line rate it does
not. A 10 Gbps link produces flow records faster than a boosted forest with SHAP attached can
consume them, and the two answers this repository already has are both about *making scoring
cheaper*: the [cascade](cascade.md) puts a cheap model in front of the expensive one and keeps
full coverage, and the [sketches](sketches.md) count in fixed memory without scoring at all.
Neither answers the question an operator actually faces when the budget is hard: **if only one
flow in twenty can be scored, which twenty — and what can still be said about the nineteen
that were not?**

That second half is the part usually left out, and it is a sampling problem with a
hundred-year-old answer. Under a design with known inclusion probabilities, the
**Horvitz-Thompson estimator** (1952) weights each observed flow by the reciprocal of its
probability of being observed and is *unbiased for the population total* — the number of
attacks in the whole stream, including the part nobody looked at. It comes with a variance,
so the answer is an interval rather than a number, and the interval's coverage is validated
here by simulation rather than asserted.

Four designs are compared at the same budget:

- **Uniform** Bernoulli sampling. The honest baseline: detection equals the budget, the
  estimator is unbiased, the interval is wide.
- **Stratified by service** with proportional and Neyman allocation (the latter spends the
  budget where the variance is, which is the textbook optimum for a fixed cost).
- **Priority sampling**, where a cheap logistic pre-filter sets each flow's inclusion
  probability proportional to how attack-like it looks, floored at a minimum so nothing is
  unreachable. This is probability-proportional-to-size sampling, and it is the design that
  can win on detection *and* variance at once — but only if the correction is applied.
- **Greedy top-k**: score the k most suspicious flows, deterministically. It maximises
  detections and it is the one design under which **no unbiased estimator of the total
  exists**, because flows below the cut have inclusion probability exactly zero and nothing
  observed can speak for them.

The last row is the point of the report. Greedy is what everybody actually builds, it wins
the metric everybody reports, and it silently destroys the ability to answer "how much did we
miss?" — which is the question that decides whether the budget was enough. The floor that
fixes it is the same exploration budget the [off-policy evaluation study](ope.md) found was
needed before a logged policy could be evaluated at all, arriving here from the sampling side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DESTINATION_PORT
from netsentry.data.services import service_of
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import SamplingConfig

logger = get_logger(__name__)

REPORT_NAME = "sampling.md"
FIGURE_NAME = "sampling_frontier.png"

UNIFORM = "uniform"
PROPORTIONAL = "stratified (proportional)"
NEYMAN = "stratified (Neyman)"
PRIORITY = "priority (probability-proportional-to-size)"
GREEDY = "greedy top-k"

_EPS = 1e-12

#: Deterministic Poisson sampling keeps a flow when its inclusion probability rounds up.
#: A rounding rule, not an operating point -- named so it cannot be mistaken for one.
_DETERMINISTIC_INCLUSION_CUT = 0.5


# --------------------------------------------------------------------------------------
# Designs: each returns the inclusion probability of every flow.
# --------------------------------------------------------------------------------------


def uniform_probabilities(n: int, budget: float) -> np.ndarray:
    """Every flow equally likely. The baseline every other design has to beat."""
    return np.full(n, float(np.clip(budget, 0.0, 1.0)))


def priority_probabilities(
    weights: np.ndarray, budget: float, floor: float, *, max_iterations: int = 50
) -> np.ndarray:
    """Inclusion probability proportional to a cheap score, floored and capped at one.

    Proportional-to-size sampling is only defined up to the cap: once a flow's share exceeds
    one it must be taken with certainty and its surplus redistributed, which is why this is a
    fixed-point iteration rather than a division. The ``floor`` is not a detail — it is what
    keeps every flow reachable, and with it the estimator defined at all.
    """
    n = len(weights)
    target = float(np.clip(budget, 0.0, 1.0)) * n
    w = np.clip(np.asarray(weights, dtype=float), _EPS, None)
    probabilities = np.full(n, float(np.clip(budget, 0.0, 1.0)))
    free = np.ones(n, dtype=bool)
    remaining = target
    for _ in range(max_iterations):
        share = w[free] / max(w[free].sum(), _EPS)
        candidate = np.clip(remaining * share, floor, 1.0)
        probabilities[free] = candidate
        saturated = free & (probabilities >= 1.0 - _EPS)
        if not saturated.any() or free.sum() == saturated.sum():
            break
        remaining = target - float(np.sum(probabilities[saturated]))
        free = free & ~saturated
        if remaining <= 0 or not free.any():
            break
    out: np.ndarray = np.clip(probabilities, floor, 1.0)
    return out


def stratified_probabilities(
    strata: np.ndarray, budget: float, *, allocation: str, values: np.ndarray | None = None
) -> np.ndarray:
    """Per-stratum inclusion probabilities under proportional or Neyman allocation.

    Proportional allocation gives every stratum the same rate, which is uniform sampling with
    a guarantee that no service is missed entirely. Neyman allocation spends more where the
    within-stratum standard deviation is larger, which is the variance-minimising split of a
    fixed sample size — and needs a variance estimate from somewhere, here the cheap
    pre-filter's scores rather than the labels nobody has yet.
    """
    n = len(strata)
    probabilities = np.zeros(n, dtype=float)
    total = float(np.clip(budget, 0.0, 1.0)) * n
    labels, counts = np.unique(strata, return_counts=True)
    if allocation == "neyman" and values is not None:
        spreads = np.array(
            [max(float(np.std(np.asarray(values)[strata == label])), _EPS) for label in labels]
        )
        weights = counts * spreads
    else:
        weights = counts.astype(float)
    allocation_sizes = total * weights / max(weights.sum(), _EPS)
    for label, count, size in zip(labels, counts, allocation_sizes, strict=True):
        probabilities[strata == label] = float(np.clip(size / max(count, 1), _EPS, 1.0))
    return probabilities


def greedy_probabilities(scores: np.ndarray, budget: float) -> np.ndarray:
    """Deterministic top-k selection expressed as (degenerate) inclusion probabilities.

    Every probability is 0 or 1, which is exactly why no unbiased estimator of the population
    total exists here: the Horvitz-Thompson weight of an unselected flow is 1/0.
    """
    n = len(scores)
    k = max(round(float(np.clip(budget, 0.0, 1.0)) * n), 1)
    cut = np.sort(np.asarray(scores))[::-1][k - 1]
    out: np.ndarray = (np.asarray(scores) >= cut).astype(float)
    return out


# --------------------------------------------------------------------------------------
# Estimation.
# --------------------------------------------------------------------------------------


def horvitz_thompson(observed: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Unbiased total and its variance estimate under Poisson sampling.

    ``sum(y_i / pi_i)`` over the sampled flows estimates the population total whatever the
    design, provided every ``pi_i > 0``. The variance estimator is the Poisson-design one,
    ``sum(y_i^2 (1 - pi_i) / pi_i^2)``, which needs no joint inclusion probabilities because
    independent Bernoulli draws have none worth tracking.
    """
    y = np.asarray(observed, dtype=float)
    pi = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1.0)
    total = float(np.sum(y / pi))
    variance = float(np.sum(y**2 * (1.0 - pi) / pi**2))
    return total, variance


def naive_total(observed: np.ndarray, probabilities: np.ndarray) -> float:
    """The estimate somebody writes when they forget the design: count, then scale by 1/rate.

    Correct under uniform sampling and badly wrong under any design that oversamples the
    thing being counted, which is exactly what a good sampler does.
    """
    y = np.asarray(observed, dtype=float)
    rate = float(np.mean(probabilities))
    return float(y.sum() / max(rate, _EPS))


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class DesignRow:
    """One sampling design at one budget, averaged over simulated draws."""

    design: str
    budget: float
    floor: float
    detection_rate: float
    scored_fraction: float
    ht_estimate: float
    ht_relative_error: float
    ci_width: float
    ci_coverage: float
    naive_estimate: float
    naive_relative_error: float
    estimable: bool


@dataclass
class SamplingStudy:
    """Everything the report renders."""

    rows: list[DesignRow]
    truth: int
    n_flows: int
    n_simulations: int
    budgets: list[float]
    floor: float
    strata: int
    cheap_pr_auc: float
    full_pr_auc: float


def _cheap_scores(
    settings: Settings, train: pd.DataFrame, frames: list[pd.DataFrame]
) -> list[np.ndarray]:
    """A pre-filter cheap enough to run on every flow: logistic regression on the pipeline.

    The design needs a score for *every* flow — including the ones it will decide not to score
    with the real model — so the pre-filter has to be something that costs a dot product. This
    is the same stage-1 idea the [cascade](cascade.md) uses, reused here to set inclusion
    probabilities instead of a forwarding cut.

    It is fitted **without** class rebalancing, which is the opposite of what every other model
    in this project does and is required here. Probability-proportional-to-size sampling wants
    a size measure proportional to `E[y | x]`, i.e. an actual probability of being an attack;
    a class-weighted fit deliberately distorts that probability towards the rare class, which
    inflates every benign flow's share of the budget and flattens the design towards uniform.
    Rebalancing helps a classifier and hurts a sampler.
    """
    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(train))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    return [
        positive_scores(
            model.predict_proba(np.asarray(pipeline.transform(frame))),
            np.asarray(model.classes_),
        )
        for frame in frames
    ]


def _simulate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    truth: int,
    n_simulations: int,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> tuple[float, float, float, float, float, float, float]:
    """Draw the sample ``n_simulations`` times and measure detection and estimation."""
    detections, scored, totals, widths, covered, naive = [], [], [], [], [], []
    z = 1.96
    for _ in range(n_simulations):
        if deterministic:
            taken = probabilities >= _DETERMINISTIC_INCLUSION_CUT
        else:
            taken = rng.random(len(probabilities)) < probabilities
        observed = labels * taken
        detections.append(float(observed.sum()) / max(truth, 1))
        scored.append(float(taken.mean()))
        total, variance = horvitz_thompson(observed[taken], probabilities[taken])
        half = z * float(np.sqrt(max(variance, 0.0)))
        totals.append(total)
        widths.append(2 * half)
        covered.append(float(abs(total - truth) <= half))
        naive.append(naive_total(observed[taken], probabilities))
    return (
        float(np.mean(detections)),
        float(np.mean(scored)),
        float(np.mean(totals)),
        float(np.mean(widths)),
        float(np.mean(covered)),
        float(np.mean(naive)),
        0.0,
    )


def run_sampling_study(settings: Settings) -> SamplingStudy:
    """Compare four designs at several budgets on detection *and* on what they can estimate."""
    cfg: SamplingConfig = settings.sampling
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    fit = fit_supervised(variant)
    benign_label = variant.labels.benign_label
    full_scores = attack_probability(fit.proba_test, fit.classes, benign_label)
    labels = (
        (np.asarray(fit.y_test) != benign_label).astype(int)
        if np.asarray(fit.y_test).dtype.kind in "OU"
        else np.asarray(fit.y_test).astype(int)
    )
    cheap = _cheap_scores(variant, train, [test])[0]

    from sklearn.metrics import average_precision_score

    truth = int(labels.sum())
    n = len(labels)
    ports = (
        test[DESTINATION_PORT].to_numpy()
        if DESTINATION_PORT in test.columns
        else np.zeros(n, dtype=float)
    )
    strata = np.array([service_of(float(p)) for p in ports])

    rows: list[DesignRow] = []
    for budget in cfg.budgets:
        designs: list[tuple[str, np.ndarray, bool]] = [
            (UNIFORM, uniform_probabilities(n, budget), False),
            (
                PROPORTIONAL,
                stratified_probabilities(strata, budget, allocation="proportional"),
                False,
            ),
            (
                NEYMAN,
                stratified_probabilities(strata, budget, allocation="neyman", values=cheap),
                False,
            ),
            (PRIORITY, priority_probabilities(cheap, budget, cfg.floor), False),
            (GREEDY, greedy_probabilities(cheap, budget), True),
        ]
        for name, probabilities, deterministic in designs:
            detection, scored, total, width, coverage, naive, _ = _simulate(
                probabilities,
                labels,
                truth=truth,
                n_simulations=1 if deterministic else cfg.n_simulations,
                rng=rng,
                deterministic=deterministic,
            )
            estimable = bool(np.all(probabilities > 0))
            rows.append(
                DesignRow(
                    design=name,
                    budget=budget,
                    floor=cfg.floor if name == PRIORITY else 0.0,
                    detection_rate=detection,
                    scored_fraction=scored,
                    ht_estimate=total,
                    ht_relative_error=(total - truth) / max(truth, 1),
                    ci_width=width,
                    ci_coverage=coverage if estimable else float("nan"),
                    naive_estimate=naive,
                    naive_relative_error=(naive - truth) / max(truth, 1),
                    estimable=estimable,
                )
            )
        logger.info("Budget simulated", extra={"budget": budget})

    return SamplingStudy(
        rows=rows,
        truth=truth,
        n_flows=n,
        n_simulations=cfg.n_simulations,
        budgets=list(cfg.budgets),
        floor=cfg.floor,
        strata=len(set(strata.tolist())),
        cheap_pr_auc=float(average_precision_score(labels, cheap)),
        full_pr_auc=float(average_precision_score(labels, full_scores)),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_sampling_report(settings: Settings) -> Path:
    """Run the sampling study and write the report + figure."""
    study = run_sampling_study(settings)
    budgets = np.array(study.budgets, dtype=float)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for design in (UNIFORM, PROPORTIONAL, NEYMAN, PRIORITY, GREEDY):
        points = [row for row in study.rows if row.design == design]
        if points:
            series[design] = (
                budgets,
                np.array([row.detection_rate for row in points], dtype=float),
            )
    figure = plots.plot_lines(
        series,
        xlabel="fraction of flows the model is allowed to score",
        ylabel="attacks detected (share of all attacks in the stream)",
        title="What each sampling design buys at the same compute budget",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote sampling report", extra={"path": str(out_path)})

    with track_run(settings, "sampling") as run:
        run.log_params({"budgets": str(study.budgets), "floor": study.floor})
        run.log_metrics(
            {
                f"detection_{row.design[:16].replace(' ', '_')}_{row.budget:g}": row.detection_rate
                for row in study.rows
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _design_table(study: SamplingStudy, budget: float) -> str:
    rows = [
        "| design | attacks detected | HT estimate of the total | relative error | 95% CI width "
        "| CI coverage | naive estimate | naive error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in study.rows:
        if row.budget != budget:
            continue
        coverage = "**undefined**" if not row.estimable else f"{row.ci_coverage:.0%}"
        estimate = "**none exists**" if not row.estimable else f"{row.ht_estimate:,.0f}"
        error = "—" if not row.estimable else f"{row.ht_relative_error:+.1%}"
        width = "—" if not row.estimable else f"{row.ci_width:,.0f}"
        rows.append(
            f"| {row.design} | {row.detection_rate:.1%} | {estimate} | {error} | {width} | "
            f"{coverage} | {row.naive_estimate:,.0f} | {row.naive_relative_error:+.1%} |"
        )
    return "\n".join(rows)


def _detection_table(study: SamplingStudy) -> str:
    header = "| design | " + " | ".join(f"{b:.0%} budget" for b in study.budgets) + " |"
    rows = [header, "|" + "---|" * (1 + len(study.budgets))]
    for design in (UNIFORM, PROPORTIONAL, NEYMAN, PRIORITY, GREEDY):
        cells = []
        for budget in study.budgets:
            match = next((r for r in study.rows if r.design == design and r.budget == budget), None)
            cells.append(f"{match.detection_rate:.1%}" if match else "—")
        rows.append(f"| {design} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _headline(study: SamplingStudy) -> str:
    budget = study.budgets[0]
    at_budget = {row.design: row for row in study.rows if row.budget == budget}
    uniform = at_budget.get(UNIFORM)
    priority = at_budget.get(PRIORITY)
    greedy = at_budget.get(GREEDY)
    if not (uniform and priority and greedy):
        return ""
    return (
        f"At a **{budget:.0%} compute budget** — one flow in "
        f"{round(1 / budget)} reaches the model — uniform sampling finds "
        f"{uniform.detection_rate:.1%} of the attacks, which is the budget and nothing more. "
        f"Priority sampling finds **{priority.detection_rate:.1%}** by spending its draws "
        "where a cheap logistic pre-filter says attacks are, and greedy top-k finds "
        f"{greedy.detection_rate:.1%} by removing the randomness altogether.\n\n"
        "Read only that column and the answer is obvious: take the top k. The next columns are "
        "why it is wrong."
    )


def _estimation_read(study: SamplingStudy) -> str:
    budget = study.budgets[0]
    at_budget = {row.design: row for row in study.rows if row.budget == budget}
    priority = at_budget.get(PRIORITY)
    uniform = at_budget.get(UNIFORM)
    if not (priority and uniform):
        return ""
    neyman = at_budget.get(NEYMAN)
    if neyman is None:
        return ""
    return (
        "A sampled detector answers a different question from a full one. It cannot say *"
        "these were the attacks*; it can only say *these are the attacks we looked at*, and "
        "the operational question — was the budget enough? — is about the ones nobody looked "
        "at. Horvitz-Thompson answers it: weight each observed flow by the reciprocal of its "
        "inclusion probability and the sum is unbiased for the population total, whatever the "
        "design, provided every probability is strictly positive.\n\n"
        f"Every randomised design is unbiased, and they are unbiased to within a percent or "
        f"two of the true {study.truth:,} — uniform at {uniform.ht_relative_error:+.1%}, "
        f"priority at {priority.ht_relative_error:+.1%}. The interval widths are where they "
        "part company, and the ordering is the opposite of the detection column:\n\n"
        f"- **Neyman-allocated stratification** produces the *narrowest* interval "
        f"({neyman.ci_width:,.0f}) while also improving detection to "
        f"{neyman.detection_rate:.1%};\n"
        f"- **priority sampling** detects the most of any randomised design "
        f"({priority.detection_rate:.1%}) and produces the *widest* interval "
        f"({priority.ci_width:,.0f}), wider than plain uniform sampling's "
        f"{uniform.ci_width:,.0f}.\n\n"
        "That inversion is worth understanding rather than tuning away. "
        "Probability-proportional-to-size sampling is variance-optimal when the size measure "
        "is proportional to the quantity being totalled — and here the quantity is a 0/1 "
        "attack indicator, so the optimal design would take every attack with certainty and "
        "no benign flow at all. The pre-filter is a *noisy* stand-in for that, and its "
        "mistakes are expensive in a specific way: an attack the pre-filter scores low is "
        "sampled with a tiny probability and therefore arrives carrying an enormous "
        "`1 / pi` weight. The variance of the estimate is dominated by exactly the attacks the "
        "sampler is worst at recognising. A better pre-filter narrows this interval; a "
        "confident and wrong one widens it without warning.\n\n"
        f"Coverage is measured rather than asserted — {uniform.ci_coverage:.0%} for uniform "
        f"and {priority.ci_coverage:.0%} for priority across {study.n_simulations} draws "
        "against a nominal 95%. That check is not a formality: the interval is a normal one "
        "around a statistic whose distribution is skewed by a handful of enormous weights, so "
        "it is the kind of interval that can quietly miss its level, and the only way to know "
        "is to draw the sample a few hundred times and count."
    )


def _naive_read(study: SamplingStudy) -> str:
    budget = study.budgets[0]
    at_budget = {row.design: row for row in study.rows if row.budget == budget}
    priority = at_budget.get(PRIORITY)
    uniform = at_budget.get(UNIFORM)
    if not (priority and uniform):
        return ""
    return (
        "The naive columns are the trap. Counting the attacks you found and dividing by the "
        "sampling rate is correct under uniform sampling — its error is "
        f"{uniform.naive_relative_error:+.1%} — and catastrophically wrong under any design "
        "that deliberately oversamples the thing being counted. The priority design's naive "
        f"estimate is {priority.naive_estimate:,.0f} against a true {study.truth:,}, an error "
        f"of **{priority.naive_relative_error:+.0%}**. The better the sampler, the worse the "
        "naive estimate, because the bias *is* the sampler's skill counted twice. Any dashboard "
        "that reports 'attacks seen / sampling rate' on a smart sampler is reporting a number "
        "with this bias baked in."
    )


def _greedy_read(study: SamplingStudy) -> str:
    budget = study.budgets[0]
    greedy = next((r for r in study.rows if r.design == GREEDY and r.budget == budget), None)
    priority = next((r for r in study.rows if r.design == PRIORITY and r.budget == budget), None)
    if not (greedy and priority):
        return ""
    return (
        f"Greedy top-k detects {greedy.detection_rate:.1%} at this budget against priority "
        f"sampling's {priority.detection_rate:.1%} — a real advantage, and it costs the "
        "estimator entirely. Every flow below the cut has inclusion probability exactly zero, "
        "so its Horvitz-Thompson weight is `1/0` and **no unbiased estimator of the total "
        "exists**. This is not a limitation of the technique used here; it is a theorem about "
        "the design. Nothing observed can speak for a region that could never have been "
        "observed.\n\n"
        "The consequence is operational rather than statistical. A greedy sampler cannot tell "
        "you whether its budget is adequate, because the evidence that would say so lives "
        "exactly where it never looks — and when the traffic mix shifts underneath it, the "
        "alert count stays flat and looks like stability. The fix is cheap: a floor on the "
        f"inclusion probability (here {study.floor:g}) makes every flow reachable, costs a "
        "sliver of the budget, and turns 'we found this many' into 'there were about this "
        "many'. It is the same exploration budget the [off-policy evaluation study](ope.md) "
        "found was the difference between a log that can be evaluated and one that cannot."
        + _crossover_read(study)
    )


def _crossover_read(study: SamplingStudy) -> str:
    """Where the randomised design overtakes greedy, if it does inside the swept budgets."""
    crossover = next(
        (
            budget
            for budget in study.budgets
            if (
                priority := next(
                    (r for r in study.rows if r.design == PRIORITY and r.budget == budget), None
                )
            )
            and (
                greedy := next(
                    (r for r in study.rows if r.design == GREEDY and r.budget == budget), None
                )
            )
            and priority.detection_rate > greedy.detection_rate
        ),
        None,
    )
    if crossover is None:
        return (
            "\n\nGreedy keeps its detection lead across every budget swept here, so the "
            "argument for the floor is entirely about what can be *said*, not about what is "
            "caught."
        )
    priority = next(r for r in study.rows if r.design == PRIORITY and r.budget == crossover)
    greedy = next(r for r in study.rows if r.design == GREEDY and r.budget == crossover)
    return (
        "\n\n**And greedy's detection lead is not even permanent.** By a "
        f"{crossover:.0%} budget the randomised priority design overtakes it — "
        f"{priority.detection_rate:.1%} against {greedy.detection_rate:.1%} — because greedy "
        "spends its entire budget inside the region where the pre-filter is already confident, "
        "and the attacks the pre-filter does not recognise are unreachable by construction no "
        "matter how large the budget grows. Randomisation is not only what makes the stream "
        "estimable; past a certain budget it is also what finds more attacks."
    )


def _recommendation_read(study: SamplingStudy) -> str:
    budget = study.budgets[0]
    at_budget = {row.design: row for row in study.rows if row.budget == budget}
    priority = at_budget.get(PRIORITY)
    neyman = at_budget.get(NEYMAN)
    greedy = at_budget.get(GREEDY)
    if not (priority and neyman and greedy):
        return ""
    return (
        "There is no single winner here, and pretending otherwise is how sampling gets "
        "designed badly. The three columns rank the designs differently, so the choice is a "
        "statement about which question the deployment has to answer:\n\n"
        "| if the job is... | run | because |\n|---|---|---|\n"
        f"| catch as many attacks as the budget allows, and nothing else is asked | greedy "
        f"top-k | {greedy.detection_rate:.1%} detection, and no answer to 'what did we miss' "
        "at any price |\n"
        f"| catch attacks *and* report coverage | priority sampling with a floor | "
        f"{priority.detection_rate:.1%} detection, unbiased totals, the widest interval |\n"
        f"| estimate the threat level in the stream | Neyman-allocated strata | "
        f"{neyman.ci_width:,.0f}-wide interval, {neyman.detection_rate:.1%} detection |\n\n"
        "The middle row is the default worth arguing for. At this budget it gives up "
        f"{greedy.detection_rate - priority.detection_rate:.1%} of detection — "
        f"{priority.detection_rate / max(greedy.detection_rate, 1e-9):.0%} of greedy's rate — "
        "and buys back the ability to say *there were about this many attacks, plus or minus "
        "this much*, which is the sentence a capacity decision is made from and the one a "
        "greedy sampler can never produce. At larger budgets it gives up nothing at all, "
        "because it overtakes greedy outright."
    )


def _render(study: SamplingStudy, figure: Path) -> str:
    return f"""# NetSentry — Scoring a Fraction of the Stream, and Estimating the Rest

_Four sampling designs over {study.n_flows:,} test-day flows containing {study.truth:,} attacks,
each simulated {study.n_simulations} times per budget. The cheap pre-filter that drives the
adaptive designs reaches {study.cheap_pr_auc:.3f} PR-AUC against the full model's
{study.full_pr_auc:.3f}._

## Why this report exists

Every measurement in this project assumes the model sees every flow. At line rate it does not.
The [cascade](cascade.md) makes scoring cheaper while keeping full coverage and the
[sketches](sketches.md) count without scoring at all; neither answers the question that arrives
when the budget is genuinely hard — **if one flow in twenty can be scored, which twenty, and
what can be said about the nineteen that were not?**

## What each design detects

![Detection by budget](../figures/{figure.name})

{_detection_table(study)}

{_headline(study)}

## The nineteen flows nobody looked at

{_design_table(study, study.budgets[0])}

{_estimation_read(study)}

## The estimator everybody writes instead

{_naive_read(study)}

## Why the best detector is the worst design

{_greedy_read(study)}

## Which design to actually run

{_recommendation_read(study)}

## Scope and honest limits

- **Poisson sampling, not fixed-size.** Each flow is an independent Bernoulli draw, so the
  realised sample size varies around the budget; that is what makes the variance estimator a
  one-liner with no joint inclusion probabilities. A fixed-size design (systematic or
  conditional Poisson) removes the size variation and needs second-order probabilities the
  operator would then have to track.
- **The pre-filter is trained on the training days** and applied to the test days, so its
  quality carries the same temporal-shift discount as everything else here. A pre-filter that
  degrades makes the priority design's inclusion probabilities wrong — which costs variance,
  not bias, because Horvitz-Thompson is unbiased for *any* strictly positive design.
- **The total estimated here is a flow count.** A SOC cares about incidents, and flows within
  one attack burst are near-duplicates, so the effective sample behind the interval is
  smaller than the flow count suggests. The [campaign study](campaigns.md) is where the
  incident-level unit is measured.
- **Neyman allocation uses the pre-filter's scores as its variance proxy**, because the labels
  it formally requires are exactly what the sampling exists to avoid needing. That makes it an
  approximation of the optimal allocation rather than the optimal allocation.
- **Compute is modelled as a flow count.** Real scoring cost varies per flow (SHAP is ~73% of
  request latency, per the [serving benchmark](../../README.md)), so a budget in flows is a
  budget in the average case rather than the worst one."""
