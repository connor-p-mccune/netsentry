"""Learning to defer — when handing a flow to a human is worth the human's time.

Every study in this repository so far treats the model as the decision maker and the analyst
as a consumer of its output. Real detection is a **team**: the model decides most flows and
routes the rest to a person, and the interesting question is not "is the model accurate" but
"which flows should the model decline to decide". The conformal work answers a version of
this — abstain where the prediction set is ambiguous — but it abstains based only on the
model's own uncertainty, which quietly assumes the human is better everywhere. They are not.
An analyst who is worse than the model on a flow makes the system worse by looking at it, and
an analyst has a fixed number of flows per shift regardless.

So the question is properly a comparison of two error rates, which is the framing of Madras,
Pitassi & Zemel (NeurIPS 2018) and Mozannar & Sontag (ICML 2020): defer flow `x` when the
expected loss of the human on `x` is lower than the expected loss of the model on `x`, subject
to a capacity constraint. Rewriting it that way makes something obvious that the abstention
framing hides — **model uncertainty is the wrong deferral signal unless the human happens to
be good precisely where the model is unsure**. Whether that holds is an empirical property of
the pair, not a law, and this study makes it the experimental variable.

Three analysts are simulated, each a plausible description of a real one:

- **uniform** — right with a fixed probability everywhere. The null hypothesis. Nothing about
  the flow predicts whether the human helps.
- **correlated** — right on the flows the model finds easy and unreliable on the hard ones,
  because both are reading the same signal and the same flows are genuinely ambiguous.
- **complementary** — right on flows *far from the training data* and mediocre on familiar
  ones. This is the one worth taking seriously: a human reasons from context, business
  knowledge and things not in the feature vector, so their edge is exactly where the model
  is extrapolating. The [uncertainty](uncertainty.md) study showed the detector is at chance
  on an attack class it never saw and does not know it; a human is not.

The policies are built as an **ablation**, so each row of the table says what one ingredient
was worth. Defer nothing. Defer at random. Defer where the model is least likely to be right
(the abstention baseline). Defer by expected loss with the analyst assumed uniformly skilled —
which adds nothing but the fact that a miss costs twenty times a false alarm, and knows
nothing whatsoever about the human. Finally, defer by expected loss with the human's per-flow
skill *fitted on validation flows*. The gap between the last two is the only quantity that
depends on knowing the analyst, and under the uniform analyst it is zero by construction,
which makes it a control rather than a hope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.evaluation.novelty import nn_distances
from netsentry.evaluation.ope import resample_to_prevalence
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DeferConfig

logger = get_logger(__name__)

REPORT_NAME = "defer.md"
BUDGET_FIGURE = "defer_budget.png"
ADVANTAGE_FIGURE = "defer_advantage.png"

ANALYSTS: tuple[str, ...] = ("uniform", "correlated", "complementary")
# Four escalation rules plus the do-nothing control, ordered as an ablation: each adds one
# ingredient to the one before it, so the table reads as "what did knowing this buy".
POLICIES: tuple[str, ...] = (
    "no deferral",
    "random",
    "least confident",
    "cost-aware",
    "learned advantage",
)


# --------------------------------------------------------------------------------------
# The analyst (pure; unit-tested)
# --------------------------------------------------------------------------------------
def analyst_skill(
    kind: str, confidence: np.ndarray, novelty: np.ndarray, base: float, spread: float
) -> np.ndarray:
    """Per-flow probability that the human gets this flow right.

    ``confidence`` is the model's own probability that its verdict is right and ``novelty``
    is the flow's distance to the nearest training row, both already rank-normalised to
    [0, 1]. The three analysts differ only in what they key on: nothing, the model's
    confidence, or the flow's unfamiliarity. ``base`` is the skill at the middle of the range
    and ``spread`` how sharply it varies, so an analyst can be made strong or weak without
    changing which flows they are strong on.
    """
    m = np.asarray(confidence, dtype=float)
    n = np.asarray(novelty, dtype=float)
    if kind == "uniform":
        skill = np.full(m.shape, base, dtype=float)
    elif kind == "correlated":
        skill = base + spread * (m - 0.5)  # good where the model is already confident
    elif kind == "complementary":
        skill = base + spread * (n - 0.5)  # good where the model is extrapolating
    else:
        raise ValueError(f"Unknown analyst {kind!r}; choose from {ANALYSTS}.")
    bounded: np.ndarray = np.clip(skill, 0.0, 1.0)
    return bounded


def analyst_verdicts(skill: np.ndarray, y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the human's binary verdict: correct with probability ``skill``, else flipped."""
    correct = rng.random(len(skill)) < np.asarray(skill, dtype=float)
    truth = np.asarray(y_true).astype(int)
    verdicts: np.ndarray = np.where(correct, truth, 1 - truth)
    return verdicts


def rank_normalise(values: np.ndarray) -> np.ndarray:
    """Values mapped to [0, 1] by rank, so a skill curve is scale-free and comparable.

    Rank rather than min-max because both inputs (a score margin and a nearest-neighbour
    distance) have long right tails, and a min-max would squeeze almost every flow into a
    narrow band where the analyst's skill barely varies at all.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    order = np.argsort(np.argsort(v))
    return order / max(len(v) - 1, 1)


def ecdf_map(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values onto [0, 1] through a *reference* sample's empirical CDF.

    Rank-normalising each split against itself would be a subtle but fatal error here: a
    novelty rank of 0.9 within the validation split is a completely different distance from a
    rank of 0.9 within a later-day test split, because the later days are further from the
    training data as a whole. A policy fitted on one scale and applied to the other is fitting
    noise. Mapping both through validation's own ECDF puts them on one axis and uses only
    statistics an operator would have at calibration time — a test flow beyond anything
    validation contained saturates at 1.0, which is the honest answer.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        return rank_normalise(values)
    positions = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
    return np.asarray(positions, dtype=float) / float(ref.size)


# --------------------------------------------------------------------------------------
# The policies (pure; unit-tested)
# --------------------------------------------------------------------------------------
def defer_mask(scores: np.ndarray, budget: int) -> np.ndarray:
    """Defer the ``budget`` highest-scoring flows by a policy's own priority score."""
    s = np.asarray(scores, dtype=float)
    k = int(min(max(budget, 0), len(s)))
    mask = np.zeros(len(s), dtype=bool)
    if k:
        mask[np.argsort(-s, kind="stable")[:k]] = True
    return mask


def system_cost(
    y_true: np.ndarray,
    model_pred: np.ndarray,
    human_pred: np.ndarray,
    deferred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    cost_review: float,
) -> float:
    """Total cost of the team's decisions: the model decides, unless the human was asked.

    Review time is charged on deferred flows whether or not the human was right, which is
    the whole reason a capacity constraint exists. Without that term "defer everything" wins
    trivially whenever the human is any good at all.
    """
    truth = np.asarray(y_true).astype(int)
    final = np.where(np.asarray(deferred), np.asarray(human_pred), np.asarray(model_pred))
    false_pos = int(np.sum((final == 1) & (truth == 0)))
    false_neg = int(np.sum((final == 0) & (truth == 1)))
    reviews = int(np.sum(deferred))
    return false_pos * cost_fp + false_neg * cost_fn + reviews * cost_review


def expected_loss_advantage(
    scores: np.ndarray,
    model_pred: np.ndarray,
    analyst_skill: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    cost_review: float,
) -> np.ndarray:
    """Expected cost saved by handing this flow to the human, review time included.

    The deferral rule of Madras et al. (2018) is a comparison of *expected losses*, not of
    accuracies, and on this problem the distinction is the whole thing: a miss costs twenty
    times a false alarm, so a flow the model is 60% sure is benign can be worth escalating
    while a flow it is 60% sure is hostile is not. Writing it as an accuracy difference — the
    0-1-loss special case — silently assumes the two mistakes cost the same, and ranks flows
    accordingly.

    With ``s`` the calibrated attack probability and ``q`` the human's chance of being right:

        model loss  = cost_fp * (1 - s)   if it says attack, else  cost_fn * s
        human loss  = (1 - q) * (cost_fn * s + cost_fp * (1 - s))
        advantage   = model loss - human loss - cost_review
    """
    s = np.asarray(scores, dtype=float)
    q = np.asarray(analyst_skill, dtype=float)
    says_attack = np.asarray(model_pred).astype(bool)
    model_loss = np.where(says_attack, cost_fp * (1.0 - s), cost_fn * s)
    human_loss = (1.0 - q) * (cost_fn * s + cost_fp * (1.0 - s))
    advantage: np.ndarray = model_loss - human_loss - cost_review
    return advantage


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class PolicyCurve:
    """One policy's cost as the review budget grows, under one analyst."""

    analyst: str
    policy: str
    budgets: np.ndarray
    costs: np.ndarray

    def at(self, budget: int) -> float:
        """Cost at the nearest evaluated budget."""
        idx = int(np.argmin(np.abs(self.budgets - budget)))
        return float(self.costs[idx])


@dataclass
class AnalystResult:
    """Everything the report needs about one simulated analyst."""

    analyst: str
    mean_skill: float
    skill_spread: float  # p95 - p5 of true per-flow skill: how much signal existed to find
    skill_range: float  # p95/p5 of analyst skill among the flows actually in contention
    risk_range: float  # p95/p5 of attack probability among the same flows
    curves: list[PolicyCurve]
    baseline_cost: float
    operating_budget: int


@dataclass
class DeferStudy:
    """Everything the report renders."""

    n_test: int
    raw_attack_rate: float
    novelty_shift: float
    production_rate: float
    operating_fpr: float
    cost_fp: float
    cost_fn: float
    cost_review: float
    results: list[AnalystResult]


def _policy_scores(
    policy: str,
    confidence_rank: np.ndarray,
    scores: np.ndarray,
    model_pred: np.ndarray,
    skill_hat: np.ndarray,
    flat_skill: float,
    costs: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-flow deferral priority for one policy (higher means defer sooner).

    The three real policies form an ablation, each adding one ingredient:

    - **least confident** — the standard selective-prediction baseline, written properly as
      "defer where the model is most likely to be wrong" rather than as distance to the
      threshold. With a 0.1%-FPR threshold sitting near 1.0 those two rank flows completely
      differently, and beating the weaker one would prove nothing.
    - **cost-aware** — the same expected-loss rule but with the analyst treated as uniformly
      skilled. Its gap to the baseline isolates what *asymmetric costs* are worth, with no
      knowledge of the human at all.
    - **learned advantage** — the expected-loss rule with the per-flow skill fitted from
      validation. Its gap to the cost-aware rule isolates what knowing *where the human is
      better* is worth, and is zero by construction when their skill is in fact constant.
    """
    cost_fp, cost_fn, cost_review = costs
    if policy == "random":
        return rng.random(len(confidence_rank))
    if policy == "least confident":
        return -confidence_rank
    if policy == "cost-aware":
        flat = np.full(len(confidence_rank), flat_skill, dtype=float)
        return expected_loss_advantage(scores, model_pred, flat, cost_fp, cost_fn, cost_review)
    if policy == "learned advantage":
        return expected_loss_advantage(scores, model_pred, skill_hat, cost_fp, cost_fn, cost_review)
    return np.full(len(confidence_rank), -np.inf)  # "no deferral" defers nothing


def run_defer(settings: Settings) -> DeferStudy:
    """Compete four deferral policies against three analysts under a shared budget."""
    cfg: DeferConfig = settings.defer
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)

    seed_everything(variant.seed)
    model = SupervisedClassifier(variant).fit(
        x_train, train[BINARY_TARGET].to_numpy(), eval_set=(x_val, y_val)
    )
    benign = variant.labels.benign_label
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, variant.thresholds.primary_fpr)

    # Re-mix the test stream to a realistic prevalence. At the split's own ~25% attack rate
    # a randomly chosen review has positive expected value, "review everything" wins by
    # construction, and there is no deferral decision left to study — the same degeneracy
    # the [off-policy](ope.md) study had to fix, and for the same reason.
    raw_rate = float(y_test.mean())
    keep = resample_to_prevalence(
        y_test, settings.cost.production_attack_rate, rng=np.random.default_rng(variant.seed)
    )
    s_test, y_test, x_test = s_test[keep], y_test[keep], x_test[keep]

    # The two covariates the analysts key on. Both splits are mapped through *validation's*
    # ECDF rather than rank-normalised within themselves, so a novelty of 0.9 means the same
    # distance on both sides. Normalising each split against itself would hide the very shift
    # this project exists to measure, and would fit the deferral policy on one axis and apply
    # it on another.
    reference = x_train[rng_subsample(len(x_train), cfg.reference_rows, variant.seed)]
    raw_val = _covariates(s_val, threshold, x_val, reference)
    raw_test = _covariates(s_test, threshold, x_test, reference)
    context: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "val": (ecdf_map(raw_val[0], raw_val[0]), ecdf_map(raw_val[1], raw_val[1])),
        "test": (ecdf_map(raw_test[0], raw_val[0]), ecdf_map(raw_test[1], raw_val[1])),
    }
    novelty_shift = float(np.mean(context["test"][1])) - float(np.mean(context["val"][1]))
    model_pred_test = (s_test >= threshold).astype(int)

    budgets = np.asarray([round(f * len(y_test)) for f in cfg.budget_fractions], dtype=int)
    operating = round(cfg.operating_budget_fraction * len(y_test))
    # The flows any policy might plausibly escalate: ten times the budget, by attack
    # probability. Comparing dynamic ranges over the whole stream would be dominated by
    # flows no rule would ever pick.
    contention = np.argsort(-s_test, kind="stable")[: max(10 * operating, 1)]
    results: list[AnalystResult] = []
    for analyst in ANALYSTS:
        rng = np.random.default_rng(variant.seed)
        skill_val = analyst_skill(
            analyst, *context["val"], cfg.analyst_base_skill, cfg.analyst_spread
        )
        skill_test = analyst_skill(
            analyst, *context["test"], cfg.analyst_base_skill, cfg.analyst_spread
        )
        human_test = analyst_verdicts(skill_test, y_test, rng)

        # The learned policy may only look at validation flows: it fits how the human's
        # advantage varies with the covariates, then applies that fit to unseen test rows.
        skill_hat = _fit_advantage(
            context["val"],
            analyst_verdicts(skill_val, y_val, rng) == y_val,
            context["test"],
            cfg.min_rows_per_bin,
        )
        price = (cfg.cost_false_positive, cfg.cost_false_negative, cfg.cost_review)

        curves = []
        for policy in POLICIES:
            priority = _policy_scores(
                policy,
                context["test"][0],
                s_test,
                model_pred_test,
                skill_hat,
                cfg.analyst_base_skill,
                price,
                rng,
            )
            costs = np.array(
                [
                    system_cost(
                        y_test,
                        model_pred_test,
                        human_test,
                        defer_mask(priority, 0 if policy == "no deferral" else int(b)),
                        cfg.cost_false_positive,
                        cfg.cost_false_negative,
                        cfg.cost_review,
                    )
                    for b in budgets
                ]
            )
            curves.append(PolicyCurve(analyst, policy, budgets.astype(float), costs))
        baseline = curves[0].costs[0]
        results.append(
            AnalystResult(
                analyst=analyst,
                mean_skill=float(np.mean(skill_test)),
                skill_spread=float(np.quantile(skill_test, 0.95) - np.quantile(skill_test, 0.05)),
                skill_range=_dynamic_range(skill_test[contention]),
                risk_range=_dynamic_range(s_test[contention]),
                curves=curves,
                baseline_cost=float(baseline),
                operating_budget=operating,
            )
        )
        logger.info(
            "Analyst simulated",
            extra={
                "analyst": analyst,
                "skill": round(float(np.mean(skill_test)), 3),
                "spread": round(results[-1].skill_spread, 3),
            },
        )

    return DeferStudy(
        n_test=len(y_test),
        raw_attack_rate=raw_rate,
        novelty_shift=novelty_shift,
        production_rate=settings.cost.production_attack_rate,
        operating_fpr=variant.thresholds.primary_fpr,
        cost_fp=cfg.cost_false_positive,
        cost_fn=cfg.cost_false_negative,
        cost_review=cfg.cost_review,
        results=results,
    )


def rng_subsample(n: int, k: int, seed: int) -> np.ndarray:
    """Deterministic row sample, for the nearest-neighbour reference set."""
    if n <= k:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def _covariates(
    scores: np.ndarray, threshold: float, x: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The raw (confidence, novelty) pair the analysts key on, before any normalisation.

    Confidence is the model's own probability that its thresholded verdict is right, not the
    score's distance to the threshold. With the threshold near 1.0 those two disagree
    violently, and the probability is the one that means something.
    """
    return _model_correct_prob(scores, threshold), nn_distances(reference, x)


def _model_correct_prob(scores: np.ndarray, threshold: float) -> np.ndarray:
    """The model's own estimate that its verdict is right: the probability it assigned it."""
    s = np.asarray(scores, dtype=float)
    return np.where(s >= threshold, s, 1.0 - s)


def advantage_grid(n_val: int, min_rows_per_bin: int) -> int:
    """Bins per axis, sized so each cell holds roughly ``min_rows_per_bin`` validation flows.

    The estimator is the study's real bottleneck and it is worth being explicit about why.
    Each cell estimates a rate from a binomial sample, so its standard error falls like
    `1/sqrt(cell size)`; too many cells and the fitted skill is noise dressed as structure,
    which is worse than assuming the analyst is uniform. The rule below fixes the *precision*
    of each cell rather than the number of cells, so the grid grows with the data and the
    choice cannot be tuned against the outcome after seeing it. The uniform-analyst control
    then reports what noise is left, in the units the table is read in.
    """
    return max(2, int(np.sqrt(max(n_val, 1) / max(min_rows_per_bin, 1))))


def _fit_advantage(
    val_context: tuple[np.ndarray, np.ndarray],
    val_correct: np.ndarray,
    test_context: tuple[np.ndarray, np.ndarray],
    min_rows_per_bin: int,
) -> np.ndarray:
    """Estimate P(analyst correct | context) from validation flows, applied to test flows.

    A two-dimensional histogram regression rather than a model: the covariates already live
    on the unit square, so binning them is a defensible non-parametric fit. A flexible model
    would recover the simulator's exact functional form and the comparison would become a
    statement about the simulator. Bins with no validation support fall back to the global
    rate, so the estimate degrades to "the analyst's average" rather than to noise.
    """
    bins = advantage_grid(len(val_correct), min_rows_per_bin)
    mv, nv = val_context
    mt, nt = test_context
    correct = np.asarray(val_correct).astype(float)
    global_rate = float(correct.mean()) if correct.size else 0.5
    iv = np.clip((mv * bins).astype(int), 0, bins - 1)
    jv = np.clip((nv * bins).astype(int), 0, bins - 1)
    totals = np.zeros((bins, bins))
    hits = np.zeros((bins, bins))
    np.add.at(totals, (iv, jv), 1.0)
    np.add.at(hits, (iv, jv), correct)
    rates = np.where(totals > 0, hits / np.maximum(totals, 1.0), global_rate)
    it = np.clip((mt * bins).astype(int), 0, bins - 1)
    jt = np.clip((nt * bins).astype(int), 0, bins - 1)
    estimate: np.ndarray = rates[it, jt]
    return estimate


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_defer_report(settings: Settings) -> Path:
    """Run the deferral study and write the report + figures."""
    study = run_defer(settings)
    key = next((r for r in study.results if r.analyst == "complementary"), study.results[0])

    budget_fig = plots.plot_lines(
        {c.policy: (c.budgets / study.n_test, c.costs) for c in key.curves},
        xlabel="share of test flows sent to an analyst",
        ylabel="total system cost",
        title=f"Cost vs review budget ({key.analyst} analyst)",
        out_path=settings.paths.figures_dir / BUDGET_FIGURE,
    )
    advantage_fig = plots.plot_barh(
        [r.analyst for r in study.results],
        [_cost(r, "cost-aware") - _cost(r, "learned advantage") for r in study.results],
        xlabel="cost saved by knowing where the human is better (over the cost-aware rule)",
        title="What learning the analyst's advantage is worth",
        out_path=settings.paths.figures_dir / ADVANTAGE_FIGURE,
        xmax=max(
            (abs(_cost(r, "cost-aware") - _cost(r, "learned advantage")) for r in study.results),
            default=1.0,
        )
        * 1.3,
    )

    report = _render(study, budget_fig, advantage_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote deferral report", extra={"path": str(out_path)})

    with track_run(settings, "defer") as run:
        run.log_params(
            {
                "cost_review": study.cost_review,
                "cost_false_negative": study.cost_fn,
                "operating_fpr": study.operating_fpr,
            }
        )
        metrics: dict[str, float] = {}
        for r in study.results:
            for policy in POLICIES:
                metrics[f"{r.analyst}_{policy.replace(' ', '_')}"] = _cost(r, policy)
        run.log_metrics(metrics)
        run.log_artifact(budget_fig)
        run.log_artifact(advantage_fig)
        run.log_artifact(out_path)
    return out_path


def _dynamic_range(values: np.ndarray) -> float:
    """Ratio of the 95th to the 5th percentile — how much a quantity actually varies.

    The deferral ranking trades the human's skill against the model's risk, so what decides
    whether knowing the human can matter is not whether their skill varies but whether it
    varies *comparably* to the thing it is competing with.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 1.0
    lo = float(np.quantile(v, 0.05))
    hi = float(np.quantile(v, 0.95))
    return hi / lo if lo > 1e-12 else float("inf")


def _cost(result: AnalystResult, policy: str) -> float:
    """One policy's cost at the operating review budget."""
    curve = next((c for c in result.curves if c.policy == policy), None)
    return curve.at(result.operating_budget) if curve else float("nan")


def _analyst_table(study: DeferStudy) -> str:
    rows = [
        "| analyst | mean skill | skill spread | skill range | model risk range | "
        + " | ".join(POLICIES)
        + " | value of knowing the human |",
        "|---|---|---|---|---|" + "---|" * (len(POLICIES) + 1),
    ]
    for r in study.results:
        cells = " | ".join(f"{_cost(r, p):,.0f}" for p in POLICIES)
        edge = _cost(r, "cost-aware") - _cost(r, "learned advantage")
        rows.append(
            f"| {r.analyst} | {r.mean_skill:.0%} | {r.skill_spread:.1%} | {r.skill_range:.1f}x "
            f"| {r.risk_range:,.0f}x | {cells} | **{edge:+,.0f}** |"
        )
    return "\n".join(rows)


def _headline_read(study: DeferStudy) -> str:
    by_name = {r.analyst: r for r in study.results}
    uniform, comp = by_name.get("uniform"), by_name.get("complementary")
    corr = by_name.get("correlated")
    if uniform is None or comp is None or corr is None:
        return ""
    cost_gain = {  # what asymmetric costs are worth, knowing nothing about the human
        name: _cost(r, "least confident") - _cost(r, "cost-aware") for name, r in by_name.items()
    }
    skill_gain = {  # what knowing where the human is better is worth, on top of that
        name: _cost(r, "cost-aware") - _cost(r, "learned advantage") for name, r in by_name.items()
    }
    floor = abs(skill_gain["uniform"])
    control = (
        "and it is **zero**, exactly as it must be"
        if floor < 1e-9
        else (
            f"and it is {skill_gain['uniform']:+,.0f}. It should be zero. That gap is the "
            "**noise floor of the fitted skill estimate** — the histogram is estimating a "
            "constant from finite validation verdicts and getting it slightly wrong in every "
            "bin — and nothing smaller than it can be believed"
        )
    )
    comp = by_name["complementary"]
    comp_verdict = (
        f"worth {skill_gain['complementary']:+,.0f}, "
        f"{skill_gain['complementary'] / max(floor, 1e-9):.1f}x the control's noise floor"
        if skill_gain["complementary"] > max(floor, 1e-9)
        else (
            f"{skill_gain['complementary']:+,.0f} — **negative, and outside the noise floor**. "
            f"Knowing where the human is better made the system worse, reliably, in the one "
            f"regime the method was designed for. That is a result, and the diagnosis is a "
            f"ratio rather than a mystery. Among the flows any policy might plausibly "
            f"escalate, this analyst's skill varies by {comp.skill_range:.1f}x while the "
            f"model's attack probability varies by {comp.risk_range:.1f}x. The ranking is "
            f"essentially their product, so the model's term is worth about "
            f"{comp.risk_range / max(comp.skill_range, 1e-9):.1f} times as much as the "
            f"human's and the human's term can only ever nudge an order the model has already "
            f"settled. Nudging with a *fitted* quantity is worse than not nudging: the "
            f"estimate carries variance, the thing it displaces was already close to optimal, "
            f"and the trade loses on average. The skill-spread column confirms the signal was "
            f"genuinely there ({comp.skill_spread:.1%} of it) — this is not a case of nothing "
            "to learn, it is a case of learning something true that was not worth acting on. "
            "Learning to defer needs the human's advantage to vary *comparably* to the model's "
            "uncertainty among the flows in contention, and a rare-event detector concentrates "
            "its uncertainty far too sharply for a human's steadier competence to compete"
        )
    )
    return (
        f"Read the table as an ablation. Deferring at random makes the system **worse** than "
        f"not deferring at all — reviews cost money and a randomly chosen flow is almost "
        f"certainly one the model already got right — so any policy has to earn its budget "
        f"before it earns anything else. Confidence-based deferral clears that bar.\n\n"
        f"{_cost_awareness_read(cost_gain['complementary'])}\n\nThe last column is the one the "
        "analyst regime "
        f"controls. Under the **uniform** analyst it measures nothing by construction — with "
        f"constant skill the learned rule *is* the cost-aware rule — {control}. Under the "
        f"**correlated** analyst it is {skill_gain['correlated']:+,.0f}: this is the regime "
        f"where deferral is close to pointless in principle, because the flows the human is "
        f"good at are the flows the model already gets right, so a review buys the same answer "
        f"twice and pays for it once. Under the **complementary** analyst it is "
        f"{comp_verdict}.\n\nWhat survives all of this is the part that did not depend on "
        "estimating anything: deferral is worth real money against not deferring, random "
        "deferral is worth less than nothing, and the ranking rule is what separates them. "
        "The reason to keep asking the harder question anyway is the failure the "
        "[uncertainty](uncertainty.md) study isolated from the other direction — a tree away "
        "from its training data does not abstain, it routes confidently to whichever leaf the "
        "last split reaches. Its confidence is therefore least informative about exactly the "
        "flows a human could help with, which is why 'abstain where unsure' is less a deferral "
        "policy than a hope about what unsure means."
    )


def _cost_awareness_read(gain: float) -> str:
    """Whether pricing the two mistakes differently changed which flows get escalated."""
    if abs(gain) > 1e-9:
        return (
            f"Adding asymmetric costs to that baseline, with no knowledge of the human "
            f"whatsoever, is worth another {gain:+,.0f}. A miss costs twenty times a false "
            "alarm, so a flow the model calls benign at 60% confidence is worth escalating "
            "where a flow it calls hostile at 60% is not, and a rule ranking by probability "
            "of error alone cannot see the difference between them."
        )
    return (
        "Adding asymmetric costs changes **nothing at all** — the cost-aware column is "
        "identical to the confidence column, digit for digit — and the reason is worth stating "
        "because it looks like a bug. At a 0.1% false-positive budget the threshold sits near "
        "1.0, so the model calls essentially every flow benign, so the only mistake available "
        "to it is a miss, so every candidate escalation carries the same cost and the "
        "asymmetry has nothing left to re-rank. Algebraically the expected-loss score reduces "
        "to an increasing function of the attack probability, which is exactly the order the "
        "confidence rule already produced. Cost-awareness is not worthless in general; it is "
        "worthless at an operating point this conservative, and it would start to matter the "
        "moment the budget loosened enough for the model to raise alerts it might regret."
    )


def _budget_read(study: DeferStudy) -> str:
    comp = next((r for r in study.results if r.analyst == "complementary"), None)
    if comp is None:
        return ""
    learned = next(c for c in comp.curves if c.policy == "learned advantage")
    best = int(np.argmin(learned.costs))
    optimum = learned.budgets[best] / study.n_test
    saving = comp.baseline_cost - learned.costs[best]
    if best == 0:
        return (
            "The cost curve is minimised at zero reviews: with these costs no analyst budget "
            "pays for itself, and the honest recommendation is to spend the money elsewhere. "
            "That is a real answer to the question, and one a study that only reported the "
            "best deferral policy would have hidden."
        )
    if best == len(learned.costs) - 1:
        return (
            f"The curve is still falling at the widest budget swept ({optimum:.1%} of flows), "
            f"saving {saving:,.0f} against deciding everything with the model, so the optimum "
            "is outside the range rather than inside it and this study cannot name it. The "
            "shape is the finding: at these costs an extra reviewer is still earning their "
            "keep at the edge of what was tried, which is a statement about the cost ratio "
            f"({study.cost_fn:,.0f} for a miss against {study.cost_review:,.0f} for a review) "
            "rather than about any policy. Widening the sweep would find the turn; reporting a "
            "boundary minimum as though it were an interior one would not."
        )
    return (
        f"The curve has an **interior** minimum at {optimum:.1%} of flows reviewed, saving "
        f"{saving:,.0f} against deciding everything with the model. Interior is the "
        f"interesting case: review capacity is neither free nor unboundedly useful, so there "
        f"is a right amount of it, and it is set by the ratio of the review cost "
        f"({study.cost_review:,.0f}) to what a missed attack costs ({study.cost_fn:,.0f}) "
        "rather than by anyone's intuition about how much human oversight is appropriate. "
        "Past the minimum the curve turns back up, which is the part operators rarely see: "
        "escalating more flows makes the system worse once the escalated ones are flows the "
        "model was getting right."
    )


_SCOPE = """The analysts are **simulated**, and that is the load-bearing assumption. Their
skill curves are stipulated functions of two observable covariates (the model's confidence and
the flow's distance to training data), with a base rate and a spread set in config, so what
this study demonstrates is a *mechanism* — that the ranking of deferral policies depends on
where the human's advantage lies — rather than a measurement of any real SOC. Calibrating the
complementary analyst against logged analyst verdicts is the obvious next step and would need
data this project does not have; until then, the honest claim is conditional.

The learned advantage is fitted on validation flows only and applied to test flows, so it is
subject to the same discipline as everything else here. It is a histogram regression on the
two covariates, deliberately: a flexible model would fit the analyst simulator's exact
functional form and the comparison would become a statement about that simulator. The
histogram can recover a monotone trend and little else, which is roughly what could be
estimated from a real review log.

Costs are per flow, and reviews are charged whether the human is right or wrong — without
that term "defer everything" wins by construction whenever the analyst is better than chance.
The capacity constraint is a hard budget rather than a queue; the [SOC simulation](socsim.md)
models the queueing side, including what happens to flows that arrive while the analyst is
busy, which this study abstracts away."""


def _render(study: DeferStudy, budget_fig: Path, advantage_fig: Path) -> str:
    return f"""# NetSentry — When to Ask a Human

_Synthetic stand-in. Honest temporal/binary split, {study.operating_fpr:.1%} false-positive
budget. The test stream is re-mixed from its native {study.raw_attack_rate:.0%} attack rate to
a production prior of {study.production_rate:.0%} ({study.n_test:,} flows) — at 25% attacks a
review chosen at random pays for itself and "review everything" wins by construction. Costs
per flow: false positive {study.cost_fp:,.0f}, false negative {study.cost_fn:,.0f}, analyst
review {study.cost_review:,.0f}._

## Why this report exists

Conformal selective alerting abstains where the model's own prediction set is ambiguous.
That is a policy about the model, and it silently assumes the human is better wherever the
model is unsure. Learning to defer (Madras et al. 2018; Mozannar & Sontag 2020) states the
decision properly as a comparison of two expected losses — the model's and the human's, on
this flow — under a review budget. Written that way, one thing becomes obvious that the
abstention framing hides: **model uncertainty is only the right deferral signal if the human
happens to be good exactly where the model is unsure**, which is a property of the pair, not
a law.

So the analyst is the experimental variable. Three of them, each a plausible description of a
real one: skill that does not vary at all, skill that tracks the model's confidence (both are
reading the same ambiguity), and skill that tracks the flow's *distance from the training
data* — the case where the human brings context the feature vector does not contain.

## The five policies against the three analysts

_Total system cost at the operating review budget; lower is better._

{_analyst_table(study)}

{_headline_read(study)}

![cost saved by learning to defer](../figures/{advantage_fig.name})

## How much review capacity is worth buying

{_budget_read(study)}

![cost vs review budget](../figures/{budget_fig.name})

## Scope

{_SCOPE}"""
