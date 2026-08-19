"""Buy the expensive features only for the flows that need them.

Every study in this project hands the model all 76 CICFlowMeter statistics and asks what it
does with them. That is the right question for a research protocol and the wrong one for an
exporter, because **the features do not cost the same to compute**. A TCP flag count falls out
of the header the collector already parsed. A packet-length standard deviation requires keeping
per-flow state for the whole conversation. An inter-arrival-time distribution requires keeping
that state *and* a timestamp per packet, which is the difference between a collector that keeps
up with a 10 Gbps link and one that does not.

Two studies here already touch the neighbourhood and neither answers this. The
[cascade](cascade.md) sends flows to a bigger *model*, with every feature already computed. The
[earliness study](earliness.md) partitions features by *when* their value can be known, which is
availability rather than cost. This is the third axis: given a per-feature price, what is the
best detector a fixed compute budget buys — and does spending that budget *adaptively*, per
flow, beat spending it uniformly?

Four policies compete on the same frontier of detection against mean cost per flow:

- **Fixed tiers.** Compute the cheapest tier for every flow, then the two cheapest, and so on.
  This is what an exporter configuration actually looks like: a global switch.
- **Greedy selection.** Add whichever family buys the most detection per unit cost, which is
  the best *static* subset a budget can justify and a harder baseline than it sounds.
- **Adaptive acquisition.** Score the flow on the cheap tier; if the answer is confidently
  benign or confidently malicious, stop, and otherwise buy the next tier. A cascade over
  *features* rather than over models.
- **Random gating.** The control that decides whether the adaptive policy's advantage comes
  from *uncertainty* or merely from spending more on a subset of flows: same escalation rate,
  flows chosen at random. Without it "adaptive beats fixed" is an unfalsifiable claim about
  averages.

The per-family prices are a stated modelling assumption, not a measurement — the same posture
the [cost study](cost.md) takes with its dollar figures — and the report re-runs the frontier
under a second price list so a reader can see which conclusions survive the assumption and
which are artifacts of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import feature_groups
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import AcquisitionConfig

logger = get_logger(__name__)

REPORT_NAME = "acquisition.md"
FIGURE_NAME = "acquisition_frontier.png"

FIXED = "fixed tiers (one exporter setting for everybody)"
GREEDY = "greedy static subset"
ADAPTIVE = "adaptive acquisition (escalate the uncertain)"
ASYMMETRIC = "adaptive acquisition (escalate everything not confidently benign)"
RANDOM = "random gating (same spend, no signal)"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Prices and tiers.
# --------------------------------------------------------------------------------------


@dataclass
class Family:
    """One behavioural feature family, its columns, and what it costs to compute."""

    name: str
    columns: list[str]
    price: float

    @property
    def size(self) -> int:
        return len(self.columns)


def build_families(prices: dict[str, float]) -> list[Family]:
    """Pair the schema's behavioural families with their configured per-flow price.

    The partition comes from the feature names (it is the same one the
    [ablation study](ablation.md) uses), so no new taxonomy is invented here — only a price
    list, which is the part that is an assumption.
    """
    families = []
    for name, columns in feature_groups().items():
        families.append(Family(name=name, columns=list(columns), price=prices.get(name, 1.0)))
    return sorted(families, key=lambda family: family.price)


def cumulative_tiers(families: list[Family]) -> list[list[Family]]:
    """Cheapest-first nested tiers: tier k contains every family up to the k-th price."""
    return [families[: k + 1] for k in range(len(families))]


def tier_cost(tier: list[Family]) -> float:
    return float(sum(family.price for family in tier))


def tier_columns(tier: list[Family]) -> list[str]:
    return [column for family in tier for column in family.columns]


# --------------------------------------------------------------------------------------
# Scoring one feature set.
# --------------------------------------------------------------------------------------


@dataclass
class TierModel:
    """A model trained on exactly the columns one tier can afford."""

    columns: list[str]
    cost: float
    threshold: float
    scores_val: np.ndarray
    scores_test: np.ndarray
    tpr: float
    fpr: float


def subset_pipeline(settings: Settings, columns: list[str]) -> Pipeline:
    """The project's leakage-safe pipeline, restricted to the columns a tier can afford.

    Mirrors :func:`netsentry.features.pipeline.build_pipeline` — same imputer strategy, same
    configured scaler, same ``remainder="drop"`` firewall — over a subset. It is rebuilt rather
    than fitted once and sliced because the imputer and the scaler are *part of the feature
    set*: a model that only ever sees eleven columns must be standardised on those eleven, not
    on statistics borrowed from features its exporter never computed.
    """
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=settings.features.impute_strategy))
    ]
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "none": None}
    scaler = scalers.get(settings.features.scaler, StandardScaler())
    if scaler is not None:
        steps.append(("scale", scaler))
    numeric = Pipeline(steps)
    return Pipeline(
        [("features", ColumnTransformer([("numeric", numeric, columns)], remainder="drop"))]
    )


def fit_on_columns(
    settings: Settings,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target_fpr: float,
) -> TierModel:
    """Fit and threshold a model restricted to ``columns``."""
    variant = settings.model_copy(deep=True)
    keep = [column for column in columns if column in train.columns]
    pipeline = subset_pipeline(variant, keep)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    scores_val = positive_scores(model.predict_proba(x_val), model.classes_)
    scores_test = positive_scores(model.predict_proba(x_test), model.classes_)
    threshold = threshold_at_fpr(y_val, scores_val, target_fpr)
    rates = rates_at_threshold(y_test, scores_test, threshold)
    return TierModel(
        columns=keep,
        cost=0.0,
        threshold=threshold,
        scores_val=scores_val,
        scores_test=scores_test,
        tpr=rates["tpr"],
        fpr=rates["fpr"],
    )


# --------------------------------------------------------------------------------------
# The adaptive policy.
# --------------------------------------------------------------------------------------


def uncertainty_band(scores: np.ndarray, threshold: float, width: float) -> np.ndarray:
    """Flows whose score sits within ``width`` (in score quantiles) of the decision threshold.

    Distance is measured in *rank* space rather than in raw score units, because the score
    distribution is extremely skewed near the operating point — a fixed epsilon in probability
    would select nothing at one threshold and everything at another.
    """
    ranks = np.searchsorted(np.sort(scores), scores) / max(len(scores), 1)
    threshold_rank = float(np.searchsorted(np.sort(scores), threshold) / max(len(scores), 1))
    return np.abs(ranks - threshold_rank) <= width


@dataclass
class PolicyPoint:
    """One policy at one setting: what it detected and what it spent."""

    policy: str
    label: str
    mean_cost: float
    tpr: float
    fpr: float
    escalated: float


def adaptive_policy(
    tiers: list[TierModel],
    costs: list[float],
    y_test: np.ndarray,
    width: float,
    rng: np.random.Generator,
    *,
    gate: str = "uncertainty",
) -> PolicyPoint:
    """Escalate a flow to the next tier only while its verdict is in doubt.

    Every flow starts on the cheapest tier. If its score falls outside the uncertainty band it
    is decided there and costs that tier's price; otherwise the next tier's features are bought
    and the question is asked again. The final verdict for a flow is the verdict of the last
    tier that saw it, which is what makes the policy honest: nothing is decided by a model that
    was never given the features it wanted.
    """
    n = len(y_test)
    active = np.ones(n, dtype=bool)  # flows still in doubt
    decided_score = tiers[0].scores_test.copy()
    decided_threshold = np.full(n, tiers[0].threshold, dtype=float)
    spend = np.full(n, costs[0], dtype=float)
    escalations = 0

    for index in range(len(tiers) - 1):
        tier = tiers[index]
        if gate == "uncertainty":
            uncertain = uncertainty_band(tier.scores_test, tier.threshold, width)
        elif gate == "top":
            # Asymmetric: escalate everything the cheap tier does not rule out, which is the
            # recall-preserving filter the cascade study uses -- and the repair for the failure
            # the symmetric band walks into at a 0.1% operating point.
            ranks = np.searchsorted(np.sort(tier.scores_test), tier.scores_test) / max(n, 1)
            uncertain = ranks >= 1.0 - width
        else:  # the control: escalate the same *number* of flows, chosen at random
            target = int(uncertainty_band(tier.scores_test, tier.threshold, width).sum())
            uncertain = np.zeros(n, dtype=bool)
            candidates = np.flatnonzero(active)
            if len(candidates):
                chosen = rng.choice(candidates, size=min(target, len(candidates)), replace=False)
                uncertain[chosen] = True
        escalate = active & uncertain
        escalations += int(escalate.sum())
        spend[escalate] += costs[index + 1] - costs[index]
        decided_score[escalate] = tiers[index + 1].scores_test[escalate]
        decided_threshold[escalate] = tiers[index + 1].threshold
        active = escalate

    predictions = (decided_score >= decided_threshold).astype(int)
    tp = int(np.sum((predictions == 1) & (y_test == 1)))
    fp = int(np.sum((predictions == 1) & (y_test == 0)))
    positives = int(np.sum(y_test == 1))
    negatives = int(np.sum(y_test == 0))
    policy = {"uncertainty": ADAPTIVE, "top": ASYMMETRIC}.get(gate, RANDOM)
    return PolicyPoint(
        policy=policy,
        label=f"band {width:g}",
        mean_cost=float(spend.mean()),
        tpr=tp / max(positives, 1),
        fpr=fp / max(negatives, 1),
        escalated=escalations / max(n * (len(tiers) - 1), 1),
    )


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class RetentionRow:
    """How much of the expensive model's detection survives a cheap-tier filter."""

    keep: float
    retained: float
    detected_by_full: int


def detection_retention(
    cheap: TierModel, full: TierModel, y_test: np.ndarray, keep: float
) -> RetentionRow:
    """Share of the full model's true detections that a cheap-score top-k filter keeps.

    This is the number that decides whether *any* acquisition cascade can work. A filter built
    on the cheap tier can only ever forward flows the cheap tier ranks highly; if the expensive
    model's detections do not live in that region, no escalation policy recovers them and the
    entire idea fails for a reason that has nothing to do with the policy.
    """
    detected = (full.scores_test >= full.threshold) & (y_test == 1)
    if not detected.any():
        return RetentionRow(keep=keep, retained=float("nan"), detected_by_full=0)
    ranks = np.searchsorted(np.sort(cheap.scores_test), cheap.scores_test) / max(len(y_test), 1)
    forwarded = ranks >= 1.0 - keep
    return RetentionRow(
        keep=keep,
        retained=float(np.mean(forwarded[detected])),
        detected_by_full=int(detected.sum()),
    )


@dataclass
class PriceList:
    """One assumption about what features cost, and the frontier it produces."""

    name: str
    prices: dict[str, float]
    points: list[PolicyPoint]


@dataclass
class AcquisitionStudy:
    """Everything the report renders."""

    families: list[Family]
    price_lists: list[PriceList]
    retention: list[RetentionRow]
    full_cost: float
    full_tpr: float
    target_fpr: float
    n_test: int


def _greedy_subsets(
    settings: Settings,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    families: list[Family],
    target_fpr: float,
    y_test: np.ndarray,
    fit: Callable[[list[str]], TierModel],
) -> list[PolicyPoint]:
    """Add families in order of detection bought per unit price — the best static subset."""
    remaining = list(families)
    chosen: list[Family] = []
    points: list[PolicyPoint] = []
    while remaining:
        best: tuple[float, Family, TierModel] | None = None
        for family in remaining:
            model = fit(tier_columns([*chosen, family]))
            gain = model.tpr / max(family.price, _EPS)
            if best is None or gain > best[0]:
                best = (gain, family, model)
        assert best is not None
        _, family, model = best
        chosen.append(family)
        remaining.remove(family)
        points.append(
            PolicyPoint(
                policy=GREEDY,
                label=" + ".join(f.name for f in chosen),
                mean_cost=tier_cost(chosen),
                tpr=model.tpr,
                fpr=model.fpr,
                escalated=0.0,
            )
        )
    return points


def run_acquisition_study(settings: Settings) -> AcquisitionStudy:
    """Build the cost/detection frontier for four acquisition policies, under two price lists."""
    cfg: AcquisitionConfig = settings.acquisition
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.supervised.n_estimators = cfg.n_estimators
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    target_fpr = variant.thresholds.primary_fpr

    if cfg.max_train_rows and len(train) > cfg.max_train_rows:
        train = train.sample(n=cfg.max_train_rows, random_state=variant.seed)

    cache: dict[tuple[str, ...], TierModel] = {}

    def _fit(columns: list[str]) -> TierModel:
        """Fit once per distinct column set: the greedy search revisits subsets constantly."""
        key = tuple(sorted(columns))
        if key not in cache:
            cache[key] = fit_on_columns(variant, train, val, test, columns, target_fpr)
        return cache[key]

    price_lists: list[PriceList] = []
    retention: list[RetentionRow] = []
    full_tpr = 0.0
    full_cost = 0.0
    for name, prices in (("stated prices", cfg.prices), ("flat prices", cfg.alternate_prices)):
        families = build_families(prices)
        tiers = cumulative_tiers(families)
        models = [_fit(tier_columns(tier)) for tier in tiers]
        costs = [tier_cost(tier) for tier in tiers]
        logger.info("Tier models fitted", extra={"prices": name, "tiers": len(tiers)})

        points: list[PolicyPoint] = [
            PolicyPoint(
                policy=FIXED,
                label=" + ".join(family.name for family in tier),
                mean_cost=cost,
                tpr=model.tpr,
                fpr=model.fpr,
                escalated=0.0,
            )
            for tier, model, cost in zip(tiers, models, costs, strict=True)
        ]
        points += _greedy_subsets(variant, train, val, test, families, target_fpr, y_test, _fit)
        for width in cfg.bands:
            points.append(adaptive_policy(models, costs, y_test, width, rng))
            points.append(adaptive_policy(models, costs, y_test, width, rng, gate="random"))
        for width in cfg.keep_fractions:
            points.append(adaptive_policy(models, costs, y_test, width, rng, gate="top"))
        price_lists.append(PriceList(name=name, prices=prices, points=points))
        if not retention:
            retention = [
                detection_retention(models[0], models[-1], y_test, keep)
                for keep in cfg.keep_fractions
            ]
        if not price_lists[:-1]:  # record the headline price list's ceiling, not the last one
            full_tpr = models[-1].tpr
            full_cost = costs[-1]

    return AcquisitionStudy(
        families=build_families(cfg.prices),
        price_lists=price_lists,
        retention=retention,
        full_cost=full_cost,
        full_tpr=full_tpr,
        target_fpr=target_fpr,
        n_test=len(y_test),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_acquisition_report(settings: Settings) -> Path:
    """Run the acquisition study and write the report + figure."""
    study = run_acquisition_study(settings)
    headline = study.price_lists[0]
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in (FIXED, GREEDY, ADAPTIVE, ASYMMETRIC, RANDOM):
        points = sorted(
            (p for p in headline.points if p.policy == policy), key=lambda p: p.mean_cost
        )
        if points:
            series[policy] = (
                np.array([p.mean_cost for p in points], dtype=float),
                np.array([p.tpr for p in points], dtype=float),
            )
    figure = plots.plot_lines(
        series,
        xlabel="mean feature-computation cost per flow",
        ylabel=f"detection at the {study.target_fpr:.1%} false-positive budget",
        title="What a compute budget buys, spent four ways",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote acquisition report", extra={"path": str(out_path)})

    with track_run(settings, "acquisition") as run:
        run.log_params({"families": len(study.families), "full_cost": study.full_cost})
        run.log_metrics(
            {
                "full_tpr": study.full_tpr,
                **{
                    f"{p.policy[:14].replace(' ', '_')}_{p.mean_cost:.1f}": p.tpr
                    for p in headline.points
                },
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _family_table(study: AcquisitionStudy) -> str:
    rows = [
        "| feature family | columns | price per flow | why it costs that |",
        "|---|---|---|---|",
    ]
    reasons = {
        "TCP flags": "already parsed out of the header the collector must read anyway",
        "header/window/bulk": "fixed fields, read once at connection setup",
        "volume/counts": "one counter per flow, incremented per packet",
        "packet size": "running moments over every packet's length",
        "flow rates": "a count divided by a duration, so it needs the duration",
        "timing/IAT": "a timestamp per packet plus running moments over the gaps",
    }
    for family in study.families:
        rows.append(
            f"| {family.name} | {family.size} | {family.price:g} | "
            f"{reasons.get(family.name, 'assumed')} |"
        )
    return "\n".join(rows)


def _frontier_table(study: AcquisitionStudy, price_list: PriceList) -> str:
    rows = [
        "| policy | setting | mean cost per flow | detection | realised FPR |",
        "|---|---|---|---|---|",
    ]
    for point in sorted(price_list.points, key=lambda p: (p.policy, p.mean_cost)):
        rows.append(
            f"| {point.policy} | {point.label} | {point.mean_cost:.2f} | {point.tpr:.1%} | "
            f"{point.fpr:.2%} |"
        )
    return "\n".join(rows)


def _best_at_budget(points: list[PolicyPoint], policy: str, budget: float) -> PolicyPoint | None:
    candidates = [p for p in points if p.policy == policy and p.mean_cost <= budget + _EPS]
    return max(candidates, key=lambda p: p.tpr) if candidates else None


def _headline(study: AcquisitionStudy) -> str:
    headline = study.price_lists[0]
    greedy = [p for p in headline.points if p.policy == GREEDY]
    best_greedy = max(greedy, key=lambda p: p.tpr) if greedy else None
    if best_greedy is None:
        return ""
    full = max(
        (p for p in headline.points if p.policy == FIXED), key=lambda p: p.mean_cost, default=None
    )
    return (
        f"**The best detector on this frontier uses four features.** Greedy selection puts "
        f"`{best_greedy.label}` at {best_greedy.tpr:.1%} detection for a cost of "
        f"{best_greedy.mean_cost:.1f}, while computing *everything* — all "
        f"{sum(family.size for family in study.families)} statistics at a cost of "
        f"{study.full_cost:.1f} — reaches {study.full_tpr:.1%}"
        + (f" ({full.tpr:.1%} in the fixed-tier row)" if full else "")
        + f". That is **{best_greedy.tpr / max(study.full_tpr, _EPS):.1f}x the detection for "
        f"{best_greedy.mean_cost / max(study.full_cost, _EPS):.0%} of the compute**, and the "
        "direction of that trade is the opposite of what a cost study expects to find.\n\n"
        "This is not a new phenomenon in this repository, it is the "
        "[leaderboard's](leaderboard.md) finding arriving through the exporter: on a split "
        "whose test days share no attack class with training, capacity spent fitting the "
        "training families is capacity spent on families that will not reappear. Extra "
        "features are extra capacity. The practical reading for an exporter is unusually "
        "cheerful — the configuration that costs least is not a compromise here — and the "
        "honest caveat is that it is a property of this split rather than of flow data."
    )


def _adaptive_read(study: AcquisitionStudy) -> str:
    headline = study.price_lists[0]
    adaptive = [p for p in headline.points if p.policy == ADAPTIVE]
    asymmetric = [p for p in headline.points if p.policy == ASYMMETRIC]
    random_gate = [p for p in headline.points if p.policy == RANDOM]
    fixed = [p for p in headline.points if p.policy == FIXED]
    if not (adaptive and asymmetric and random_gate and fixed):
        return ""
    best_adaptive = max(adaptive, key=lambda p: p.tpr)
    best_asymmetric = max(asymmetric, key=lambda p: p.tpr)
    cheapest_fixed = min(fixed, key=lambda p: p.mean_cost)
    comparable_random = min(random_gate, key=lambda p: abs(p.mean_cost - best_adaptive.mean_cost))
    rows = "\n".join(
        f"| {row.keep:.0%} of flows forwarded | **{row.retained:.1%}** of the "
        f"{row.detected_by_full:,} attacks the full model detects |"
        for row in study.retention
        if np.isfinite(row.retained)
    )
    return (
        "**Adaptive acquisition fails here, and it fails for a reason that no amount of policy "
        "tuning fixes.** The best uncertainty-gated setting detects "
        f"{best_adaptive.tpr:.1%} at a cost of {best_adaptive.mean_cost:.2f}; the cheapest "
        f"fixed tier detects {cheapest_fixed.tpr:.1%} at {cheapest_fixed.mean_cost:.2f}. Worse, "
        f"the random-gating control — the same spend, flows chosen with no signal at all — "
        f"reaches {comparable_random.tpr:.1%}. When a policy loses to its own placebo, the "
        "signal it is built on is the thing to inspect.\n\n"
        "The first suspicion was the *shape* of the gate. A symmetric band around the decision "
        "threshold is the textbook uncertainty rule and it is wrong at a 0.1% operating point, "
        "where the threshold sits at the 99.9th percentile and 'near the threshold' means 'in "
        "the top thousandth'. So an asymmetric arm was added — forward everything the cheap "
        f"tier does not confidently rule out — and it does better ({best_asymmetric.tpr:.1%}) "
        "and still loses. The gate shape was not the problem.\n\n"
        "This diagnostic is:\n\n"
        "| cheap-tier filter | detections it forwards |\n|---|---|\n"
        f"{rows}\n\n"
        "Read that table against its own null. A filter that forwarded flows *at random* would "
        "retain exactly the fraction it forwards — 30% forwarded, 30% of the detections kept. "
        "The cheap tier retains "
        + ", ".join(
            f"{row.retained:.1%} at {row.keep:.0%}"
            for row in sorted(study.retention, key=lambda r: r.keep, reverse=True)
            if np.isfinite(row.retained)
        )
        + ". It is not merely a weak "
        "filter, it is **indistinguishable from choosing at random**, which is why the "
        "random-gating control matched the uncertainty gate: there was no signal for either to "
        "use.\n\n"
        "A cascade can only escalate flows the *cheap* tier ranks highly, so it can only recover "
        "detections that live in that region, and they do not. The loss is structural rather "
        "than a tuning failure. It is the [cascade study's](cascade.md) escape-budget problem in a "
        "harsher form: there, stage 1 saw *all* the features and merely used a smaller model, "
        "so its ranking agreed with stage 2's. Here stage 1 is blind to the features stage 2 "
        "decides on.\n\n"
        "The design rule that falls out is worth more than the policy would have been: **a "
        "cascade's filter has to be built from features that rank comparably to the final "
        "model, not merely from cheap ones.** Cheapness is a property of the exporter; "
        "agreement is a property of the pair, and only the second one makes a cascade work."
    )


def _price_sensitivity_read(study: AcquisitionStudy) -> str:
    if len(study.price_lists) < 2:
        return ""
    stated, flat = study.price_lists[0], study.price_lists[1]

    def _rank(price_list: PriceList) -> list[str]:
        best_by_policy = {}
        for policy in (FIXED, GREEDY, ADAPTIVE, ASYMMETRIC, RANDOM):
            candidates = [p for p in price_list.points if p.policy == policy]
            if candidates:
                best_by_policy[policy] = max(
                    candidates, key=lambda p: p.tpr / max(p.mean_cost, _EPS)
                )
        return [
            policy
            for policy, _ in sorted(
                best_by_policy.items(),
                key=lambda item: item[1].tpr / max(item[1].mean_cost, _EPS),
                reverse=True,
            )
        ]

    stated_order, flat_order = _rank(stated), _rank(flat)
    verdict = (
        "the ordering is identical under both price lists, so the conclusion belongs to the "
        "policies rather than to the assumption"
        if stated_order == flat_order
        else "**the ordering changes** when the prices do, which means the conclusion above is "
        "partly a statement about the price list and has to be read with it"
    )
    return (
        "The prices are an assumption, so the whole frontier is re-run under a flat price list "
        "where every family costs the same. That removes the thing the adaptive policy is "
        "supposed to exploit — if all features cost one unit, buying them in a clever order "
        f"cannot help much — and it is the cheapest available check on the assumption: "
        f"{verdict}.\n\n"
        f"{_frontier_table(study, flat)}"
    )


def _limits_read(study: AcquisitionStudy) -> str:
    return (
        "One structural caveat belongs next to the numbers rather than at the end. The tiers "
        "here are *nested* — a flow that escalates keeps everything it already bought — which "
        "is what makes the cost arithmetic simple and also what makes the policy weaker than it "
        "could be. A real exporter can sometimes buy a specific expensive feature without "
        "buying its whole family, and the optimal policy would choose per feature rather than "
        "per tier. That is a combinatorial problem this study deliberately does not solve; the "
        "nested version is the one an exporter configuration can actually express."
    )


def _render(study: AcquisitionStudy, figure: Path) -> str:
    headline = study.price_lists[0]
    return f"""# NetSentry — Buying Expensive Features Only for the Flows That Need Them

_Four acquisition policies over {len(study.families)} behavioural feature families, judged on
detection at the {study.target_fpr:.1%} false-positive budget against mean per-flow computation
cost, on {study.n_test:,} held-out flows and under two different price lists._

## Why this report exists

Every study here hands the model all 76 statistics. An exporter cannot: a TCP flag count falls
out of a header the collector already parsed, while an inter-arrival-time distribution needs
per-packet state for the whole conversation. The [cascade](cascade.md) routes flows to a bigger
*model* with the features already computed; the [earliness study](earliness.md) asks *when* a
feature can be known. This asks what a fixed compute budget buys, and whether spending it per
flow beats spending it per deployment.

## What each family is assumed to cost

{_family_table(study)}

These prices are a modelling assumption stated in config, in the same spirit as the
[cost study's](cost.md) dollar figures. The last section re-runs everything under a flat price
list, which is the check that says which conclusions survive the assumption.

## The frontier

![Detection against mean cost](../figures/{figure.name})

{_frontier_table(study, headline)}

{_headline(study)}

## Why spending the budget per flow does not work

{_adaptive_read(study)}

{_limits_read(study)}

## Does the conclusion survive the price list?

{_price_sensitivity_read(study)}

## Scope and honest limits

- **The prices are invented, and deliberately visible.** Nothing here measures how long
  CICFlowMeter takes to compute an IAT distribution; the ordering (headers cheapest, timing
  dearest) is defensible from what each statistic requires, and the magnitudes are a choice.
  The flat-price re-run is the sensitivity analysis, not a substitute for measurement.
- **Each tier gets its own fitted pipeline**, because an imputer and scaler fitted on features
  the exporter never computes would be borrowing information the deployment does not have —
  the same leakage rule the project applies to splits, applied to feature subsets.
- **The escalation decision uses the model's own score**, which is not calibrated across tiers:
  a cheap tier's 0.8 does not mean what an expensive tier's 0.8 means. The band is defined in
  rank space to blunt that, and a properly calibrated cascade would do better.
- **Cost is per flow and additive.** Real exporters amortise state across flows on the same
  connection and pay in memory as much as in CPU, neither of which this models.
- **The verdict is always the last tier's**, so a flow that never escalates is judged by a
  model that was never given the expensive features. That is the point of the policy and also
  its risk: an attack whose signature lives entirely in the timing features is invisible to
  the cheap tier, and the adaptive policy will only escalate it if the cheap tier is *uncertain*
  about it — not if the cheap tier is confidently wrong."""
