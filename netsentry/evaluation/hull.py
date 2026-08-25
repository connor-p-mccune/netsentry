"""Is the deployed operating point even on the frontier?

Every decision this project ships is a **threshold**: pick a score cut on validation at a
false-positive budget, apply it to everything after. That is the standard construction and it
quietly assumes something nobody here has checked -- that a threshold is the best rule
available at its own false-positive rate.

It need not be. The classical result (Provost & Fawcett, ML 2001) is that the achievable
operating points of a scoring classifier are the **convex hull** of its ROC points, not the ROC
curve itself: wherever the curve dips below its own hull, a *randomised* rule -- flip a biased
coin, use one of two thresholds -- strictly dominates every plain threshold at that
false-positive rate. Free detection, in exchange for a coin.

Three questions follow, and the module answers them in order.

1. **Is the deployed cut dominated, and by how much?** Measured on validation, where the cut is
   chosen, in points of detection.
2. **Does the gain survive contact with the later days?** A dip in an empirical ROC can be real
   structure or a finite-sample wobble, and those two look identical until the rule derived from
   one is applied to the other. This is the question the study exists for.
3. **What would it cost even if it worked?** A randomised rule returns different verdicts for
   the same flow, which is not a metaphor -- it is the exact determinism relation the
   [metamorphic study](metamorphic.md) tests for and the exact property the
   [canary](state_machine.md) checks at load time.

The last two sections leave thresholds behind entirely, because a single false-positive budget
is a statement about one deployment. **Cost curves** (Drummond & Holte, ML 2006) show the range
of operating conditions over which each cut is optimal, and **net benefit** (Vickers & Elkin,
Med Decis Making 2006) asks the question a clinical paper always asks and an ML paper almost
never does: is the model better than alerting on everything, or on nothing, at the preferences
an operator might actually hold?
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_curve

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import HullConfig

logger = get_logger(__name__)

REPORT_NAME = "hull.md"
HULL_FIGURE = "hull_frontier.png"
BENEFIT_FIGURE = "hull_net_benefit.png"


# --------------------------------------------------------------------------------------
# The frontier.
# --------------------------------------------------------------------------------------


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every achievable (FPR, TPR) and the threshold that reaches it."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)


def upper_hull(fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
    """Indices of the ROC convex hull's vertices, in increasing false-positive rate.

    A monotone chain keeping only left turns, which for points sorted by ``fpr`` leaves the
    upper hull -- the set of operating points no mixture can improve on. Everything strictly
    inside is dominated: some coin-weighted pair of hull vertices achieves a higher detection
    rate at the same false-positive rate.
    """
    order = np.lexsort((tpr, fpr))
    hull: list[int] = []
    for index in order:
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (fpr[second] - fpr[first]) * (tpr[index] - tpr[first]) - (
                tpr[second] - tpr[first]
            ) * (fpr[index] - fpr[first])
            # Pop on a left turn: `second` sits on or below the chord from `first` to `index`,
            # so no mixture would ever choose it. Popping on the *right* turn instead builds
            # the lower hull, which is the same code returning the diagonal -- the first
            # version of this function did exactly that, and the giveaway was a "frontier"
            # whose detection rate equalled its false-positive rate.
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(int(index))
    return np.array(hull, dtype=int)


def hull_detection_at(
    fpr: np.ndarray, tpr: np.ndarray, hull: np.ndarray, budget: float
) -> tuple[float, tuple[int, int], float]:
    """The best achievable detection at a false-positive budget, and the mixture that reaches it.

    Between two hull vertices the achievable set is the straight line joining them, because a
    coin that picks the left rule with probability ``1 - w`` and the right one with ``w``
    realises exactly that convex combination. So the answer is a linear interpolation, and the
    weight is the thing an operator would have to implement.
    """
    vertices = hull[np.argsort(fpr[hull])]
    hull_fpr, hull_tpr = fpr[vertices], tpr[vertices]
    if budget <= hull_fpr[0]:
        return float(hull_tpr[0]), (int(vertices[0]), int(vertices[0])), 0.0
    if budget >= hull_fpr[-1]:
        return float(hull_tpr[-1]), (int(vertices[-1]), int(vertices[-1])), 0.0
    right = int(np.searchsorted(hull_fpr, budget, side="left"))
    left = max(right - 1, 0)
    span = hull_fpr[right] - hull_fpr[left]
    weight = 0.0 if span <= 0 else float((budget - hull_fpr[left]) / span)
    detection = float(hull_tpr[left] + weight * (hull_tpr[right] - hull_tpr[left]))
    return detection, (int(vertices[left]), int(vertices[right])), weight


def threshold_detection_at(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, budget: float
) -> tuple[float, float, float]:
    """The best a *plain threshold* achieves inside a budget: detection, its FPR, and the cut.

    Deliberately not an interpolation: a threshold can only land on an ROC point, so the honest
    comparison is the highest detection among the points at or under the budget.
    """
    allowed = np.flatnonzero(fpr <= budget + 1e-12)
    if len(allowed) == 0:
        return 0.0, 0.0, float(thresholds[0])
    best = allowed[int(np.argmax(tpr[allowed]))]
    return float(tpr[best]), float(fpr[best]), float(thresholds[best])


# --------------------------------------------------------------------------------------
# Leaving thresholds behind.
# --------------------------------------------------------------------------------------


def net_benefit(y_true: np.ndarray, scores: np.ndarray, threshold_probability: float) -> float:
    """Vickers-Elkin net benefit at one exchange rate between a miss and a false alarm.

    ``NB = TP/n - FP/n * p/(1-p)``, where ``p`` is the probability at which an operator would
    be indifferent between alerting and not. It puts detections and false alarms in one number
    without inventing a currency, which is why clinical work uses it and why it belongs beside
    a fixed-FPR budget that assumes one exchange rate and never states it.
    """
    if not 0.0 < threshold_probability < 1.0:
        return 0.0
    alerts = scores >= threshold_probability
    total = len(y_true)
    if total == 0:
        return 0.0
    true_positive = float(np.sum(alerts & (y_true == 1))) / total
    false_positive = float(np.sum(alerts & (y_true == 0))) / total
    odds = threshold_probability / (1.0 - threshold_probability)
    return true_positive - false_positive * odds


def normalised_cost(fpr: float, tpr: float, skew: float) -> float:
    """Drummond-Holte normalised expected cost at one probability-cost skew.

    ``NEC = (1 - tpr) * skew + fpr * (1 - skew)``: the vertical axis of a cost curve. The skew
    folds prevalence and the cost ratio into one number, which is the point -- an operating
    point is optimal over a *range* of deployments, and a single FPR budget hides which range.
    """
    return (1.0 - tpr) * skew + fpr * (1.0 - skew)


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRow:
    """One false-positive budget, judged against the frontier that budget allows."""

    budget: float
    threshold_detection: float
    hull_detection: float
    realised_fpr: float
    mixing_weight: float
    on_hull: bool

    @property
    def gap(self) -> float:
        """Detection a randomised rule could add, on the split the rule was derived from."""
        return self.hull_detection - self.threshold_detection


@dataclass(frozen=True)
class TransferRow:
    """The randomised rule, derived on validation and applied to the later days."""

    budget: float
    promised_gain: float
    delivered_gain: float
    threshold_detection: float
    randomised_detection: float
    realised_fpr: float
    disagreement: float

    @property
    def survived(self) -> bool:
        """Whether the gain promised on validation showed up at all."""
        return self.delivered_gain > 0.0


@dataclass(frozen=True)
class BenefitRow:
    """Net benefit at one exchange rate, against the two trivial policies."""

    threshold_probability: float
    model: float
    alert_on_everything: float
    alert_on_nothing: float
    model_production: float
    everything_production: float

    @property
    def best(self) -> str:
        """Which policy an operator with these preferences should pick, at the split's rate."""
        return self._winner(self.model, self.alert_on_everything)

    @property
    def best_production(self) -> str:
        """The same question at the production base rate, which is the one that ships."""
        return self._winner(self.model_production, self.everything_production)

    @staticmethod
    def _winner(model: float, everything: float) -> str:
        options = {"the model": model, "alert on everything": everything, "alert on nothing": 0.0}
        return max(options, key=lambda name: options[name])


@dataclass(frozen=True)
class CostRow:
    """One operating point's normalised expected cost across the skew range."""

    label: str
    optimal_from: float
    optimal_to: float
    share_of_range: float


@dataclass
class HullStudy:
    """Everything the report needs, computed once."""

    budgets: list[BudgetRow]
    transfers: list[TransferRow]
    benefits: list[BenefitRow]
    costs: list[CostRow]
    validation_curve: tuple[np.ndarray, np.ndarray]
    hull_curve: tuple[np.ndarray, np.ndarray]
    n_validation: int
    n_test: int
    n_attacks: int
    prevalence: float
    production_prevalence: float
    seconds: float = 0.0

    def dominated(self) -> list[BudgetRow]:
        """The budgets where a plain threshold is not on the frontier."""
        return [row for row in self.budgets if not row.on_hull]

    def survivors(self) -> list[TransferRow]:
        """The budgets where the promised gain actually arrived."""
        return [row for row in self.transfers if row.survived]

    def useful_range(self, production: bool = False) -> tuple[float, float]:
        """The span of exchange rates over which the model beats both trivial policies."""
        winning = [
            row.threshold_probability
            for row in self.benefits
            if (row.best_production if production else row.best) == "the model"
        ]
        return (min(winning), max(winning)) if winning else (0.0, 0.0)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _randomised_scores(
    scores: np.ndarray,
    cut_low: float,
    cut_high: float,
    weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the mixture: use the strict cut with probability ``1 - weight``, the loose one else.

    The coin is per flow, which is what makes the rule achieve the interpolated operating point
    -- and also what makes it return different verdicts for the same flow on two occasions.
    Both halves of that are the point.
    """
    coin = rng.random(len(scores)) < weight
    return np.where(coin, scores >= cut_high, scores >= cut_low)


def run_hull_study(settings: Settings) -> HullStudy:
    """Ask whether the deployed cut is on the frontier, and whether the frontier is real."""
    start = time.perf_counter()
    cfg: HullConfig = settings.hull
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    column = list(model.classes_).index(1)
    val_scores = np.asarray(model.predict_proba(x_val))[:, column]
    test_scores = np.asarray(model.predict_proba(x_test))[:, column]

    fpr, tpr, thresholds = roc_points(y_val, val_scores)
    hull = upper_hull(fpr, tpr)

    budgets: list[BudgetRow] = []
    transfers: list[TransferRow] = []
    for budget in cfg.budgets:
        plain, realised, plain_cut = threshold_detection_at(fpr, tpr, thresholds, budget)
        best, (left, right), weight = hull_detection_at(fpr, tpr, hull, budget)
        budgets.append(
            BudgetRow(
                budget=budget,
                threshold_detection=plain,
                hull_detection=best,
                realised_fpr=realised,
                mixing_weight=weight,
                on_hull=bool(best - plain <= cfg.tolerance),
            )
        )

        # The transfer test: take the *rule* the validation hull prescribes -- two cuts and a
        # coin -- and run it on the later days, beside the plain threshold chosen the same way.
        cut_low = float(thresholds[left])
        cut_high = float(thresholds[right])
        best_index = int(np.flatnonzero(fpr <= budget + 1e-12)[-1]) if np.any(fpr <= budget) else 0
        plain_cut = (
            float(
                thresholds[
                    np.flatnonzero(fpr <= budget + 1e-12)[
                        int(np.argmax(tpr[np.flatnonzero(fpr <= budget + 1e-12)]))
                    ]
                ]
            )
            if np.any(fpr <= budget)
            else float(thresholds[best_index])
        )
        attacks = y_test == 1
        benign = y_test == 0
        plain_alerts = test_scores >= plain_cut
        mixed_alerts = _randomised_scores(test_scores, cut_low, cut_high, weight, rng)
        plain_detection = float(np.mean(plain_alerts[attacks])) if attacks.any() else 0.0
        mixed_detection = float(np.mean(mixed_alerts[attacks])) if attacks.any() else 0.0
        # How often the two rules disagree on the *same* flow: the price of the coin, in the
        # currency the metamorphic study measures determinism in.
        repeat = _randomised_scores(test_scores, cut_low, cut_high, weight, rng)
        transfers.append(
            TransferRow(
                budget=budget,
                promised_gain=best - plain,
                delivered_gain=mixed_detection - plain_detection,
                threshold_detection=plain_detection,
                randomised_detection=mixed_detection,
                realised_fpr=float(np.mean(mixed_alerts[benign])) if benign.any() else 0.0,
                disagreement=float(np.mean(mixed_alerts != repeat)),
            )
        )

    # The same question at two prevalences, because net benefit depends on the base rate and
    # the split's 25% is a property of how the data was captured rather than of a deployment.
    from netsentry.evaluation.ope import resample_to_prevalence

    keep = resample_to_prevalence(y_test, settings.cost.production_attack_rate, rng=rng)
    y_production, scores_production = y_test[keep], test_scores[keep]
    benefits = [
        BenefitRow(
            threshold_probability=probability,
            model=net_benefit(y_test, test_scores, probability),
            alert_on_everything=net_benefit(y_test, np.ones_like(test_scores), probability),
            alert_on_nothing=0.0,
            model_production=net_benefit(y_production, scores_production, probability),
            everything_production=net_benefit(
                y_production, np.ones_like(scores_production), probability
            ),
        )
        for probability in cfg.threshold_probabilities
    ]

    # Cost curves: for each candidate operating point, the range of skews where it is cheapest.
    skews = np.linspace(cfg.skew_min, cfg.skew_max, cfg.skew_points)
    candidates: list[tuple[str, float, float]] = []
    for budget in cfg.budgets:
        plain, realised, _ = threshold_detection_at(fpr, tpr, thresholds, budget)
        candidates.append((f"the {budget:.1%} budget", realised, plain))
    candidates.append(("alert on everything", 1.0, 1.0))
    candidates.append(("alert on nothing", 0.0, 0.0))
    matrix = np.array([[normalised_cost(f, t, skew) for skew in skews] for _, f, t in candidates])
    winners = np.argmin(matrix, axis=0)
    costs = []
    for index, (label, _, _) in enumerate(candidates):
        owned = skews[winners == index]
        costs.append(
            CostRow(
                label=label,
                optimal_from=float(owned.min()) if len(owned) else 0.0,
                optimal_to=float(owned.max()) if len(owned) else 0.0,
                share_of_range=float(len(owned) / len(skews)),
            )
        )

    study = HullStudy(
        budgets=budgets,
        transfers=transfers,
        benefits=benefits,
        costs=costs,
        validation_curve=(fpr, tpr),
        hull_curve=(fpr[hull][np.argsort(fpr[hull])], tpr[hull][np.argsort(fpr[hull])]),
        n_validation=len(y_val),
        n_test=len(y_test),
        n_attacks=int(np.sum(y_test == 1)),
        prevalence=float(np.mean(y_test)),
        production_prevalence=settings.cost.production_attack_rate,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Hull study complete",
        extra={
            "dominated": len(study.dominated()),
            "survivors": len(study.survivors()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _budget_table(study: HullStudy) -> str:
    rows = "\n".join(
        f"| {row.budget:.1%} | {row.threshold_detection:.1%} | {row.hull_detection:.1%} | "
        f"**{row.gap * 100:+.2f} pts** | {row.mixing_weight:.2f} | "
        + ("on the frontier" if row.on_hull else "**dominated**")
        + " |"
        for row in study.budgets
    )
    return (
        "| false-positive budget | best plain threshold | best achievable (hull) | free "
        "detection | coin weight | verdict |\n|---|---|---|---|---|---|\n" + rows
    )


def _transfer_table(study: HullStudy) -> str:
    rows = "\n".join(
        f"| {row.budget:.1%} | {row.promised_gain * 100:+.2f} pts | "
        f"**{row.delivered_gain * 100:+.2f} pts** | {row.threshold_detection:.1%} | "
        f"{row.randomised_detection:.1%} | {row.realised_fpr:.2%} | {row.disagreement:.2%} |"
        for row in study.transfers
    )
    return (
        "| budget | promised on validation | delivered on the later days | plain threshold | "
        "randomised rule | realised FPR | verdicts that are a coin flip |\n"
        "|---|---|---|---|---|---|---|\n" + rows
    )


def _benefit_table(study: HullStudy) -> str:
    rows = "\n".join(
        f"| {row.threshold_probability:.0%} | {row.model:+.4f} | "
        f"{row.alert_on_everything:+.4f} | {row.best} | {row.model_production:+.4f} | "
        f"{row.everything_production:+.4f} | **{row.best_production}** |"
        for row in study.benefits
    )
    return (
        "| indifference probability | model | alert on everything | winner | model @ "
        "production rate | everything @ production rate | winner |\n"
        "|---|---|---|---|---|---|---|\n" + rows
    )


def _cost_table(study: HullStudy) -> str:
    rows = "\n".join(
        f"| {row.label} | {row.optimal_from:.2f} - {row.optimal_to:.2f} | "
        f"{row.share_of_range:.1%} |"
        for row in study.costs
        if row.share_of_range > 0.0
    )
    return "| operating point | optimal over skew | share of the range |\n|---|---|---|\n" + rows


def _lead(study: HullStudy) -> str:
    dominated = study.dominated()
    survivors = study.survivors()
    best = max(study.transfers, key=lambda row: row.delivered_gain) if study.transfers else None
    tight = study.budgets[0] if study.budgets else None
    return (
        f"**Every deployed operating point is dominated, and almost none of the gain is real.**"
        f"\n\n"
        f"On validation -- where the thresholds are chosen -- {len(dominated)} of "
        f"{len(study.budgets)} budgets sit strictly *below* the ROC convex hull. At the tightest "
        f"one the gap is {tight.gap * 100 if tight else 0:+.2f} points of detection: a coin that "
        f"picks between two thresholds with weight {tight.mixing_weight if tight else 0:.2f} "
        f"achieves a higher detection rate at the identical false-positive rate than any single "
        f"cut can. Free detection, for a coin.\n\n"
        f"Then the rule is carried to the later days, and **{len(survivors)} of "
        f"{len(study.transfers)} gains survive**. "
        + (
            f"The one that does is the tightest budget, where it delivers "
            f"{best.delivered_gain * 100:+.2f} points against a promised "
            f"{best.promised_gain * 100:+.2f} -- detection at the "
            f"{best.budget:.1%} budget going {best.threshold_detection:.1%} to "
            f"{best.randomised_detection:.1%}. "
            if best and best.delivered_gain > 0
            else ""
        )
        + "Everywhere else the promised gain is a wobble in a finite ROC curve, and the "
        "randomised rule delivers nothing or slightly less than the threshold it replaced.\n\n"
        "And the coin is not free. "
        + (
            f"At the budget where the gain is real, {best.disagreement:.2%} of flows get a "
            f"different verdict from one run to the next -- roughly "
            f"{best.disagreement * study.n_test:.0f} of {study.n_test:,}. "
            if best
            else ""
        )
        + "That is not a metaphor for instability; it is the exact property the "
        "[metamorphic study](metamorphic.md) tests as a determinism relation and the "
        "[load-time canary](state_machine.md) checks by replaying fixed flows. **The dominance "
        "result is real and the thing it sells is an invariant two other parts of this system "
        "are built to enforce.**"
    )


def _render(study: HullStudy, frontier: Path, benefit: Path) -> str:
    best = max(study.transfers, key=lambda row: row.delivered_gain) if study.transfers else None
    low, high = study.useful_range()
    plow, phigh = study.useful_range(production=True)
    owner = max(study.costs, key=lambda row: row.share_of_range) if study.costs else None
    tightest = study.costs[0] if study.costs else None
    return f"""# NetSentry — Is the Operating Point on the Frontier?

_The deployed false-positive budgets against the ROC convex hull of the same scores, derived on
{study.n_validation:,} validation flows and carried to {study.n_test:,} later-day flows at a
{study.prevalence:.0%} attack rate. Regenerate with `netsentry hull`._

## Why this report exists

Every decision this project ships is a **threshold**: a score cut chosen on validation at a
false-positive budget, applied to everything after. That construction assumes something nobody
here had checked -- that a threshold is the best rule available at its own false-positive rate.

It need not be. The achievable operating points of a scoring classifier are the **convex hull**
of its ROC points, not the curve itself (Provost & Fawcett 2001). Wherever the curve dips below
its own hull, a randomised rule -- flip a biased coin, use one of two thresholds -- strictly
dominates every plain threshold at that false-positive rate.

{_lead(study)}

## The frontier, on the split where the threshold is chosen

![The deployed cuts against their own convex hull](../figures/{frontier.name})

{_budget_table(study)}

The coin-weight column is the thing an operator would have to implement: use the strict cut with
probability `1 - w` and the loose one otherwise, per flow. The interpolation is exact -- between
two hull vertices, the achievable set *is* the straight line joining them -- so the middle column
is not an estimate but the best any rule of this class can do.

Building this found its own bug worth recording: the first hull popped on the wrong turn and
returned the **lower** hull, which is the diagonal. The giveaway was a "frontier" whose detection
rate equalled its false-positive rate exactly, at every budget, which is what a coin with no
model achieves. A frontier that agrees with chance is not a subtle error.

## Does the gain survive the later days?

{_transfer_table(study)}

This is the question the study exists for. A dip below the hull can be real structure in the
score distribution or a wobble in a finite sample, and the two look identical until the rule
derived from one split is applied to another.

{
    f"At the {best.budget:.1%} budget it is real: {best.delivered_gain * 100:+.2f} points "
    f"delivered against {best.promised_gain * 100:+.2f} promised."
    if best and best.delivered_gain > 0
    else "None of it survives."
}
That is the budget where the ROC is most jagged -- fewest positives above the cut, so the
largest genuine steps between adjacent operating points. At the looser budgets the promised
gains are hundredths of a point and arrive as nothing or slightly negative, which is what an
overfitted frontier looks like when it is asked to keep a promise.

The last column is the price. A randomised rule returns a different answer for the same flow on
a re-run, and the rate is not negligible at the budget where the gain is. Two other parts of
this system exist to prevent exactly that: the [metamorphic oracle](metamorphic.md) tests
determinism as a label-free correctness relation, and the [load-time canary](state_machine.md)
replays fixed flows and flips `/health` to degraded on a mismatch. **A randomised operating
point would fail both**, and it would fail them by design rather than by accident, which is a
much harder conversation to have with an auditor than a hundredth of a point is worth.

## Without a threshold at all: net benefit

![Net benefit against the trivial policies](../figures/{benefit.name})

{_benefit_table(study)}

Net benefit (Vickers & Elkin 2006) is the question clinical work always asks and machine
learning almost never does: at an operator's own exchange rate between a miss and a false alarm,
is the model better than *alerting on everything* or *alerting on nothing*? It needs no
currency, only the indifference probability.

The two halves of the table are the same question at two base rates, and they disagree. At the
split's own {study.prevalence:.0%} attack rate the model wins only for indifference
probabilities between {low:.0%} and {high:.0%} -- below that, alerting on everything is better,
because one flow in four really is an attack. At the {study.production_prevalence:.0%}
production rate the model wins from {plow:.0%} to {phigh:.0%}, and alerting on everything is
never sensible.

That is the [base-rate study](base_rate.md) arriving from a different direction, and it is the
reason a fixed-FPR budget should never be quoted without the prevalence it assumes.

## Without a threshold at all: cost curves

{_cost_table(study)}

A cost curve (Drummond & Holte 2006) plots normalised expected cost against the
**probability-cost skew** -- prevalence and the cost ratio folded into one number -- so an
operating point owns a *range* of deployments rather than a point.

{
    f"The {owner.label} owns {owner.share_of_range:.0%} of the range."
    if owner
    else "No operating point dominates."
}
{
    f"The {tightest.label} -- the tightest one this project ships -- owns "
    f"{tightest.share_of_range:.0%}."
    if tightest
    else ""
}
That is the honest reading of a fixed-FPR budget: it is optimal for a particular exchange rate
between a miss and a false alarm, and a deployment whose economics sit elsewhere on this axis
should be running a different cut. The [cost study](cost.md) picks one such point by assuming a
price; this says which range that price corresponds to.

## Scope and honest limits

- **The hull is computed on the same scores the threshold is chosen from**, which is the
  correct construction and also why the transfer test is the load-bearing part. Anything else
  would be reporting a fit as a result.
- **Randomisation here is per flow.** A rule that randomised per *host* or per *day* would keep
  determinism within a flow and reach a different point; that is a genuinely different design
  and this study does not price it.
- **Net benefit assumes the score is a probability.** The deployed scores are calibrated
  (isotonic, on validation) which is what makes the axis meaningful; on raw tree outputs the
  same curve would be a statement about the calibrator.
- **Cost curves fold two unknowns into one.** A skew is a prevalence and a cost ratio
  multiplied together, so the table says which *combinations* each cut owns and cannot separate
  a rare-and-cheap deployment from a common-and-expensive one.
- **Everything here is about one model's scores.** A dominated operating point is a statement
  about the ranking, not about the model; the [multiplicity study](multiplicity.md) covers the
  case where a different equally-good model would rank differently."""


def run_hull_report(settings: Settings) -> Path:
    """Run the frontier audit and write the report + figures."""
    study = run_hull_study(settings)
    fpr, tpr = study.validation_curve
    hull_fpr, hull_tpr = study.hull_curve
    frontier = plots.plot_lines(
        {
            "the ROC curve (what a threshold can reach)": (fpr, tpr),
            "its convex hull (what a coin can reach)": (hull_fpr, hull_tpr),
        },
        xlabel="false-positive rate",
        ylabel="detection rate",
        title="Where the curve dips below its own hull, a coin does better",
        out_path=settings.paths.figures_dir / HULL_FIGURE,
        xscale="log",
    )
    probabilities = np.array([row.threshold_probability for row in study.benefits])
    benefit = plots.plot_lines(
        {
            "the model": (probabilities, np.array([row.model for row in study.benefits])),
            "alert on everything": (
                probabilities,
                np.array([row.alert_on_everything for row in study.benefits]),
            ),
            "the model, at the production base rate": (
                probabilities,
                np.array([row.model_production for row in study.benefits]),
            ),
            "alert on nothing": (probabilities, np.zeros(len(probabilities))),
        },
        xlabel="indifference probability between a miss and a false alarm",
        ylabel="net benefit",
        title="Is the model better than the two policies that need no model?",
        out_path=settings.paths.figures_dir / BENEFIT_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, frontier, benefit), encoding="utf-8")
    logger.info("Wrote hull report", extra={"path": str(out_path)})

    with track_run(settings, "hull") as run:
        run.log_params({"budgets": str([row.budget for row in study.budgets])})
        run.log_metrics(
            {
                "dominated_budgets": float(len(study.dominated())),
                "surviving_gains": float(len(study.survivors())),
                "best_delivered_gain": max(
                    (row.delivered_gain for row in study.transfers), default=0.0
                ),
            }
        )
        for artifact in (frontier, benefit, out_path):
            run.log_artifact(artifact)
    return out_path
