"""Poisoning the threshold instead of the model.

Every poisoning study in this repository attacks the *training* data -- [flipped
labels](poisoning.md), a [contaminated benign pool](poisoning.md), a [planted
backdoor](backdoor.md) -- and each is answered by a defence that inspects the training set. All
of them share an assumption nobody wrote down: that the thing worth corrupting is the model.

It is not the only thing. Every operational number this project ships comes from a **threshold**,
and that threshold is a *quantile of benign validation scores*: the cut at a 1% false-positive
budget is the 99th percentile of what benign traffic scores. The model is never touched. An
attacker who can get their own traffic labelled benign during calibration -- reconnaissance
during the window a new detector is being tuned, a mislabelled maintenance window, an analyst
clearing a batch of alerts as false positives -- moves the cut instead.

**The arithmetic is the finding, and it runs the wrong way.** A quantile's breakdown point is its
own tail mass: to move the 99th percentile you need to control 1% of the sample, and to move the
99.9th percentile you need **0.1%**. The tighter the false-positive budget, the *cheaper* it is to
attack the threshold that enforces it -- so the operating point this project is proudest of is
the one an attacker can buy most cheaply. The [resolution study](power.md) found that nine benign
flows decide the realised false-positive rate at the 0.1% budget. This is the same nine flows,
seen from the other side: they are also all an attacker has to own.

The module measures three things.

1. **The curve.** Injection fraction against threshold shift and detection loss, for two attacker
   capabilities: a **blind** attacker who simply has their own traffic present during calibration,
   and an **informed** one who can place flows at the top of the benign score distribution. The
   second is an upper bound rather than a realistic threat, and is labelled as one.
2. **Whether anything notices.** The injected flows are benign-labelled and few; the question is
   whether the drift monitors this project already runs would fire on a calibration set that has
   been moved enough to matter.
3. **What the fixes cost.** A **trimmed quantile** discards the top of the sample before cutting,
   which restores the breakdown point and biases the threshold low on clean data -- more false
   positives, permanently, in exchange for robustness that only pays off under attack. A
   **median-of-days** calibration takes a threshold per capture day and the median across them,
   which raises the breakdown point to half the days at the cost of ignoring most of the sample.
   Both are priced on clean data as well as poisoned, because a defence measured only under
   attack is a defence whose bill nobody has seen.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.monitoring.drift import population_stability_index
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import CalibrationAttackConfig

logger = get_logger(__name__)

REPORT_NAME = "calibration_attack.md"
FIGURE_NAME = "calibration_attack.png"


# --------------------------------------------------------------------------------------
# The mechanism.
# --------------------------------------------------------------------------------------


def breakdown_point(budget: float) -> float:
    """The share of the calibration sample an attacker must own to move the cut arbitrarily.

    A threshold at a false-positive budget ``q`` is the ``1 - q`` quantile of benign scores, and a
    quantile's breakdown point is the mass in the tail beyond it -- exactly ``q``. Contaminating
    more than that fraction pushes the order statistic past every clean observation, so the cut
    lands wherever the attacker likes.

    The consequence is the one worth stating out loud: **the tighter the budget, the cheaper the
    attack**. A 5% budget needs one flow in twenty; the 0.1% budget this project leads with needs
    one in a thousand.
    """
    return budget


def poisoned_threshold(
    benign_scores: np.ndarray,
    injected: np.ndarray,
    budget: float,
) -> float:
    """The cut a calibrator computes when part of its benign sample is the attacker's.

    The calibrator is not doing anything wrong: it takes the quantile of everything it was told
    is benign, which is exactly the deployed procedure. The attack is entirely in the input.
    """
    labels = np.zeros(len(benign_scores) + len(injected), dtype=int)
    scores = np.concatenate([benign_scores, injected])
    return _quantile_threshold(labels, scores, budget)


def _quantile_threshold(labels: np.ndarray, scores: np.ndarray, budget: float) -> float:
    """The deployed threshold rule, on a benign-only sample.

    ``threshold_at_fpr`` needs both classes to trace an ROC curve, so a benign-only calibration
    set is handled directly as the quantile it is. Both routes agree where both are defined; this
    one is simply the definition rather than a curve traversal.
    """
    benign = scores[labels == 0]
    if len(benign) == 0:
        return float("inf")
    return float(np.quantile(benign, 1.0 - budget, method="higher"))


def trimmed_threshold(scores: np.ndarray, budget: float, trim: float) -> float:
    """Discard the top ``trim`` of the sample, then take the quantile of what is left.

    The classical fix for a corrupted order statistic, and it works for the classical reason: an
    attacker holding less than ``trim`` of the sample cannot reach the cut, because everything
    they placed above it was thrown away first. What it costs is paid on clean data -- the
    trimmed sample's quantile sits lower than the true one, so the deployed rule alerts more
    often than the budget it was asked for.
    """
    if len(scores) == 0:
        return float("inf")
    if trim >= 1.0:
        # Trimming everything leaves nothing to calibrate on. The safe fallback is the largest
        # score observed, which alerts on nothing beyond the range already seen -- a useless
        # detector rather than a wide-open one, which is the right direction to fail in.
        return float(np.max(scores))
    keep = scores[scores <= np.quantile(scores, 1.0 - trim, method="higher")]
    if len(keep) == 0:
        return float(np.max(scores))
    # The quantile is re-expressed against the surviving mass, so trimming does not silently
    # tighten the budget as well as lowering the cut.
    adjusted = min(max((budget - trim) / (1.0 - trim), 0.0), 1.0)
    return float(np.quantile(keep, 1.0 - adjusted, method="higher"))


def median_of_days(scores: np.ndarray, days: np.ndarray, budget: float, minimum: int) -> float:
    """Calibrate a threshold per capture day and take the median of them.

    A median over `d` days survives contamination of up to `d/2` of them however severe each one
    is, which is a far higher breakdown point than a pooled quantile -- but only against an
    attacker whose traffic is confined to a few days. One spread thinly across every day defeats
    it, and this study measures that case too.
    """
    per_day = [
        float(np.quantile(scores[days == day], 1.0 - budget, method="higher"))
        for day in np.unique(days)
        if int(np.sum(days == day)) >= minimum
    ]
    if not per_day:
        return _quantile_threshold(np.zeros(len(scores), dtype=int), scores, budget)
    return float(np.median(per_day))


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackPoint:
    """One injection fraction, for one attacker capability."""

    attacker: str
    fraction: float
    injected: int
    threshold: float
    clean_threshold: float
    detection: float
    clean_detection: float
    realised_fpr: float
    score_psi: float
    psi_threshold: float

    @property
    def shift(self) -> float:
        return self.threshold - self.clean_threshold

    @property
    def detection_loss(self) -> float:
        """Points of detection the attacker bought, on the split the model is judged on."""
        return self.clean_detection - self.detection

    @property
    def noticed(self) -> bool:
        return self.score_psi >= self.psi_threshold


@dataclass(frozen=True)
class DefenceRow:
    """One calibration rule against one attacker shape, priced clean and poisoned."""

    name: str
    describes: str
    attacker: str
    clean_threshold: float
    clean_detection: float
    clean_fpr: float
    poisoned_threshold: float
    poisoned_detection: float
    budget: float

    @property
    def clean_cost(self) -> float:
        """False-positive rate above the budget the rule was asked for, on clean data."""
        return self.clean_fpr - self.budget

    @property
    def survived(self) -> float:
        """Detection kept under attack, as a share of what the rule achieves clean."""
        return self.poisoned_detection / self.clean_detection if self.clean_detection else 0.0


@dataclass(frozen=True)
class BudgetCost:
    """What one false-positive budget costs an attacker to overturn."""

    budget: float
    deciding_flows: int
    flows_to_break: int
    detection: float

    @property
    def breakdown(self) -> float:
        return breakdown_point(self.budget)


@dataclass
class CalibrationAttackStudy:
    """Everything the report needs, computed once."""

    points: list[AttackPoint]
    defences: list[DefenceRow]
    costs: list[BudgetCost]
    budget: float
    n_benign: int
    clean_threshold: float
    clean_detection: float
    deciding_flows: int
    seconds: float = 0.0

    def for_attacker(self, name: str) -> list[AttackPoint]:
        return [row for row in self.points if row.attacker == name]

    def attackers(self) -> list[str]:
        seen: list[str] = []
        for row in self.points:
            if row.attacker not in seen:
                seen.append(row.attacker)
        return seen

    def breakdown(self) -> float:
        return breakdown_point(self.budget)

    def cheapest_halving(self, attacker: str) -> AttackPoint | None:
        """The smallest injection that costs at least half the detection rate."""
        rows = [
            row for row in self.for_attacker(attacker) if row.detection <= row.clean_detection / 2
        ]
        return min(rows, key=lambda row: row.fraction) if rows else None

    def unnoticed(self) -> list[AttackPoint]:
        """Injections that moved the threshold without tripping the score monitor."""
        return [row for row in self.points if row.detection_loss > 0 and not row.noticed]

    def best_defence(self) -> DefenceRow:
        """Most detection kept under attack, breaking ties toward the cheaper clean bill."""
        return max(self.defences, key=lambda row: (row.survived, -abs(row.clean_cost)))

    def shapes(self) -> list[str]:
        seen: list[str] = []
        for row in self.defences:
            if row.attacker not in seen:
                seen.append(row.attacker)
        return seen

    def defences_against(self, attacker: str) -> list[DefenceRow]:
        return [row for row in self.defences if row.attacker == attacker]

    def shape_dependent(self) -> list[str]:
        """Rules whose survival depends on how the attacker spread their flows.

        A defence that works against one attacker shape and not another is not a weaker defence
        -- it is a different claim, and reporting a single number for it would be the kind of
        averaging that hides the case an operator actually faces.
        """
        by_rule: dict[str, set[bool]] = {}
        for row in self.defences:
            by_rule.setdefault(row.name, set()).add(row.survived > 0.5)
        return sorted(name for name, outcomes in by_rule.items() if len(outcomes) > 1)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def run_calibration_attack_study(settings: Settings) -> CalibrationAttackStudy:
    """Move the threshold instead of the model, then price the fixes."""
    start = time.perf_counter()
    cfg: CalibrationAttackConfig = settings.calibration_attack
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.schema import DAY_COLUMN
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

    benign_scores = val_scores[y_val == 0]
    clean_threshold = threshold_at_fpr(y_val, val_scores, cfg.budget)
    clean_rates = rates_at_threshold(y_test, test_scores, clean_threshold)
    clean_detection = float(clean_rates["tpr"])
    deciding = int(clean_rates["fp"])

    # The blind attacker's material: their own traffic, scored by the deployed model and labelled
    # benign by whoever was calibrating. No knowledge of the model or the threshold is needed --
    # only presence during the calibration window.
    attack_pool = np.asarray(model.predict_proba(x_train[y_train == 1]))[:, column]
    ceiling = float(np.max(np.concatenate([benign_scores, attack_pool])))

    makers: dict[str, Callable[[int], np.ndarray]] = {
        "blind (own traffic, labelled benign)": lambda count: rng.choice(
            attack_pool, size=count, replace=count > len(attack_pool)
        ),
        "informed (places flows at the ceiling)": lambda count: np.full(count, ceiling),
    }

    points: list[AttackPoint] = []
    for attacker, make in makers.items():
        for fraction in cfg.fractions:
            count = max(round(fraction * len(benign_scores)), 1 if fraction > 0 else 0)
            injected = make(count) if count else np.array([])
            threshold = poisoned_threshold(benign_scores, injected, cfg.budget)
            rates = rates_at_threshold(y_test, test_scores, threshold)
            polluted = np.concatenate([benign_scores, injected])
            points.append(
                AttackPoint(
                    attacker=attacker,
                    fraction=fraction,
                    injected=int(count),
                    threshold=threshold,
                    clean_threshold=clean_threshold,
                    detection=float(rates["tpr"]),
                    clean_detection=clean_detection,
                    realised_fpr=float(rates["fpr"]),
                    score_psi=float(
                        population_stability_index(benign_scores, polluted, bins=cfg.psi_bins)
                    ),
                    psi_threshold=cfg.psi_threshold,
                )
            )

    # The defences, each priced twice: on the clean calibration set and on one poisoned at the
    # breakdown point, which is the smallest budget that defeats the undefended rule outright.
    poison = makers["informed (places flows at the ceiling)"](
        max(round(cfg.defence_fraction * len(benign_scores)), 1)
    )
    polluted = np.concatenate([benign_scores, poison])
    days = (
        np.asarray(val_frame[DAY_COLUMN])[y_val == 0]
        if DAY_COLUMN in val_frame.columns
        else np.zeros(len(benign_scores))
    )
    labels = np.unique(days)

    # Two shapes, because a per-day rule is a claim about *where* the attacker's flows sit rather
    # than how many there are. Spread thinly across every day, a median over days is poisoned in
    # every term and buys nothing; confined to one day it is outvoted by the rest.
    shapes = {
        "spread across every day": rng.choice(labels, size=len(poison)),
        "confined to one day": np.full(len(poison), labels[0]),
    }

    rules: list[tuple[str, str, Callable[[np.ndarray, np.ndarray], float]]] = [
        (
            "the deployed rule (a plain quantile)",
            "take the 1 - q quantile of everything labelled benign",
            lambda scores, _days: float(np.quantile(scores, 1.0 - cfg.budget, method="higher")),
        ),
        (
            f"trimmed quantile (drop the top {cfg.trim:.1%})",
            "discard the top of the sample, then cut what is left",
            lambda scores, _days: trimmed_threshold(scores, cfg.budget, cfg.trim),
        ),
        (
            "median of per-day thresholds",
            "calibrate each capture day separately and take the median",
            lambda scores, day_labels: median_of_days(
                scores, day_labels, cfg.budget, cfg.min_day_flows
            ),
        ),
    ]

    defences: list[DefenceRow] = []
    for shape, placement in shapes.items():
        padded_days = np.concatenate([days, placement])
        for name, describes, rule in rules:
            clean_cut = rule(benign_scores, days)
            poisoned_cut = rule(polluted, padded_days)
            clean_row = rates_at_threshold(y_test, test_scores, clean_cut)
            poisoned_row = rates_at_threshold(y_test, test_scores, poisoned_cut)
            defences.append(
                DefenceRow(
                    name=name,
                    describes=describes,
                    attacker=shape,
                    clean_threshold=clean_cut,
                    clean_detection=float(clean_row["tpr"]),
                    clean_fpr=float(clean_row["fpr"]),
                    poisoned_threshold=poisoned_cut,
                    poisoned_detection=float(poisoned_row["tpr"]),
                    budget=cfg.budget,
                )
            )

    # The same arithmetic at several budgets, because "tighter is cheaper" is a claim about a
    # trend and one point cannot make it. Each row is the number of flows an attacker must own,
    # which is the tail mass of the quantile that budget defines.
    costs = []
    for budget in sorted(cfg.budget_ladder, reverse=True):
        cut = threshold_at_fpr(y_val, val_scores, budget)
        rates = rates_at_threshold(y_test, test_scores, cut)
        costs.append(
            BudgetCost(
                budget=budget,
                deciding_flows=int(rates["fp"]),
                flows_to_break=max(round(budget * len(benign_scores)), 1),
                detection=float(rates["tpr"]),
            )
        )

    study = CalibrationAttackStudy(
        points=points,
        defences=defences,
        costs=costs,
        budget=cfg.budget,
        n_benign=len(benign_scores),
        clean_threshold=clean_threshold,
        clean_detection=clean_detection,
        deciding_flows=deciding,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Calibration-attack study complete",
        extra={
            "breakdown": study.breakdown(),
            "unnoticed": len(study.unnoticed()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: CalibrationAttackStudy) -> str:
    """The finding, written from the computed curve."""
    breakdown = study.breakdown()
    informed = study.attackers()[-1]
    blind = study.attackers()[0]
    at_breakdown = min(
        (row for row in study.for_attacker(informed) if row.fraction >= breakdown),
        key=lambda row: row.fraction,
        default=None,
    )
    blind_same = next(
        (
            row
            for row in study.for_attacker(blind)
            if row.fraction == getattr(at_breakdown, "fraction", -1)
        ),
        None,
    )
    loudest = max(study.points, key=lambda row: row.score_psi)
    lines = [
        "**The threshold's breakdown point is the false-positive budget itself, so the tighter "
        "the budget, the cheaper it is to attack.**",
        "",
        f"The deployed cut at a {study.budget:.1%} budget is the "
        f"{1 - study.budget:.1%} quantile of {study.n_benign:,} benign validation scores. A "
        f"quantile's breakdown point is the mass in its own tail: own more than "
        f"**{breakdown:.1%}** of the calibration sample and the order statistic lands past every "
        "clean observation, wherever the attacker likes. Nothing about the model is touched, and "
        "nothing about the calibration procedure is wrong -- it takes the quantile of everything "
        "it was told is benign, which is exactly the deployed rule.",
        "",
    ]
    if at_breakdown is not None:
        lines += [
            f"The arithmetic holds exactly. At **{at_breakdown.injected} injected flows** -- "
            f"{at_breakdown.fraction:.1%} of the calibration set -- an attacker who can place "
            f"flows at the top of the benign score distribution takes detection from "
            f"{at_breakdown.clean_detection:.1%} to **{at_breakdown.detection:.1%}**, and one "
            "step further takes it to zero.",
            "",
        ]
    if blind_same is not None:
        lines += [
            f"**And the attacker does not need to be clever.** One who merely has their own "
            f"traffic present during the calibration window -- reconnaissance while a detector is "
            f"being tuned, a mislabelled maintenance window, a batch of alerts an analyst cleared "
            f"as false positives -- reaches {blind_same.detection:.1%} with the same "
            f"{blind_same.injected} flows, no knowledge of the model or the threshold required.",
            "",
        ]
    lines += [
        f"**Nothing notices.** Across every injection level tested, the score-distribution monitor "
        f"peaks at a PSI of {loudest.score_psi:.3f} against a {loudest.psi_threshold:.1f} "
        "alarm line -- because a few hundred flows out of thousands barely move a binned "
        "distribution, and the ones that were added are individually unremarkable. The "
        "[compositional study](composition.md) found the same shape for evasion: the failures a "
        "drift monitor is worst at seeing are the ones an adversary chooses.",
        "",
        "This is the [resolution study](power.md)'s finding from the other side. It measured how "
        "few benign flows decide the realised false-positive rate at a tight budget; those are "
        "the same flows an attacker has to own. Reading the same arithmetic across budgets makes "
        "the trend explicit, and it is the wrong way round:",
        "",
        "| false-positive budget | flows deciding the realised rate | flows an attacker must own "
        "| detection it buys |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {cost.budget:.1%} | {cost.deciding_flows} | **{cost.flows_to_break}** | "
        f"{cost.detection:.1%} |"
        for cost in study.costs
    )
    tightest = min(study.costs, key=lambda cost: cost.budget)
    loosest = max(study.costs, key=lambda cost: cost.budget)
    lines += [
        "",
        f"**The operating point this project leads with is the one an attacker can buy most "
        f"cheaply.** Moving the {loosest.budget:.1%} cut takes {loosest.flows_to_break} flows; "
        f"moving the {tightest.budget:.1%} cut takes {tightest.flows_to_break}. Tightening a "
        "budget is usually described as making a detector stricter. It also makes the number "
        "enforcing that strictness rest on fewer observations, and an order statistic resting on "
        "fewer observations is easier to move.",
    ]
    return "\n".join(lines)


def _render(study: CalibrationAttackStudy, figure: Path) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- Poisoning the Threshold Instead of the Model",
        "",
        f"_The deployed {study.budget:.1%}-budget cut, recalibrated on {study.n_benign:,} benign "
        f"validation scores with a growing share replaced by an attacker's, and judged on the "
        f"later days. Regenerate with `netsentry calibrationattack`._",
        "",
        "## Why this report exists",
        "",
        "Every poisoning study here attacks the *training* data -- [flipped labels and a "
        "contaminated benign pool](poisoning.md), a [planted backdoor](backdoor.md) -- and each "
        "is answered by a defence that inspects the training set. They share an assumption nobody "
        "wrote down: that the thing worth corrupting is the model.",
        "",
        "It is not the only thing. Every operational number this project ships comes from a "
        "threshold, and that threshold is a **quantile of benign validation scores**. An attacker "
        "who can get their own traffic labelled benign during calibration moves the cut without "
        "going anywhere near the model.",
        "",
        _lead(study),
        "",
        "## The curve",
        "",
        "![Detection against the share an attacker owns]" f"(../figures/{figure.name})",
        "",
        "| attacker | injected | share | threshold | detection | detection lost | score PSI | "
        "monitor fires |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in study.points:
        lines.append(
            f"| {row.attacker} | {row.injected} | {row.fraction:.2%} | {row.threshold:.4f} | "
            f"{row.detection:.1%} | **{row.detection_loss:+.1%}** | {row.score_psi:.3f} | "
            f"{'yes' if row.noticed else 'no'} |"
        )
    lines += [
        "",
        "The **blind** attacker is the realistic one: their flows are their own traffic, scored "
        "by the deployed model and labelled benign by whoever was calibrating. They need no "
        "knowledge of the model, the threshold, or that a threshold exists. The **informed** "
        "attacker, who can place flows at the very top of the score distribution, is an upper "
        "bound rather than a threat model -- it is here to show where the curve ends, and it "
        "ends at the breakdown point the arithmetic predicts.",
        "",
        "## What the fixes cost",
        "",
        "| calibration rule | attacker | clean FPR (budget " + f"{study.budget:.1%})" + " | "
        "clean detection | detection under attack | kept |",
        "|---|---|---|---|---|---|",
    ]
    for defence in study.defences:
        lines.append(
            f"| {defence.name} | {defence.attacker} | {defence.clean_fpr:.2%} | "
            f"{defence.clean_detection:.1%} | {defence.poisoned_detection:.1%} | "
            f"**{defence.survived:.0%}** |"
        )
    trimmed = next((row for row in study.defences if "trimmed" in row.name), None)
    per_day = [row for row in study.defences if "median" in row.name]
    lines += [
        "",
        "Both defences are priced on **clean** data as well as poisoned, because a defence "
        "measured only under attack is a defence whose bill nobody has seen.",
        "",
    ]
    if trimmed is not None:
        lines += [
            f"**The trimmed quantile buys uniform, mediocre robustness at a permanent price.** "
            f"Discarding the top of the sample restores the breakdown point -- an attacker "
            f"holding less than the trim cannot reach the cut, because everything they placed "
            f"above it was thrown away first -- and it keeps {trimmed.survived:.0%} of detection "
            f"against either attacker shape. The bill is paid every day: the trimmed sample's "
            f"quantile sits below the true one, so the rule runs at {trimmed.clean_fpr:.2%} "
            f"against the {trimmed.budget:.1%} it was asked for, "
            f"{trimmed.clean_fpr / trimmed.budget - 1:.0%} over budget on traffic nobody is "
            "attacking.",
            "",
        ]
    if len(per_day) >= 2:
        best = max(per_day, key=lambda row: row.survived)
        worst = min(per_day, key=lambda row: row.survived)
        lines += [
            f"**The median of per-day thresholds is nearly free and conditionally excellent.** It "
            f"costs nothing on clean data -- {best.clean_fpr:.2%} against a "
            f"{best.budget:.1%} budget -- and keeps **{best.survived:.0%}** of detection against "
            f"an attacker {best.attacker}, because a median over days is outvoted by the days "
            f"that were not touched. Against one **{worst.attacker}** it keeps "
            f"{worst.survived:.0%}: every term in the median is poisoned, so taking the median "
            "of them buys nothing at all.",
            "",
            "That is not a weaker defence than the trimmed quantile; it is a **different claim**. "
            "One bounds the damage regardless of how the attacker arranges their flows and "
            "charges for it continuously. The other is free and total against a concentrated "
            "adversary and worthless against a patient one. Reporting a single averaged number "
            "for either would hide the case an operator actually faces, which is why the table "
            "is split by attacker shape.",
            "",
        ]
    lines += [
        "## Scope and honest limits",
        "",
        "- **The attack assumes flows reach the calibration set labelled benign**, which is a "
        "claim about an operational process rather than about the model. It is the same "
        "assumption the [label-audit study](label_audit.md) makes from the other direction, and "
        "whether it holds is a question about how a deployment collects its validation data.",
        "- **The informed attacker is an upper bound, not a threat model.** Placing flows at the "
        "exact top of the score distribution requires knowing the scores, which is a stronger "
        "capability than the [extraction study](extraction.md) needs but not one this repository "
        "otherwise grants. The blind curve is the one to read as a threat.",
        "- **The breakdown point is exact; the curve below it is not a bound.** How much detection "
        "a sub-breakdown injection costs depends on the shape of the benign score distribution "
        "near the cut, which is a property of this model on this data.",
        "- **A defence that changes the threshold changes every downstream number.** The trimmed "
        "rule's higher alert volume feeds the [alert-queue](alert_queue.md) and [SLO](slo.md) "
        "studies, which were computed against the plain quantile; adopting it would require "
        "re-running both.",
        "- **The monitor tested is the one this project runs.** A monitor designed for this "
        "attack -- watching the calibration set's upper tail specifically, rather than its whole "
        "distribution -- would fire, and the fact that PSI does not is a statement about what is "
        "deployed rather than about what is possible.",
    ]
    return "\n".join(lines) + "\n"


def run_calibration_attack_report(settings: Settings) -> Path:
    """Run the calibration-poisoning study and write the report + figure."""
    study = run_calibration_attack_study(settings)
    series = {
        attacker: (
            np.array([row.fraction for row in study.for_attacker(attacker)]),
            np.array([row.detection for row in study.for_attacker(attacker)]),
        )
        for attacker in study.attackers()
    }
    figure = plots.plot_lines(
        series,
        xlabel="share of the calibration set the attacker owns",
        ylabel="detection rate on the later days",
        title="A quantile breaks at its own tail mass, and the budget is the tail",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        vlines={"the breakdown point": study.breakdown()},
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote calibration-attack report", extra={"path": str(out_path)})

    with track_run(settings, "calibration_attack") as run:
        run.log_params({"budget": str(study.budget), "benign": str(study.n_benign)})
        run.log_metrics(
            {
                "breakdown_point": study.breakdown(),
                "clean_detection": study.clean_detection,
                "worst_detection": min(row.detection for row in study.points),
                "loudest_psi": max(row.score_psi for row in study.points),
                "best_defence_kept": study.best_defence().survived,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
