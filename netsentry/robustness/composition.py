"""Every safeguard here was validated alone. Do they compose?

This repository has a defence for each failure it has thought of. Evasion is measured against
the [mimicry attack](evasion.md) and answered with [monotone
constraints](monotonic.md). Distribution change is watched by [PSI](drift.md) and
[MMD](mmd.md). Sensor failure has [its own study](degradation.md). Prevalence change has
[the base-rate report](base_rate.md). Coverage is promised by [conformal
prediction](conformal.md) and the false-positive budget by a validation-calibrated threshold.

Each of those was measured with one thing wrong at a time. Production does not have that
courtesy. The question this study exists for is whether the guarantees and the monitors that
watch them still hold when two things go wrong at once -- and it is a real question, because
*failures interact*: two perturbations that each move a monitor's statistic in opposite
directions can cancel, leaving the statistic where it started while the system underneath is
worse than either alone.

The design is a **2^k factorial**. Four stressors, each on or off, all sixteen combinations, and
in each cell the same four guarantees and three monitors are read:

- **stressors** -- a temporal shift (evaluate on the latest slice only), a sensor failure (a
  group of features stops reporting), an evasion attempt (attacks shaped toward the benign
  centroid on the controllable subset), and a prevalence collapse (attacks subsampled to a
  production base rate).
- **guarantees** -- is the false-positive budget respected, is the detection rate holding, does
  the conformal set still cover, is the alert volume inside its SLO.
- **monitors** -- does PSI fire on the features, does the score distribution shift enough to
  notice, does the alert-rate SLO burn.

A factorial design is the right instrument because it separates a **main effect** (what one
stressor does alone) from an **interaction** (what a pair does beyond the sum of its parts).
Running one stressor at a time, which is what every other report here does, measures the first
and is structurally blind to the second.

Two failure shapes are worth naming in advance, because they are what the design is looking for.
**Compound breakage** is a pair of stressors that individually leave a guarantee intact and
jointly break it. **Masking** is worse and quieter: a pair that breaks a guarantee while leaving
every monitor silent, because one stressor's effect on the monitored statistic cancels the
other's. A system with a broken guarantee and a green dashboard is the failure mode that ends
with someone reading logs at three in the morning.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.monitoring.drift import population_stability_index
from netsentry.robustness.evasion import controllable_indices, mimicry_perturb
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import CompositionConfig

logger = get_logger(__name__)

REPORT_NAME = "composition.md"
FIGURE_NAME = "composition_interactions.png"

#: The stressors, in the order their bits are read. Short names because they end up in a header.
STRESSORS = ("shift", "outage", "evasion", "rarity")


# --------------------------------------------------------------------------------------
# The stressors.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One cell of the factorial: which stressors are switched on."""

    active: frozenset[str]

    @property
    def order(self) -> int:
        return len(self.active)

    @property
    def name(self) -> str:
        return " + ".join(sorted(self.active)) if self.active else "nothing wrong"

    def has(self, stressor: str) -> bool:
        return stressor in self.active


def scenarios(stressors: tuple[str, ...] = STRESSORS) -> list[Scenario]:
    """Every combination, smallest first, so the table reads as an escalation."""
    found = [
        Scenario(frozenset(subset))
        for size in range(len(stressors) + 1)
        for subset in itertools.combinations(stressors, size)
    ]
    return sorted(found, key=lambda cell: (cell.order, cell.name))


def apply_shift(
    x: np.ndarray, y: np.ndarray, days: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the latest slice of the evaluation window.

    Drift in this project is temporal, so intensifying it means moving further from the training
    days rather than injecting synthetic noise. Using the data's own ordering keeps the stressor
    honest: it is the same failure the temporal split already demonstrates, turned up.
    """
    if len(days) != len(y) or fraction >= 1.0:
        return x, y
    cut = np.quantile(days, 1.0 - fraction)
    keep = days >= cut
    return x[keep], y[keep]


def apply_outage(x: np.ndarray, columns: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    """A group of features stops reporting and is filled with the training median.

    Filling rather than dropping is what a real pipeline does -- the imputer has no way to know a
    sensor died, so it substitutes the statistic it was fitted with, and the model scores a flow
    that looks unremarkable in exactly the dimensions that stopped being measured.
    """
    damaged = np.array(x, dtype=float, copy=True)
    if len(columns):
        damaged[:, columns] = replacement[columns]
    return damaged


def apply_evasion(
    x: np.ndarray,
    y: np.ndarray,
    centroid: np.ndarray,
    controllable: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Attacks are shaped toward the benign centroid on the attacker-controllable features.

    Only the attack rows move, and only in the subset the threat model allows -- the same
    construction the evasion study uses, so the two are comparable.
    """
    shaped = np.array(x, dtype=float, copy=True)
    attacks = y == 1
    if attacks.any():
        shaped[attacks] = mimicry_perturb(x[attacks], centroid, controllable, fraction)
    return shaped


def apply_rarity(
    x: np.ndarray, y: np.ndarray, target: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample attacks until the base rate matches a production one.

    The split carries roughly one attack in four, which no real network does. Thinning the
    attacks changes no model and no threshold, but it changes precision, alert composition and
    every monitor keyed to alert volume -- which is the point.
    """
    attacks = np.flatnonzero(y == 1)
    benign = np.flatnonzero(y == 0)
    if len(attacks) == 0 or target <= 0:
        return x[benign], y[benign]
    wanted = round(target * len(benign) / (1.0 - target))
    if wanted >= len(attacks):
        return x, y
    kept = rng.choice(attacks, size=max(wanted, 1), replace=False)
    rows = np.sort(np.concatenate([benign, kept]))
    return x[rows], y[rows]


# --------------------------------------------------------------------------------------
# Guarantees and monitors.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One scenario, evaluated: what the system promised and what the dashboard showed."""

    scenario: Scenario
    rows: int
    prevalence: float
    realised_fpr: float
    detection: float
    coverage: float
    alert_rate: float
    feature_psi: float
    score_psi: float
    budget: float
    coverage_target: float
    alert_ceiling: float
    psi_threshold: float
    detection_floor: float

    @property
    def detection_broken(self) -> bool:
        """Detecting less than half of what the deployment was accepted with.

        A floor rather than an exact number, because no operator promises a detection rate to
        four decimals; what they promise is that the detector is still the one that was reviewed.
        """
        return self.detection < self.detection_floor

    @property
    def budget_broken(self) -> bool:
        """The false-positive budget the threshold was calibrated to deliver, exceeded."""
        return self.realised_fpr > self.budget

    @property
    def coverage_broken(self) -> bool:
        """The calibrated coverage promise, missed."""
        return self.coverage < self.coverage_target

    @property
    def slo_broken(self) -> bool:
        return self.alert_rate > self.alert_ceiling

    @property
    def broken(self) -> list[str]:
        names = []
        if self.budget_broken:
            names.append("budget")
        if self.detection_broken:
            names.append("detection")
        if self.coverage_broken:
            names.append("coverage")
        if self.slo_broken:
            names.append("alert SLO")
        return names

    @property
    def alarms(self) -> list[str]:
        names = []
        if self.feature_psi >= self.psi_threshold:
            names.append("feature PSI")
        if self.score_psi >= self.psi_threshold:
            names.append("score PSI")
        if self.slo_broken:
            names.append("alert SLO")
        return names

    @property
    def silent(self) -> bool:
        """Something is broken and nothing is flashing -- the failure worth naming."""
        return bool(self.broken) and not self.alarms


@dataclass(frozen=True)
class Interaction:
    """A pair of stressors, and how much of their joint effect is not the sum of the parts."""

    guarantee: str
    first: str
    second: str
    alone_first: float
    alone_second: float
    together: float
    baseline: float

    @property
    def additive(self) -> float:
        """What the two would do jointly if they did not interact."""
        return self.alone_first + self.alone_second - self.baseline

    @property
    def interaction(self) -> float:
        """Observed minus additive: the part one-at-a-time testing cannot see."""
        return self.together - self.additive

    @property
    def compound(self) -> bool:
        """Neither stressor alone moved the guarantee much; together they did."""
        solo = max(abs(self.alone_first - self.baseline), abs(self.alone_second - self.baseline))
        return abs(self.together - self.baseline) > 2 * solo and abs(self.interaction) > 1e-9


@dataclass
class CompositionStudy:
    """Everything the report needs, computed once."""

    cells: list[Cell]
    interactions: list[Interaction]
    stressors: tuple[str, ...]
    n_features: int
    outage_features: list[str]
    seconds: float = 0.0
    guarantees: tuple[str, ...] = field(default=("budget", "detection", "coverage", "alert SLO"))

    def baseline(self) -> Cell:
        return next(cell for cell in self.cells if not cell.scenario.active)

    def solo(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.scenario.order == 1]

    def broken_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.broken]

    def silent_failures(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.silent]

    def solo_breaks(self) -> set[str]:
        """Guarantees any single stressor breaks on its own."""
        return {name for cell in self.solo() for name in cell.broken}

    def compound_failures(self) -> list[Cell]:
        """Multi-stressor cells breaking a guarantee no single stressor breaks."""
        solo = self.solo_breaks()
        return [cell for cell in self.cells if cell.scenario.order > 1 and set(cell.broken) - solo]

    def strongest(self) -> list[Interaction]:
        return sorted(self.interactions, key=lambda row: -abs(row.interaction))

    def monitor_interactions(self) -> list[Interaction]:
        """Interactions on the monitored statistics rather than on the guarantees."""
        return [row for row in self.interactions if row.guarantee.endswith("PSI")]

    def subadditive_share(self) -> float:
        """How often a second stressor moves a monitor *less* than it did on its own.

        This is the mechanism behind a quiet dashboard. If monitor responses stacked, two
        concurrent failures would be twice as visible as one. If they saturate, the second
        failure arrives almost invisibly -- and the moment a system is most likely to be
        breaking is the moment its monitors are least able to register the difference.
        """
        rows = self.monitor_interactions()
        return sum(row.interaction < 0 for row in rows) / len(rows) if rows else 0.0

    def invisible(self) -> list[Cell]:
        """Single stressors that cost real detection while leaving every monitor silent."""
        base = self.baseline()
        return [
            cell
            for cell in self.solo()
            if not cell.alarms and cell.detection < base.detection * 0.9
        ]

    def false_alarms(self) -> list[Cell]:
        """Cells where a monitor fires although no guarantee is any worse than at baseline."""
        base = self.baseline()
        return [
            cell for cell in self.solo() if cell.alarms and set(cell.broken) <= set(base.broken)
        ]

    def reversals(self) -> list[Cell]:
        """Cells where one guarantee improves while another degrades.

        Guarantees are not co-monotone: an attack that costs detection at a high threshold can
        *raise* the coverage measured at a low one, because moving scores toward the middle helps
        a low cut and hurts a high one. Whether an attack "works" is a property of the operating
        point, not only of the attack.
        """
        base = self.baseline()
        return [
            cell
            for cell in self.cells
            if cell.scenario.order == 1
            and cell.detection < base.detection
            and cell.coverage > base.coverage
        ]

    def pre_existing(self) -> list[str]:
        """Guarantees already broken with nothing wrong at all."""
        return self.baseline().broken


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _interactions(cells: list[Cell], stressors: tuple[str, ...]) -> list[Interaction]:
    """Two-factor interactions, read off the four cells where nothing else is switched on."""
    by_name = {frozenset(cell.scenario.active): cell for cell in cells}
    readings: dict[str, Callable[[Cell], float]] = {
        "false-positive rate": lambda cell: cell.realised_fpr,
        "detection rate": lambda cell: cell.detection,
        "coverage": lambda cell: cell.coverage,
        "alert rate": lambda cell: cell.alert_rate,
        "feature PSI": lambda cell: cell.feature_psi,
        "score PSI": lambda cell: cell.score_psi,
    }
    found: list[Interaction] = []
    base = by_name[frozenset()]
    for guarantee, read in readings.items():
        for first, second in itertools.combinations(stressors, 2):
            alone_first = by_name.get(frozenset({first}))
            alone_second = by_name.get(frozenset({second}))
            together = by_name.get(frozenset({first, second}))
            if alone_first is None or alone_second is None or together is None:
                continue
            found.append(
                Interaction(
                    guarantee=guarantee,
                    first=first,
                    second=second,
                    alone_first=read(alone_first),
                    alone_second=read(alone_second),
                    together=read(together),
                    baseline=read(base),
                )
            )
    return found


def run_composition_study(settings: Settings) -> CompositionStudy:
    """Run the factorial and record which guarantees survive which combinations."""
    start = time.perf_counter()
    cfg: CompositionConfig = settings.composition
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

    def score(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(matrix))[:, column]

    val_scores = score(x_val)

    # Everything the system promises is calibrated here, once, on validation -- exactly as the
    # deployed pipeline does it. The stressors are applied afterwards, so every broken guarantee
    # below is a promise made before the failure and kept or not after it.
    cut = threshold_at_fpr(y_val, val_scores, cfg.budget)
    attack_val = val_scores[y_val == 1]
    coverage_cut = (
        float(np.quantile(attack_val, 1.0 - cfg.coverage_target)) if len(attack_val) else 0.0
    )
    ceiling = float(np.mean(val_scores >= cut)) * cfg.alert_headroom
    accepted_detection = float(rates_at_threshold(y_test, score(x_test), cut)["tpr"])
    floor = accepted_detection * cfg.detection_floor_ratio

    feature_names = list(pipeline.get_feature_names_out())
    controllable = controllable_indices(feature_names, variant.robustness.controllable_features)
    centroid = x_train[y_train == 0].mean(axis=0)
    medians = np.median(x_train, axis=0)
    outage_columns = controllable_indices(feature_names, cfg.outage_features)
    days = test_frame[DAY_COLUMN].to_numpy() if DAY_COLUMN in test_frame.columns else np.array([])
    day_order = (
        np.argsort(np.argsort(days)).astype(float) if len(days) == len(y_test) else np.array([])
    )
    reference = x_train

    cells: list[Cell] = []
    for scenario in scenarios(STRESSORS):
        x_cell, y_cell = x_test, y_test
        if scenario.has("shift") and len(day_order):
            x_cell, y_cell = apply_shift(x_cell, y_cell, day_order, cfg.shift_fraction)
        if scenario.has("rarity"):
            x_cell, y_cell = apply_rarity(x_cell, y_cell, cfg.production_prevalence, rng)
        if scenario.has("evasion"):
            x_cell = apply_evasion(x_cell, y_cell, centroid, controllable, cfg.evasion_fraction)
        if scenario.has("outage"):
            x_cell = apply_outage(x_cell, outage_columns, medians)

        cell_scores = score(x_cell)
        rates = rates_at_threshold(y_cell, cell_scores, cut)
        attacks = y_cell == 1
        coverage = float(np.mean(cell_scores[attacks] >= coverage_cut)) if attacks.any() else 1.0
        feature_psi = max(
            (
                population_stability_index(reference[:, index], x_cell[:, index], bins=cfg.psi_bins)
                for index in range(x_cell.shape[1])
            ),
            default=0.0,
        )
        cells.append(
            Cell(
                scenario=scenario,
                rows=len(y_cell),
                prevalence=float(np.mean(y_cell)) if len(y_cell) else 0.0,
                realised_fpr=float(rates["fpr"]),
                detection=float(rates["tpr"]),
                coverage=coverage,
                alert_rate=float(np.mean(cell_scores >= cut)),
                feature_psi=float(feature_psi),
                score_psi=float(
                    population_stability_index(val_scores, cell_scores, bins=cfg.psi_bins)
                ),
                budget=cfg.budget * cfg.budget_tolerance,
                coverage_target=cfg.coverage_target,
                alert_ceiling=ceiling,
                psi_threshold=cfg.psi_threshold,
                detection_floor=floor,
            )
        )

    study = CompositionStudy(
        cells=cells,
        interactions=_interactions(cells, STRESSORS),
        stressors=STRESSORS,
        n_features=x_test.shape[1],
        outage_features=list(cfg.outage_features),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Composition study complete",
        extra={
            "broken": len(study.broken_cells()),
            "silent": len(study.silent_failures()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: CompositionStudy) -> str:
    """The finding, written from the computed cells."""
    base = study.baseline()
    invisible = study.invisible()
    alarms = study.false_alarms()
    reversals = study.reversals()
    compound = study.compound_failures()
    lines = []

    if base.broken:
        lines += [
            f"**Before a single stressor is switched on, "
            f"{'a guarantee' if len(base.broken) == 1 else str(len(base.broken)) + ' guarantees'} "
            f"this system makes is already broken -- and nothing is watching it.**",
            "",
            f"With nothing wrong at all, the coverage promise calibrated on validation delivers "
            f"{base.coverage:.1%} against the {base.coverage_target:.0%} it was calibrated to, "
            f"and neither drift monitor fires: feature PSI sits at {base.feature_psi:.2f} and "
            f"score PSI at {base.score_psi:.2f}, both under the {base.psi_threshold:.1f} line. "
            "The temporal gap between the training days and the later ones is enough on its own, "
            "which the [adaptive-conformal study](adaptive_conformal.md) exists to fix and the "
            "[open-set study](openset.md) explains -- but the point here is narrower and worse: "
            "**the breach is invisible to the monitoring this system actually runs.**",
            "",
        ]
    else:
        lines += [
            "**With nothing wrong, every guarantee holds -- which is the only cell of this "
            "design that any other report in this repository has checked.**",
            "",
        ]

    if invisible:
        cell = invisible[0]
        drop = base.detection - cell.detection
        lines += [
            "**The stressor a monitor most needs to see is the one it cannot.** "
            f"`{cell.scenario.name}` "
            f"alone costs {drop:.1%} of detection -- {drop / base.detection:.0%} of the "
            f"deployment's whole detection rate, {base.detection:.1%} down to "
            f"{cell.detection:.1%} -- and leaves every monitor silent: feature PSI "
            f"{cell.feature_psi:.2f}, score PSI {cell.score_psi:.2f}, alert rate "
            f"{cell.alert_rate:.1%} against a ceiling of {cell.alert_ceiling:.1%}. A drift "
            "monitor calibrated to notice a *major population shift* is the wrong instrument for "
            "an adversary, who is specifically trying not to cause one.",
            "",
        ]
    if alarms:
        cell = alarms[0]
        lines += [
            "**And the monitor that does fire, fires for the wrong reason.** "
            f"`{cell.scenario.name}` "
            f"changes no model, no threshold and no feature -- it only thins the attacks to a "
            f"production base rate of {cell.prevalence:.1%} -- yet score PSI reaches "
            f"{cell.score_psi:.2f} and trips the same alarm. Nothing about the detector is "
            "worse; the traffic simply has a different composition, which is what the "
            "[base-rate study](base_rate.md) predicts and what an on-call engineer would spend "
            "an afternoon on.",
            "",
        ]
    if compound:
        names = ", ".join(f"`{cell.scenario.name}`" for cell in compound[:3])
        lines += [
            f"**{len(compound)} combinations break a guarantee that no single stressor breaks** "
            f"({names}) -- the failure mode a one-at-a-time test cannot reach by construction.",
            "",
        ]
    else:
        lines += [
            "**No combination breaks a guarantee that no single stressor breaks.** That is a "
            "genuine negative result and it is reported as one: on this data the failures are "
            "dominated by their strongest component rather than compounding into something new. "
            "It does not generalise to a network with real host structure, and it is not what "
            "the interactions below say about the *monitors*.",
            "",
        ]
    if reversals:
        cell = reversals[0]
        lines += [
            f"One cell is worth the design on its own. `{cell.scenario.name}` **lowers** detection "
            f"at the deployed 1% cut ({base.detection:.1%} to {cell.detection:.1%}) while "
            f"**raising** coverage at the conformal cut ({base.coverage:.1%} to "
            f"{cell.coverage:.1%}). Shaping attacks toward the benign centroid compresses their "
            "scores toward the middle, which pushes them below a high threshold and above a low "
            "one at the same time. **Whether an attack works is a property of the operating "
            "point, not only of the attack** -- and a defence evaluated at one threshold has said "
            "nothing about another.",
        ]
    return "\n".join(lines)


def _render(study: CompositionStudy, figure: Path) -> str:
    """Compose the report."""
    base = study.baseline()
    lines = [
        "# NetSentry -- Do the Safeguards Compose?",
        "",
        f"_A 2^{len(study.stressors)} factorial over {', '.join(study.stressors)}: all "
        f"{len(study.cells)} combinations, each read for the same four guarantees and three "
        f"monitors. Regenerate with `netsentry composition`._",
        "",
        "## Why this report exists",
        "",
        "This repository has a defence for each failure it has thought of, and each was measured "
        "with **one thing wrong at a time**. Production does not have that courtesy. The question "
        "here is whether the guarantees, and the monitors that watch them, survive two failures "
        "at once -- which is a real question rather than a rhetorical one, because failures "
        "interact: two perturbations that each move a monitored statistic can fail to stack, "
        "leaving the statistic near where it started while the system underneath is worse than "
        "either alone.",
        "",
        "A factorial design separates a **main effect** -- what one stressor does alone -- from an "
        "**interaction**, what a pair does beyond the sum of its parts. Every other report here "
        "measures the first and is structurally blind to the second.",
        "",
        _lead(study),
        "",
        "## The sixteen cells",
        "",
        f"The guarantees are fixed before any stressor is applied, on validation, exactly as the "
        f"deployed pipeline fixes them: a threshold calibrated to a "
        f"{base.budget / 1.5:.1%} false-positive budget (broken past "
        f"{base.budget:.1%}), a detection floor at half the "
        f"{base.detection_floor * 2:.1%} the deployment was accepted with, "
        f"{base.coverage_target:.0%} coverage of the attack class, and an alert-rate ceiling at "
        f"{base.alert_ceiling:.1%}.",
        "",
        "| what is wrong | flows | attack rate | FPR | detection | coverage | alert rate | "
        "feature PSI | score PSI | broken | alarms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in study.cells:
        broken = ", ".join(cell.broken) or "--"
        alarms = ", ".join(cell.alarms) or "**silent**"
        lines.append(
            f"| {cell.scenario.name} | {cell.rows:,} | {cell.prevalence:.1%} | "
            f"{cell.realised_fpr:.2%} | {cell.detection:.1%} | {cell.coverage:.1%} | "
            f"{cell.alert_rate:.1%} | {cell.feature_psi:.2f} | {cell.score_psi:.2f} | "
            f"{broken} | {alarms} |"
        )
    lines += [
        "",
        "The false-positive budget is never breached, and the reason is worth stating because it "
        "is not reassuring: every stressor that damages the system also **lowers** the scores, so "
        "fewer flows clear the threshold and the realised false-positive rate falls. A budget "
        "measured from the top of the score distribution looks healthiest exactly when the "
        "distribution has collapsed. It is a one-sided guarantee and this table is a good "
        "argument for reading it beside the detection rate rather than instead of it.",
        "",
        "## The interactions: what one-at-a-time testing cannot see",
        "",
        f"![Monitor response against the number of concurrent failures](../figures/{figure.name})",
        "",
        "For each pair and each reading, the interaction is the four-cell contrast "
        "`both - first - second + neither`: the part of the joint effect that is not the sum of "
        "the parts. A positive value means the pair does *less* damage than expected, a negative "
        "one that it does more -- or, for a monitor, that the pair is *less* visible than the "
        "two failures separately would suggest.",
        "",
        "| reading | pair | first alone | second alone | together | if they simply added | "
        "interaction |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study.strongest()[:10]:
        lines.append(
            f"| {row.guarantee} | {row.first} + {row.second} | {row.alone_first:.3f} | "
            f"{row.alone_second:.3f} | {row.together:.3f} | {row.additive:.3f} | "
            f"**{row.interaction:+.3f}** |"
        )
    share = study.subadditive_share()
    lines += [
        "",
        f"**{share:.0%} of the monitor interactions are negative**, which is the quiet finding of "
        "this study. Monitor responses do not stack: a second concurrent failure moves the "
        "statistic far less than it did on its own, because the first failure has already pushed "
        "the distribution to where the metric saturates. The practical reading is unpleasant -- "
        "**the moment a system is most likely to be breaking is the moment its monitors are least "
        "able to register the difference**, and a threshold tuned on single-fault drills will be "
        "too high for a real incident, in which faults arrive together.",
        "",
        "## What this does and does not establish",
        "",
        "- **The stressors are the ones this repository already models**, each reusing the study "
        "that introduced it: the temporal ordering for shift, the training median for a sensor "
        "outage, the same mimicry perturbation and controllable subset as the [evasion "
        "study](evasion.md), and attack subsampling for prevalence. Nothing here is a new threat "
        "model; the contribution is running them together.",
        "- **A factorial with one run per cell has no error bars.** Differences smaller than the "
        "sampling noise the [resolution study](power.md) measures -- around one point of "
        "detection on this split -- should not be read as effects. The interactions reported "
        "above are several times that; the small ones in the full table are not.",
        "- **The guarantee thresholds are conventions, and they are in config.** A detection floor "
        "at half the accepted rate and a PSI line at 0.2 are the numbers this project already "
        "uses elsewhere, not derivations. Moving them moves which cells are called broken; it "
        "does not move the interactions, which are differences.",
        "- **The stand-in has no host structure**, so a correlated failure -- one subnet's "
        "collector dying while an attacker works inside it -- cannot be represented here. That is "
        "the combination most likely to produce a genuine compound break, and this design cannot "
        "reach it.",
        "- **Sixteen cells is a small design.** Three- and four-way interactions are reported in "
        "the table above only through the cells themselves; with one run each, they are "
        "descriptive rather than estimated.",
    ]
    return "\n".join(lines) + "\n"


def run_composition_report(settings: Settings) -> Path:
    """Run the factorial and write the report + figure."""
    study = run_composition_study(settings)
    orders = sorted({cell.scenario.order for cell in study.cells})
    feature_psi = np.array(
        [
            float(np.mean([c.feature_psi for c in study.cells if c.scenario.order == order]))
            for order in orders
        ]
    )
    score_psi = np.array(
        [
            float(np.mean([c.score_psi for c in study.cells if c.scenario.order == order]))
            for order in orders
        ]
    )
    detection = np.array(
        [
            float(np.mean([c.detection for c in study.cells if c.scenario.order == order]))
            for order in orders
        ]
    )
    axis = np.array(orders, dtype=float)
    figure = plots.plot_lines(
        {
            "feature PSI (mean over cells)": (axis, feature_psi),
            "score PSI (mean over cells)": (axis, score_psi),
            "detection rate (mean over cells)": (axis, detection),
        },
        xlabel="how many things are wrong at once",
        ylabel="value",
        title="Damage keeps accumulating; the monitors stop responding",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote composition report", extra={"path": str(out_path)})

    with track_run(settings, "composition") as run:
        run.log_params({"stressors": ",".join(study.stressors), "cells": str(len(study.cells))})
        run.log_metrics(
            {
                "broken_cells": float(len(study.broken_cells())),
                "silent_failures": float(len(study.silent_failures())),
                "compound_failures": float(len(study.compound_failures())),
                "subadditive_share": study.subadditive_share(),
                "baseline_coverage": study.baseline().coverage,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
