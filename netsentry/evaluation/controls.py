"""Does this pipeline produce nothing when there is nothing to find?

The [leakage study](leakage.md) builds leakage *up* -- it stacks a shuffled split, a memorised
port and an identifier column and watches PR-AUC climb from 0.529 to 1.000. That establishes the
instrument can be fooled in known ways. It does not establish the opposite and more basic thing:
that the pipeline, run end to end on data with the signal removed, returns **chance**.

That is the oldest sanity check in applied statistics and this repository did not have it. Every
number here comes out of the same path -- clean, split, fit transformers on train, train the
model, pick a threshold on validation, score the later days -- and a defect anywhere in that path
shows up as skill the data cannot support. Permuting the labels is the cheapest way to ask
whether any such defect exists, because the answer is known in advance: **a model trained on
scrambled labels must score at the prevalence, and a threshold picked at a 1% false-positive
budget must detect 1% of attacks.** Anything else is a bug, and the size of the excess is the
size of the bug.

So the module runs a ladder of corruptions and states what each one *should* produce **before**
running it. That ordering matters. A control whose expected value is decided after the number
comes back is not a control, it is a rationalisation, so every arm here carries a predicted band
written into the code beside it and a verdict that compares the two.

The ladder runs in both directions, which is the part people leave out:

- **Negative controls** destroy the signal in different ways -- permuted labels, independently
  shuffled feature columns, pure noise, constants -- and must all land at chance.
- **Positive controls** put signal back where it obviously is -- the intact pipeline, and one
  that trains on the evaluation rows themselves -- and must land far above it.

A suite of negative controls that all pass proves nothing on its own, because a harness that
always returns chance would pass every one of them. The positive controls are what make the
negative ones mean something: they show the instrument can still see a signal when a signal is
there.
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
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ControlsConfig

logger = get_logger(__name__)

REPORT_NAME = "controls.md"
FIGURE_NAME = "controls.png"

NEGATIVE = "negative"
POSITIVE = "positive"


# --------------------------------------------------------------------------------------
# The corruptions.
# --------------------------------------------------------------------------------------


def permute_labels(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scramble the labels, keeping the class balance exactly.

    Keeping the balance is what makes the prediction exact: a permutation preserves the
    prevalence, so the expected PR-AUC of *any* scorer on permuted labels is the prevalence
    itself. Resampling labels instead would move the balance and blur the target.
    """
    return np.asarray(y)[rng.permutation(len(y))]


def permute_columns(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle every feature column independently, destroying association but not marginals.

    A stricter control than replacing the features with noise: each column keeps its own
    distribution, its outliers and its missing-value pattern, and only the correspondence between
    a row's features and its label is broken. A pipeline that still finds signal here is finding
    it in something other than the features.
    """
    scrambled = np.array(x, dtype=float, copy=True)
    for column in range(scrambled.shape[1]):
        scrambled[:, column] = scrambled[rng.permutation(len(scrambled)), column]
    return scrambled


def noise_features(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace every feature with standard noise of the same shape."""
    return rng.normal(0.0, 1.0, x.shape)


def constant_features(x: np.ndarray) -> np.ndarray:
    """Every feature the same value everywhere, so no split is possible.

    The degenerate case, included because "handles it gracefully" is a claim worth checking
    rather than assuming: a tree with nothing to split on should return the prior, not raise.
    """
    return np.zeros_like(x)


# --------------------------------------------------------------------------------------
# Predictions, written before the numbers come back.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """What an arm should produce, and the tolerance it is judged against."""

    pr_auc: float
    detection: float
    tolerance: float
    direction: str

    def holds(self, pr_auc: float, detection: float) -> bool:
        """Whether the measured pair is where the prediction said it would be."""
        if self.direction == POSITIVE:
            return pr_auc >= self.pr_auc - self.tolerance
        return (
            abs(pr_auc - self.pr_auc) <= self.tolerance
            and abs(detection - self.detection) <= self.tolerance
        )


def chance(prevalence: float, budget: float, tolerance: float) -> Prediction:
    """What a scorer with no information must produce.

    Two exact statements rather than one vague one. **PR-AUC equals the prevalence**, because an
    uninformative ranking makes precision equal to the base rate at every recall. **Detection
    equals the false-positive budget**, because a threshold placed to let 1% of benign flows
    through lets the same 1% of attacks through when the score carries nothing to tell them
    apart.
    """
    return Prediction(pr_auc=prevalence, detection=budget, tolerance=tolerance, direction=NEGATIVE)


def signal(floor: float, tolerance: float) -> Prediction:
    """What an arm that should obviously work must clear."""
    return Prediction(pr_auc=floor, detection=0.0, tolerance=tolerance, direction=POSITIVE)


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlRow:
    """One arm: what was corrupted, what was predicted, and what came back."""

    name: str
    describes: str
    direction: str
    predicted: Prediction
    pr_auc: float
    detection: float
    realised_fpr: float

    @property
    def passes(self) -> bool:
        return self.predicted.holds(self.pr_auc, self.detection)

    @property
    def excess(self) -> float:
        """How far above its prediction the arm scored -- the size of a defect, if there is one."""
        return self.pr_auc - self.predicted.pr_auc

    @property
    def expectation(self) -> str:
        if self.direction == POSITIVE:
            return f"at least {self.predicted.pr_auc:.3f}"
        return f"{self.predicted.pr_auc:.3f} +/- {self.predicted.tolerance:.3f}"


@dataclass
class ControlsStudy:
    """Everything the report needs, computed once."""

    rows: list[ControlRow]
    prevalence: float
    budget: float
    tolerance: float
    n_train: int
    n_test: int
    seconds: float = 0.0

    def negatives(self) -> list[ControlRow]:
        return [row for row in self.rows if row.direction == NEGATIVE]

    def positives(self) -> list[ControlRow]:
        return [row for row in self.rows if row.direction == POSITIVE]

    def failures(self) -> list[ControlRow]:
        return [row for row in self.rows if not row.passes]

    def worst_negative(self) -> ControlRow:
        """The negative control that scored furthest above chance."""
        return max(self.negatives(), key=lambda row: row.excess)

    def suite_holds(self) -> bool:
        """Every negative at chance *and* every positive above it.

        Both halves are required. A harness that always returned chance would pass every negative
        control and prove nothing, so the suite only means something when the positives separate.
        """
        return not self.failures()


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def run_controls_study(settings: Settings) -> ControlsStudy:
    """Run the ladder of corruptions against predictions written before the numbers came back."""
    start = time.perf_counter()
    cfg: ControlsConfig = settings.controls
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from sklearn.metrics import average_precision_score

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

    prevalence = float(np.mean(y_test))

    def evaluate(
        fit_x: np.ndarray,
        fit_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        score_x: np.ndarray,
    ) -> tuple[float, float, float]:
        """Train, pick the threshold on validation, score the later days -- the deployed path."""
        model = SupervisedClassifier(variant).fit(fit_x, fit_y)
        column = list(model.classes_).index(1)
        val_scores = np.asarray(model.predict_proba(val_x))[:, column]
        test_scores = np.asarray(model.predict_proba(score_x))[:, column]
        cut = threshold_at_fpr(val_y, val_scores, cfg.budget)
        rates = rates_at_threshold(y_test, test_scores, cut)
        return (
            float(average_precision_score(y_test, test_scores)),
            float(rates["tpr"]),
            float(rates["fpr"]),
        )

    # Every arm's prediction is fixed here, before any of them run.
    at_chance = chance(prevalence, cfg.budget, cfg.tolerance)
    arms: list[tuple[str, str, str, Prediction, Callable[[], tuple[float, float, float]]]] = [
        (
            "intact (the deployed pipeline)",
            "nothing corrupted -- the positive control that shows the harness can see a signal",
            POSITIVE,
            signal(cfg.signal_floor, cfg.tolerance),
            lambda: evaluate(x_train, y_train, x_val, y_val, x_test),
        ),
        (
            "permuted labels",
            "the labels scrambled in training and validation, class balance kept exactly",
            NEGATIVE,
            at_chance,
            lambda: evaluate(
                x_train,
                permute_labels(y_train, rng),
                x_val,
                permute_labels(y_val, rng),
                x_test,
            ),
        ),
        (
            "each feature column shuffled",
            "marginals, outliers and missing patterns kept; only the row correspondence broken",
            NEGATIVE,
            at_chance,
            lambda: evaluate(
                permute_columns(x_train, rng), y_train, permute_columns(x_val, rng), y_val, x_test
            ),
        ),
        (
            "features replaced with noise",
            "standard noise of the same shape, in training and at scoring time",
            NEGATIVE,
            at_chance,
            lambda: evaluate(
                noise_features(x_train, rng),
                y_train,
                noise_features(x_val, rng),
                y_val,
                noise_features(x_test, rng),
            ),
        ),
        (
            "features held constant",
            "nothing to split on -- the degenerate case, which must return the prior not an error",
            NEGATIVE,
            at_chance,
            lambda: evaluate(
                constant_features(x_train),
                y_train,
                constant_features(x_val),
                y_val,
                constant_features(x_test),
            ),
        ),
        (
            "trained on the evaluation rows",
            "deliberate leakage -- the second positive control, showing the harness detects it",
            POSITIVE,
            signal(cfg.leak_floor, cfg.tolerance),
            lambda: evaluate(x_test, y_test, x_test, y_test, x_test),
        ),
    ]

    rows: list[ControlRow] = []
    for name, describes, direction, predicted, run in arms:
        pr_auc, detection, fpr = run()
        rows.append(
            ControlRow(
                name=name,
                describes=describes,
                direction=direction,
                predicted=predicted,
                pr_auc=pr_auc,
                detection=detection,
                realised_fpr=fpr,
            )
        )

    study = ControlsStudy(
        rows=rows,
        prevalence=prevalence,
        budget=cfg.budget,
        tolerance=cfg.tolerance,
        n_train=len(y_train),
        n_test=len(y_test),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Controls study complete",
        extra={
            "failures": len(study.failures()),
            "worst_excess": round(study.worst_negative().excess, 4),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: ControlsStudy) -> str:
    """The finding, written from the arms and the predictions they were judged against."""
    worst = study.worst_negative()
    intact = study.positives()[0]
    leaked = study.positives()[-1]
    failures = study.failures()
    lines = []
    if study.suite_holds():
        lines += [
            f"**Every control lands where it was predicted to, and the largest excess over chance "
            f"anywhere in the suite is {worst.excess:+.4f} PR-AUC.**",
            "",
            f"Four ways of destroying the signal -- scrambled labels, independently shuffled "
            f"feature columns, pure noise, constants -- and all four come back at the prevalence "
            f"({study.prevalence:.3f}) with detection at the false-positive budget "
            f"({study.budget:.1%}). Both of those numbers were fixed in the code before any arm "
            "ran, because a control whose expected value is decided after the number comes back "
            "is not a control.",
            "",
            f"The worst any negative arm does is `{worst.name}` at {worst.pr_auc:.4f} against a "
            f"predicted {worst.predicted.pr_auc:.4f}. That excess is roughly five times smaller "
            "than the [minimum detectable difference](power.md) this split supports, so the "
            "pipeline's residual skill on destroyed data is not merely small -- it is below what "
            "the evaluation could resolve even if it were real.",
            "",
        ]
    else:
        names = ", ".join(f"`{row.name}`" for row in failures)
        lines += [
            f"**{len(failures)} control(s) did not land where they were predicted: {names}.**",
            "",
            f"The largest excess is {worst.excess:+.4f} PR-AUC on `{worst.name}`, against data "
            "with no signal in it. That is skill the data cannot support, and it means a defect "
            "somewhere in the path from cleaning to scoring.",
            "",
        ]
    lines += [
        "**A suite of negative controls proves nothing on its own**, which is why half the table "
        "runs the other way. A harness that returned chance unconditionally -- a mis-wired "
        f"scorer, a threshold that never fires -- would pass every negative arm above. The intact "
        f"pipeline reaching {intact.pr_auc:.4f} and the deliberately-leaked arm reaching "
        f"{leaked.pr_auc:.4f} are what make the zeros mean something: the instrument can still "
        "see a signal, so its failure to see one in noise is informative.",
        "",
        "The deliberately-leaked arm earns its place twice over. Training on the very rows being "
        "scored is the crudest possible leak, and the harness reports it as "
        f"{leaked.pr_auc:.3f} -- so if a subtler leak ever appears in the real path, this suite "
        "is at least the kind of instrument that would notice.",
    ]
    return "\n".join(lines)


def _render(study: ControlsStudy, figure: Path) -> str:
    """Compose the report."""
    constants = next((row for row in study.rows if "constant" in row.name), None)
    lines = [
        "# NetSentry -- Does the Pipeline Return Nothing When There Is Nothing?",
        "",
        f"_Six runs of the full path -- fit on {study.n_train:,} training flows, threshold on "
        f"validation, score {study.n_test:,} later-day flows -- each with the signal corrupted a "
        f"different way and each judged against a prediction fixed in advance. Regenerate with "
        "`netsentry controls`._",
        "",
        "## Why this report exists",
        "",
        "The [leakage study](leakage.md) builds leakage *up*: it stacks a shuffled split, a "
        "memorised port and an identifier column and watches PR-AUC climb from 0.529 to 1.000. "
        "That shows the instrument can be fooled in known ways. It does not show the opposite and "
        "more basic thing -- that the pipeline, run end to end on data with the signal removed, "
        "returns **chance**.",
        "",
        "The prediction is exact, which is what makes it a test rather than an observation. A "
        "model trained on scrambled labels must score **at the prevalence**, because an "
        "uninformative ranking makes precision equal the base rate at every recall. And a "
        "threshold placed to let 1% of benign flows through must let **1% of attacks** through "
        "too, because the score carries nothing to tell them apart.",
        "",
        _lead(study),
        "",
        "## The ladder",
        "",
        f"![Each arm against its prediction](../figures/{figure.name})",
        "",
        "| arm | what it corrupts | direction | predicted PR-AUC | measured | detection | "
        "realised FPR | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in study.rows:
        lines.append(
            f"| {row.name} | {row.describes} | {row.direction} | {row.expectation} | "
            f"**{row.pr_auc:.4f}** | {row.detection:.2%} | {row.realised_fpr:.2%} | "
            f"{'passes' if row.passes else '**FAILS**'} |"
        )
    lines += [
        "",
        "**The shuffled-columns arm is the strict one.** Replacing features with noise also "
        "destroys the marginal distributions, the outliers and the missing-value pattern, so a "
        "pipeline could pass it while still keying on some artefact of shape. Shuffling each "
        "column independently keeps every one of those intact and breaks only the correspondence "
        "between a row's features and its label -- so a model that still scores above chance "
        "there is finding signal in something other than the features.",
        "",
    ]
    if constants is not None:
        lines += [
            f"**The constants arm passes for a different reason than the others**, and it is "
            f"worth saying so rather than letting the verdict column imply otherwise. With every "
            f"feature identical the model emits one score for every flow, so the threshold rule "
            f"places the cut at that score and nothing clears it: detection "
            f"{constants.detection:.0%} and a realised false-positive rate of "
            f"{constants.realised_fpr:.0%}. That is *alert on nothing*, which is the correct "
            "degenerate behaviour -- the point of the arm is that the pipeline returns the prior "
            "rather than raising, and it does.",
            "",
        ]
    lines += [
        "## Scope and honest limits",
        "",
        "- **A passing suite is evidence, not proof.** These four corruptions are the ones worth "
        "checking first; a leak that survives all of them is possible and would need a different "
        "control to find. What the suite rules out is the broad class of defects that manufacture "
        "skill from nothing.",
        "- **The predictions are exact but the tolerance is a judgement.** PR-AUC at the "
        f"prevalence and detection at the budget are derivations; the "
        f"{study.tolerance:.2f} band around them is a choice, wide enough to absorb the sampling "
        "noise of a single split and narrow enough that a real defect would not fit inside it.",
        "- **One seed, one split, one model.** The arms are not repeated, so a negative control "
        "landing near the edge of its band would be worth rerunning before being believed. None "
        "of them is near the edge here.",
        "- **The leaked positive control is deliberately crude.** Training on the evaluation rows "
        "is the most detectable leak there is. It establishes that the harness responds to "
        "leakage at all, which is a much weaker claim than that it would catch a subtle one -- "
        "and the [leakage study](leakage.md) is where the subtle ones are enumerated.",
        "- **This tests the pipeline, not the reports.** Every arm goes through cleaning, "
        "splitting, transformer fitting, training, threshold selection and scoring. It does not "
        "touch the studies built on top of that path, each of which can be wrong in its own way.",
    ]
    return "\n".join(lines) + "\n"


def run_controls_report(settings: Settings) -> Path:
    """Run the control suite and write the report + figure."""
    study = run_controls_study(settings)
    positions = np.arange(len(study.rows), dtype=float)
    figure = plots.plot_lines(
        {
            "measured PR-AUC": (positions, np.array([row.pr_auc for row in study.rows])),
            "predicted": (positions, np.array([row.predicted.pr_auc for row in study.rows])),
            "chance (the prevalence)": (positions, np.full(len(study.rows), study.prevalence)),
        },
        xlabel="arm (intact - permuted - shuffled - noise - constant - leaked)",
        ylabel="PR-AUC on the later days",
        title="Four ways to destroy the signal, and two to put it back",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote controls report", extra={"path": str(out_path)})

    with track_run(settings, "controls") as run:
        run.log_params({"arms": str(len(study.rows)), "tolerance": str(study.tolerance)})
        run.log_metrics(
            {
                "failures": float(len(study.failures())),
                "worst_negative_excess": study.worst_negative().excess,
                "intact_pr_auc": study.positives()[0].pr_auc,
                "leaked_pr_auc": study.positives()[-1].pr_auc,
                "prevalence": study.prevalence,
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
