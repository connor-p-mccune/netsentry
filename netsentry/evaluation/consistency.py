"""The same number, stated a dozen times. Do the reports agree with each other?

A great many reports here open by saying what the deployed model scores -- as a baseline to beat,
an incumbent to compare against, a control arm. [`netsentry claims`](claims.md) checks that each
of those numbers appears in the report a reader is sent to. Nothing checks whether the reports
agree **with one another**.

A naive harvest says they badly do not: seven distinct values across a dozen reports, spanning a
third of the metric's usable range, every one of them reading like the thing this project's
classifier scores.

**Four of the seven are not disagreements at all**, and finding that out is most of the work.
Two are ROC-AUC, stated as "AUC" a few words from a PR-AUC in the same sentence -- different
metrics, not comparable, one insensitive to prevalence and the other not. One is the far side of
a comparison ("rises from 0.433 (static) to 0.544 (retrained)"), where the qualifier owning the
number *follows* it. One is a per-site local model. **Each is a trap a reader skimming the same
sentence falls into as readily as a regular expression does**, which is why this module reports
its rejections instead of quietly applying them.

What survives is three values spanning 0.021 PR-AUC, and the second half of the module asks what
produces them. It recomputes the score under a ladder of one-knob variations from the canonical
configuration -- cap the training rows, thin the ensemble, score a fraction of the split, average
over time-ordered batches, score a single capture day -- and matches each surviving value to the
rung that reaches it.

Two design choices keep that from being a fit rather than a diagnosis. A rung whose knob is
random is an **interval** rather than a point, because a study evaluating on a random third does
not land on one number. And attribution prefers the **narrowest** rung that covers a value, then
reports how many rungs covered it at all: a value several rungs reach has been bracketed, not
explained, and saying so is the difference between a diagnosis and a curve that fits everything.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ConsistencyConfig

logger = get_logger(__name__)

REPORT_NAME = "consistency.md"

#: A number attributed to the deployed model. The qualifier has to come first and stay close, or
#: the pattern harvests every value in a comparison table rather than the incumbent's.
INCUMBENT = re.compile(
    r"(?P<word>incumbent|deployed|LightGBM|baseline|static|centralized|central)"
    r"[^|\n]{0,60}?(?P<value>0\.[0-9]{3})",
    re.IGNORECASE,
)

#: A rank metric that is *not* average precision. Reports state both, often in the same sentence,
#: and they are not comparable: ROC-AUC is insensitive to prevalence where PR-AUC is not.
OTHER_METRIC = re.compile(r"(?<!PR-)\b(?:AUC|AUROC|ROC[- ]AUC)\b", re.IGNORECASE)

#: Words naming a *different* model. When one of these sits closer to the value than the
#: incumbent qualifier does, the sentence is a comparison and the number belongs to the other arm.
RIVAL = re.compile(
    r"\b(?:retrained|adaptive|challenger|shadow|MLP|transformer|FT-Transformer|neural|"
    r"logistic|local|per-site|federated|smoothed|distilled|surrogate|stolen|hardened)\b",
    re.IGNORECASE,
)

#: Nouns that turn a qualifier into something other than a model: a *baseline day* is a date, a
#: *baseline rate* is a prevalence, and neither is a classifier with a score.
NOT_A_MODEL = re.compile(r"\b(?:day|days|period|window|rate|prevalence|traffic)\b", re.IGNORECASE)

#: Why a candidate value was not counted, in the order the filters are applied.
REASONS = {
    "metric": "states a different metric (ROC-AUC, not average precision)",
    "rival": "belongs to the other arm of a comparison sentence",
    "sense": "the qualifier is not naming a model",
}


# --------------------------------------------------------------------------------------
# Harvesting what the reports say.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Quotation:
    """One report's statement of what the deployed model scores."""

    report: str
    value: float
    context: str
    reason: str = ""

    @property
    def shown(self) -> str:
        return f"{self.value:.3f}"


def names_another_metric(context: str) -> bool:
    """Whether the text around a value calls it something other than average precision.

    Named rather than inlined because it is the filter most likely to need widening: every new
    rank metric this project reports will want a line here, and a filter buried inside a larger
    function is one nobody finds.
    """
    return bool(OTHER_METRIC.search(context))


def _steals(window: str, qualifier: str, value: str, adjacency: int) -> bool:
    """Whether a rival qualifier, not the incumbent one, owns this value.

    Two ways it can, and raw distance captures neither. A rival sitting **between** the
    incumbent word and the number is the nearer description of it. A rival **immediately after**
    the number, with no clause break in between, is a trailing label -- the shape of
    "0.544 (retrained)". A rival further along in a contrasting clause is neither: in
    "the deployed model scores 0.529, unlike the MLP" the number belongs to the deployed model,
    and a rule that only measured characters would hand it to the MLP.
    """
    position = window.rfind(value)
    qualifier_at = window.lower().rfind(qualifier.lower(), 0, position)
    for match in RIVAL.finditer(window):
        if qualifier_at != -1 and qualifier_at < match.start() < position:
            return True
        gap = window[position + len(value) : match.start()]
        if 0 <= len(gap) <= adjacency and not set(gap) & {",", ".", ";", ":"}:
            return True
    return False


def rejection_reason(window: str, qualifier: str, value: str, adjacency: int = 15) -> str:
    """Why a candidate should not be counted as the incumbent's score, or empty if it should.

    All three filters exist because the first version of this study reported the values they let
    through as disagreements between reports. Each is a mistake a reader skimming the same
    sentence makes as readily as a regular expression does, which is why the rejections are
    published in the report rather than quietly applied.
    """
    if names_another_metric(window):
        return "metric"
    if _steals(window, qualifier, value, adjacency):
        return "rival"
    after = window[window.lower().find(qualifier.lower()) + len(qualifier) :]
    if NOT_A_MODEL.match(after.lstrip()[:12]):
        return "sense"
    return ""


def harvest(
    reports: dict[str, str], low: float, high: float
) -> tuple[list[Quotation], list[Quotation]]:
    """Incumbent scores stated in the corpus: the ones that count, and the ones that do not.

    The **band** keeps out coverage levels and p-values sitting next to the word *baseline*;
    the
    three filters above keep out a different metric, the other arm of a comparison, and a
    qualifier used in a non-model sense. A harvester that over-collects manufactures the
    disagreement this study is trying to measure -- which the first version did, three ways.
    """
    kept: list[Quotation] = []
    rejected: list[Quotation] = []
    for name, body in sorted(reports.items()):
        for match in INCUMBENT.finditer(body):
            value = float(match.group("value"))
            if not low <= value <= high:
                continue
            window = body[max(0, match.start() - 30) : match.end() + 30]
            reason = rejection_reason(window, match.group("word"), match.group("value"))
            quotation = Quotation(
                report=name,
                value=value,
                context=" ".join(window.split()),
                reason=reason,
            )
            target = rejected if reason else kept
            if not any(q.report == name and q.value == value for q in target):
                target.append(quotation)
    return kept, rejected


def spread(quotations: list[Quotation]) -> list[float]:
    """The distinct values a reader would encounter, in order."""
    return sorted({quotation.value for quotation in quotations})


# --------------------------------------------------------------------------------------
# Reproducing the spread.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variation:
    """One knob, turned once, from the canonical configuration.

    A rung is a **point** when the knob is deterministic and an **interval** when it is not: a
    study that evaluates on a random third of the split does not land on one number, it lands
    somewhere in a range, and holding it to a point would call a correct value unexplained.
    """

    name: str
    describes: str
    value: float
    canonical: float
    low: float | None = None
    high: float | None = None

    @property
    def delta(self) -> float:
        return self.value - self.canonical

    @property
    def random(self) -> bool:
        return self.low is not None and self.high is not None

    @property
    def width(self) -> float:
        """How much of the axis this rung claims. A point rung claims none of it."""
        if self.low is not None and self.high is not None:
            return self.high - self.low
        return 0.0

    def covers(self, other: float, tolerance: float) -> bool:
        """Whether this rung can account for a value."""
        if self.low is not None and self.high is not None:
            return self.low - tolerance <= other <= self.high + tolerance
        return abs(other - self.value) <= tolerance

    def distance(self, other: float) -> float:
        """How far a value sits from this rung, measured to the nearer edge of an interval."""
        if self.low is not None and self.high is not None:
            return max(self.low - other, other - self.high, 0.0)
        return abs(other - self.value)

    @property
    def shown(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low:.3f} - {self.high:.3f}"
        return f"{self.value:.3f}"


@dataclass(frozen=True)
class Attribution:
    """A harvested value, matched to the knob that reproduces it."""

    value: float
    reports: tuple[str, ...]
    knob: str
    reproduced: float
    tolerance: float
    covered: bool = False
    shown: str = ""
    candidates: int = 0

    @property
    def explained(self) -> bool:
        return self.covered

    @property
    def specific(self) -> bool:
        """Whether exactly one rung reaches this value.

        A value covered by several rungs has been *bracketed*, not attributed: the ladder says it
        is consistent with more than one methodology choice and cannot say which. Reporting that
        as an explanation would be the same error as a wide confidence interval called a result.
        """
        return self.covered and self.candidates == 1

    @property
    def gap(self) -> float:
        return self.value - self.reproduced


@dataclass
class ConsistencyStudy:
    """Everything the report needs, computed once."""

    quotations: list[Quotation]
    rejected: list[Quotation]
    variations: list[Variation]
    attributions: list[Attribution]
    canonical: float
    n_reports: int
    seconds: float = 0.0
    unexplained_tolerance: float = field(default=0.005)

    def distinct(self) -> list[float]:
        return spread(self.quotations)

    def range(self) -> float:
        values = self.distinct()
        return max(values) - min(values) if values else 0.0

    def reports_for(self, value: float) -> tuple[str, ...]:
        return tuple(q.report for q in self.quotations if q.value == value)

    def unexplained(self) -> list[Attribution]:
        return [row for row in self.attributions if not row.explained]

    def bracketed(self) -> list[Attribution]:
        """Values more than one rung reaches -- consistent with several stories, pinned to none."""
        return [row for row in self.attributions if row.explained and not row.specific]

    def pinned(self) -> list[Attribution]:
        return [row for row in self.attributions if row.specific]

    def widest(self) -> Variation:
        """The knob that moves the number most -- the one most worth naming in a report."""
        return max(self.variations, key=lambda row: abs(row.delta))


def _pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def attribute(
    quotations: list[Quotation], variations: list[Variation], tolerance: float
) -> list[Attribution]:
    """Match each distinct quoted value to the rung that comes closest to reaching it."""
    rows: list[Attribution] = []
    for value in spread(quotations):
        covering = [row for row in variations if row.covers(value, tolerance)]
        # The *narrowest* rung that reaches the value, not the nearest: a point rung landing on
        # it is a far stronger claim than a wide interval that happens to contain it, and picking
        # by distance alone lets the widest rung explain everything.
        best = (
            min(covering, key=lambda row: (row.width, row.distance(value)))
            if covering
            else min(variations, key=lambda row: row.distance(value))
        )
        rows.append(
            Attribution(
                value=value,
                reports=tuple(q.report for q in quotations if q.value == value),
                knob=best.name,
                reproduced=best.value,
                tolerance=tolerance,
                covered=bool(covering),
                shown=best.shown,
                candidates=len(covering),
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def run_consistency_study(settings: Settings) -> ConsistencyStudy:
    """Harvest what the reports claim, then recompute the ladder that explains the spread."""
    start = time.perf_counter()
    cfg: ConsistencyConfig = settings.consistency
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.schema import DAY_COLUMN
    from netsentry.data.split import load_split
    from netsentry.governance.claims import read_reports
    from netsentry.models.supervised import SupervisedClassifier

    reports = read_reports(Path(settings.paths.reports_dir))
    # This report quotes other reports' numbers in its own tables, so leaving it in the corpus
    # would let a previous run's output be harvested as fresh evidence. A study that reads its
    # own output is measuring itself.
    reports.pop(Path(REPORT_NAME).stem, None)
    quotations, rejected = harvest(reports, cfg.plausible_low, cfg.plausible_high)

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    test_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = test_frame[BINARY_TARGET].to_numpy().astype(int)

    def fit_and_score(rows: np.ndarray | None, trees: int | None) -> np.ndarray:
        """One model, scored on the full test split. Only the training side varies."""
        local = variant.model_copy(deep=True)
        if trees is not None:
            local.supervised.n_estimators = trees
        subset = slice(None) if rows is None else rows
        model = SupervisedClassifier(local).fit(x_train[subset], y_train[subset])
        column = list(model.classes_).index(1)
        return np.asarray(model.predict_proba(x_test))[:, column]

    canonical_scores = fit_and_score(None, None)
    canonical = _pr_auc(y_test, canonical_scores)

    variations = [
        Variation(
            name="the canonical configuration",
            describes="the full training split, the shipped hyperparameters, all later-day flows",
            value=canonical,
            canonical=canonical,
        )
    ]

    # Knob 1: fewer training rows. A study that caps its training set to afford an expensive
    # comparison arm changes the incumbent it is comparing against, which is easy to miss.
    capped = rng.choice(len(y_train), size=min(cfg.capped_rows, len(y_train)), replace=False)
    variations.append(
        Variation(
            name=f"trained on {cfg.capped_rows:,} rows",
            describes="what a study does when an expensive arm cannot afford the full split",
            value=_pr_auc(y_test, fit_and_score(np.sort(capped), None)),
            canonical=canonical,
        )
    )

    # Knob 2: fewer trees. Studies that refit many times shrink the ensemble to stay affordable.
    variations.append(
        Variation(
            name=f"{cfg.thin_trees} trees instead of {variant.supervised.n_estimators}",
            describes="what a study does when it has to refit the model dozens of times",
            value=_pr_auc(y_test, fit_and_score(None, cfg.thin_trees)),
            canonical=canonical,
        )
    )

    # Knob 3: scored on a fraction of the split rather than all of it.
    size = max(int(len(y_test) * cfg.evaluated_fraction), 2)
    draws = []
    for _ in range(cfg.subset_draws):
        rows = rng.choice(len(y_test), size=size, replace=False)
        draws.append(_pr_auc(y_test[rows], canonical_scores[rows]))
    variations.append(
        Variation(
            name=f"scored on {cfg.evaluated_fraction:.0%} of the later days",
            describes="what a study does when it holds part of the split back for its own use",
            value=float(np.mean(draws)),
            canonical=canonical,
            low=float(np.min(draws)),
            high=float(np.max(draws)),
        )
    )

    # Knob 4: averaged over time-ordered batches rather than pooled. A mean of per-batch scores
    # is not the pooled score, because each batch has its own prevalence -- and PR-AUC depends on
    # prevalence, so averaging over batches is a different estimand rather than an approximation.
    order = np.argsort(
        (
            np.asarray(test_frame[DAY_COLUMN])
            if DAY_COLUMN in test_frame.columns
            else np.arange(len(y_test))
        ),
        kind="stable",
    )
    batches = np.array_split(order, cfg.batches)
    per_batch = [_pr_auc(y_test[batch], canonical_scores[batch]) for batch in batches]
    variations.append(
        Variation(
            name=f"averaged over {cfg.batches} time-ordered batches",
            describes="what a prequential study reports: a mean of per-batch scores",
            value=float(np.nanmean(per_batch)),
            canonical=canonical,
        )
    )

    # Knob 5: one capture day. A federated or per-site study reports a model that saw one slice.
    if DAY_COLUMN in test_frame.columns:
        days = np.asarray(test_frame[DAY_COLUMN])
        # The *typical* day, not the most extreme one. Choosing the day furthest from the
        # canonical value would let this rung explain any number at all, which is how a ladder
        # stops being a diagnosis and becomes a fit.
        per_day = {
            str(day): _pr_auc(y_test[days == day], canonical_scores[days == day])
            for day in sorted(set(days.tolist()))
        }
        usable = {day: value for day, value in per_day.items() if np.isfinite(value)}
        if usable:
            median = float(np.median(list(usable.values())))
            typical = min(usable, key=lambda day: abs(usable[day] - median))
            variations.append(
                Variation(
                    name=f"one capture day ({typical}, the median of {len(usable)})",
                    describes="what a per-site or per-day arm reports",
                    value=usable[typical],
                    canonical=canonical,
                )
            )

    study = ConsistencyStudy(
        quotations=quotations,
        rejected=rejected,
        variations=variations,
        attributions=attribute(quotations, variations, cfg.tolerance),
        canonical=canonical,
        n_reports=len(reports),
        seconds=time.perf_counter() - start,
        unexplained_tolerance=cfg.tolerance,
    )
    logger.info(
        "Consistency study complete",
        extra={
            "distinct_values": len(study.distinct()),
            "unexplained": len(study.unexplained()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: ConsistencyStudy) -> str:
    """The finding, written from the harvested and recomputed numbers."""
    distinct = study.distinct()
    naive = len({q.value for q in study.quotations} | {q.value for q in study.rejected})
    reports = len({q.report for q in study.quotations} | {q.report for q in study.rejected})
    training = [row for row in study.variations if "trained" in row.name or "trees" in row.name]
    evaluation = [
        row for row in study.variations if "batches" in row.name or "capture day" in row.name
    ]
    lines = [
        f"**A naive harvest finds {naive} different answers across {reports} reports. "
        f"{naive - len(distinct)} of them are not disagreements at all.**",
        "",
        f"Every one reads like the same quantity -- the incumbent, the baseline, the static "
        f"model, the arm to beat. Reading them properly, "
        f"{sum(1 for q in study.rejected if q.reason == 'metric')} state **a different metric** "
        "(ROC-AUC, a few words from a PR-AUC in the same sentence, and not comparable to one), "
        f"and {sum(1 for q in study.rejected if q.reason == 'rival')} belong to **the far side of "
        "a comparison** -- a retrained model, a per-site model -- where the qualifier that owns "
        "the number follows it rather than preceding it. Each is a mistake a reader skimming the "
        "same sentence makes as readily as a regular expression does, and the first version of "
        "this study made all of them.",
        "",
        f"What survives is **{len(distinct)} values spanning {study.range():.3f} PR-AUC** against "
        f"a canonical configuration that scores {study.canonical:.3f} here. That is a real "
        "spread, and the ladder below says what produces it.",
        "",
    ]
    if training and evaluation:
        worst_training = max(training, key=lambda row: abs(row.delta))
        worst_evaluation = max(evaluation, key=lambda row: abs(row.delta))
        lines += [
            f"**It is not how the model was trained that makes the reports differ; it is what "
            f"population it was scored on.** Cutting the training set to "
            f"{'12,000 rows' if 'rows' in training[0].name else 'a fraction'} or the ensemble to "
            f"a tenth of its trees moves the number by at most "
            f"{abs(worst_training.delta):.3f} -- around the "
            "[resolution study](power.md)'s minimum detectable effect, so barely a difference at "
            f"all. Changing *what is scored* moves it by up to "
            f"{abs(worst_evaluation.delta):.3f}: averaging over time-ordered batches instead of "
            "pooling costs "
            f"{abs(next(r.delta for r in evaluation if 'batches' in r.name)):.3f}, and scoring a "
            "single capture day moves it further still. The knobs a study is most likely to "
            "mention are the ones that matter least.",
            "",
        ]
    pinned, bracketed, unexplained = study.pinned(), study.bracketed(), study.unexplained()
    verdicts = []
    if pinned:
        verdicts.append(f"{len(pinned)} is reached by exactly one rung")
    if bracketed:
        verdicts.append(f"{len(bracketed)} by more than one")
    if unexplained:
        verdicts.append(f"**{len(unexplained)} by none at all**")
    lines += [
        "Of the surviving values, "
        + ", ".join(verdicts)
        + ". "
        + (
            "Nothing here is a disagreement nobody chose, which is the outcome this study was "
            "built to be able to *fail* to find."
            if not unexplained
            else "A value no rung reaches is the finding: two reports disagree for a reason "
            "nobody chose."
        ),
        "",
        "**Bracketed is not explained**, and the distinction is the point. A value several rungs "
        "reach is consistent with more than one methodology choice, and the ladder cannot say "
        "which. Reporting that as an attribution would be the same error as calling a wide "
        "confidence interval a result.",
    ]
    return "\n".join(lines)


def _render(study: ConsistencyStudy) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- Do the Reports Agree With Each Other?",
        "",
        f"_Every incumbent PR-AUC stated across {study.n_reports} generated reports, against a "
        f"ladder of {len(study.variations)} one-knob recomputations from the canonical "
        f"configuration. Regenerate with `netsentry consistency`._",
        "",
        "## Why this report exists",
        "",
        "A great many reports here open by stating what the deployed model scores -- a baseline "
        "to beat, an incumbent to compare against, a control arm. "
        "[`netsentry claims`](claims.md) checks that each of those numbers appears in the report "
        "a reader is sent to. Nothing checks whether the reports agree **with one another**.",
        "",
        _lead(study),
        "",
        "## What the reports say",
        "",
        "| value | reports stating it |",
        "|---|---|",
    ]
    for value in study.distinct():
        reports = study.reports_for(value)
        shown = ", ".join(f"[{name}]({name}.md)" for name in reports)
        lines.append(f"| **{value:.3f}** | {shown} |")
    lines += [
        "",
        "The harvester collects a three-decimal value only when a word like *incumbent*, "
        "*deployed*, *baseline* or *static* appears within sixty characters before it, and only "
        "inside the band a PR-AUC for this model plausibly occupies. Without the band it collects "
        "coverage levels and p-values that happen to sit near the word *baseline*.",
        "",
        "## What a naive harvest also picks up, and should not",
        "",
        "| value | report | why it does not count | the sentence |",
        "|---|---|---|---|",
    ]
    for quotation in study.rejected:
        excerpt = quotation.context
        excerpt = excerpt if len(excerpt) <= 90 else excerpt[:87] + "..."
        lines.append(
            f"| {quotation.value:.3f} | [{quotation.report}]({quotation.report}.md) | "
            f"{REASONS.get(quotation.reason, quotation.reason)} | `{excerpt}` |"
        )
    lines += [
        "",
        "**These are the interesting rows.** Every one was reported as a disagreement by the "
        "first version of this study, and every one is a mistake a human reader makes on the same "
        "sentence. *AUC* sitting a few words from a PR-AUC is a different metric with a different "
        "sensitivity to prevalence. `rises from 0.433 (static) to 0.544 (retrained)` puts the "
        "qualifier that owns the second number *after* it, so a rule that looks backwards "
        "attributes a retrained model's score to the frozen one. A per-site model in a table row "
        "is not the deployed model at all.",
        "",
        "The reports are not wrong; each sentence is correct where it stands. But a number is "
        "only comparable with the construction that produced it, and prose puts the two far "
        "enough apart that a regular expression -- and a reader -- can lose the connection.",
        "",
        "## What reproduces the spread",
        "",
        "| configuration | what it models | PR-AUC | vs canonical |",
        "|---|---|---|---|",
    ]
    for row in study.variations:
        delta = "--" if row.name.startswith("the canonical") else f"{row.delta:+.3f}"
        spread_note = " _(range over draws)_" if row.random else ""
        lines.append(f"| {row.name} | {row.describes} | **{row.shown}**{spread_note} | {delta} |")
    lines += [
        "",
        "Each row turns exactly one knob from the canonical configuration and scores the result "
        "the same way. None of them is wrong -- they are choices real studies here made, for "
        "reasons those studies give. The random rung is shown as a range because a study "
        "evaluating on a random third does not land on one number, and holding it to a point "
        "would call a correct value unexplained.",
        "",
        "**The training knobs barely matter and the evaluation knobs dominate**, which is the "
        "useful result. Cutting the training set or thinning the ensemble moves the score by "
        "about what the [resolution study](power.md) says a difference needs to be worth "
        "reporting at all. Changing *what gets scored* moves it by ten times that. A study "
        "documenting its hyperparameters and not its evaluation population has documented the "
        "part that does not matter.",
        "",
        "**Averaging over batches deserves its own sentence**, because it is the one that looks "
        "like an approximation and is not. PR-AUC depends on prevalence, and each time-ordered "
        "batch has its own; a mean of per-batch scores is therefore a different estimand from the "
        "pooled score rather than a noisier version of it. Two studies can disagree on this row "
        "while both being exactly right.",
        "",
        "## Attribution",
        "",
        "| stated | reached by | narrowest rung that reaches it | rungs that do | verdict |",
        "|---|---|---|---|---|",
    ]
    for matched in study.attributions:
        if not matched.explained:
            verdict = f"**unexplained** (nearest misses by {matched.gap:+.3f})"
        elif matched.specific:
            verdict = "**pinned**"
        else:
            verdict = "bracketed"
        lines.append(
            f"| {matched.value:.3f} | {matched.shown} | {matched.knob} | "
            f"{matched.candidates} | {verdict} |"
        )
    lines += [
        "",
        "Three verdicts, and the middle one is the honest addition. **Pinned** means exactly one "
        "rung reaches the value, which is as close to an attribution as this method gets. "
        "**Bracketed** means several do: the value is consistent with more than one methodology "
        "choice and the ladder cannot say which, so calling it explained would be the same error "
        "as calling a wide confidence interval a result. **Unexplained** is the one worth acting "
        "on, because it means two reports differ for a reason nobody chose.",
        "",
        "Attribution takes the *narrowest* covering rung rather than the nearest, which matters "
        "once any rung is an interval: a wide enough range contains everything, and picking by "
        "distance alone would let it explain every value in the table. A point rung landing "
        "within tolerance is a much stronger claim than an interval that happens to contain the "
        "number, and the ranking says so.",
        "",
        "## Scope and honest limits",
        "",
        "- **This proves compatibility, not identity.** That a knob reproduces a value means the "
        "difference *could* have that cause. Establishing that it *does* would require reading "
        "each study's configuration, which is a human check this cannot replace.",
        "- **Only one quantity is audited.** The incumbent's PR-AUC is the most frequently "
        "restated number here, which makes it the right place to start and leaves detection "
        "rates, coverage levels and latencies unchecked.",
        "- **Text harvesting is approximate at both ends.** A report that states the incumbent's "
        "score in a table cell with no nearby qualifier is missed; a report that mentions a "
        "baseline in passing near an unrelated number could be over-collected. The band and the "
        "sixty-character window are what keep the second failure rare, at the cost of the first.",
        "- **This report excludes itself from the corpus it audits.** It quotes other reports' "
        "numbers in its own tables, so leaving it in would let a previous run's output be "
        "harvested as fresh evidence -- a study reading its own output is measuring itself.",
        "- **The ladder is not exhaustive.** It contains the knobs this repository's studies "
        "actually turn. A value it cannot reproduce may be a knob nobody thought to add rather "
        "than a genuine disagreement, which is why the verdict is *unexplained* rather than "
        "*wrong*.",
        "- **Recomputation costs a refit per rung.** The numbers here come from the shipped "
        "configuration on the full split; running this under a reduced CI config produces a "
        "different canonical value and therefore a different ladder, which is the study's own "
        "point applied to itself.",
    ]
    return "\n".join(lines) + "\n"


def run_consistency_report(settings: Settings) -> Path:
    """Run the cross-report audit and write the report."""
    study = run_consistency_study(settings)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study), encoding="utf-8")
    logger.info("Wrote consistency report", extra={"path": str(out_path)})

    with track_run(settings, "consistency") as run:
        run.log_params({"reports": str(study.n_reports)})
        run.log_metrics(
            {
                "quotations": float(len(study.quotations)),
                "distinct_values": float(len(study.distinct())),
                "spread": study.range(),
                "canonical": study.canonical,
                "unexplained": float(len(study.unexplained())),
            }
        )
        run.log_artifact(out_path)
    return out_path
