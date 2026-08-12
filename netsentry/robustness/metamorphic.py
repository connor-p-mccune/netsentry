"""Metamorphic testing: a correctness oracle for a detector that has no ground truth.

Every quality claim in this repo is settled by comparing predictions to labels. In production
that comparison is unavailable — nobody labels the live stream — so the only bugs a deployment
can catch are the ones that happen to move an aggregate it can still compute. That leaves a
whole class of defect invisible: the ones that are *correct on average* and wrong per request.

Metamorphic testing (Chen, Cheung & Yiu 1998; Xie et al. 2011 for machine learning) removes the
need for labels by testing **relations between outputs** instead of outputs themselves. Pick a
transformation of the input that must not change the answer — re-time a flow to a different
exporter clock resolution, round its fields to the precision a JSON payload survives, send it in
a different batch position — and the requirement becomes checkable on unlabeled traffic: the two
verdicts must agree. A disagreement is a bug, and it is a bug regardless of which verdict was
right.

The relations come in two kinds, and conflating them is what makes this technique look either
trivial or unreliable. **Structural** relations transform the input into *the same input* — a
different batch position, a different batch size, a different column order — so the scores must
be bit-identical and any deviation at all is a code defect. **Semantic** relations transform it
into a *different record of the same behaviour* — a different exporter clock, a lossier
serialiser — so a deviation is not a bug but a statement about what the model keys on. Only the
structural half can serve as a bug oracle; the semantic half is a fragility measurement, and on
this model it produces a genuinely uncomfortable finding about the exporter's clock.

The second half of the report is what makes the first half credible. Nine **mutants** of the
serving path are injected — a swapped column pair, a per-request scaler, per-request rank
normalisation, zero-filled missing fields, a float16 cast, a millisecond/microsecond unit slip,
an uninverted shuffle, an under-trained model — and each is put in front of *three* oracles: the
labelled accuracy check this project uses everywhere else, the unlabelled structural relations,
and the load-time canary (scores on pinned flows against a recorded reference). The resulting
kill matrix is the point of the report, and it cuts in three directions rather than one. Labels
find a model that is worse but cannot run in production. Invariants find an implementation that
is inconsistent but cannot see a model that is uniformly wrong. A recorded reference finds a
change that is neither — consistent, accuracy-neutral and wrong — but needs a prior artefact to
compare against. No oracle subsumes another, which is the argument for running all three.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.feature_sets import feature_group
from netsentry.features.pipeline import build_pipeline, feature_frame
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from netsentry.config import Settings
    from netsentry.config.settings import MetamorphicConfig

logger = get_logger(__name__)

REPORT_NAME = "metamorphic.md"
FIGURE_NAME = "metamorphic_kills.png"

ScoreFn = Callable[[pd.DataFrame], np.ndarray]


# --------------------------------------------------------------------------------------
# Input transformations. Each is semantics-preserving by construction: it changes how a flow
# is *recorded*, never what the flow did.
# --------------------------------------------------------------------------------------


def timing_columns(columns: list[str]) -> list[str]:
    """Columns measured in units of time — durations, inter-arrival times, idle/active spans."""
    return [c for c in columns if feature_group(c) == "timing/IAT"]


def rate_columns(columns: list[str]) -> list[str]:
    """Columns measured per second, which move inversely to the clock scale."""
    return [c for c in columns if feature_group(c) == "flow rates"]


def rescale_clock(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Re-time a flow as a different exporter clock would have recorded it.

    Durations and inter-arrival times scale with the clock; per-second rates scale inversely;
    counts, byte totals, sizes and flags are untouched. The result describes the *same* flow,
    so any verdict change is a property of the model, not of the traffic.
    """
    if factor <= 0:
        raise ValueError("clock factor must be positive")
    out = frame.copy()
    cols = list(frame.columns)
    for col in timing_columns(cols):
        out[col] = frame[col] * factor
    for col in rate_columns(cols):
        out[col] = frame[col] / factor
    return out


def round_significant(frame: pd.DataFrame, digits: int) -> pd.DataFrame:
    """Round every numeric field to `digits` significant figures (a lossy serialisation)."""
    if digits < 1:
        raise ValueError("digits must be at least 1")
    out = frame.copy()
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.where(values == 0, 0.0, np.floor(np.log10(np.abs(values))))
        magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.power(10.0, digits - 1 - magnitude)
        rounded = np.round(values * scale) / scale
        out[col] = np.where(np.isfinite(values), rounded, values)
    return out


def verdict_flip_rate(before: np.ndarray, after: np.ndarray, threshold: float) -> float:
    """Share of flows whose alert/no-alert decision changes across the transformation."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0
    return float(np.mean((a[mask] >= threshold) != (b[mask] >= threshold)))


def max_score_delta(before: np.ndarray, after: np.ndarray) -> float:
    """Largest absolute score movement across the transformation (0 means bit-identical)."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])))


# --------------------------------------------------------------------------------------
# The relations.
# --------------------------------------------------------------------------------------


@dataclass
class Relation:
    """One metamorphic relation: a transformation, and the invariant it must not break.

    ``kind`` separates two genuinely different claims that are easy to conflate.
    A **structural** relation is a property of the *implementation* — the transformed input is
    the same input, so the scores must be bit-identical, and any deviation at all is a code
    defect. A **semantic** relation is a property of the *model* — the transformed input is a
    different but equivalent record of the same behaviour, so a deviation means the model has
    learned a dependence on something that should not matter. Only structural relations can
    serve as a bug oracle; semantic ones are a fragility measurement.
    """

    name: str
    kind: str
    statement: str
    rationale: str
    evaluate: Callable[[ScoreFn, pd.DataFrame, np.random.Generator, int], np.ndarray]


def _rel_clock(factor: float) -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        return score(rescale_clock(frame, factor))

    return Relation(
        name=f"clock rescale (x{factor:g})",
        kind="semantic",
        statement="re-timing a flow to a different exporter clock keeps its verdict",
        rationale=(
            "durations scale with the clock and rates scale inversely, so the record describes "
            "the same traffic; the exporter's timing resolution is not part of the threat"
        ),
        evaluate=evaluate,
    )


def _rel_precision(digits: int) -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        return score(round_significant(frame, digits))

    return Relation(
        name=f"precision rounding ({digits} s.f.)",
        kind="semantic",
        statement="a flow rounded to the precision a JSON payload survives keeps its verdict",
        rationale=(
            "the API serialises floats; a verdict that depends on the digits a serialiser drops "
            "is a verdict that depends on the transport"
        ),
        evaluate=evaluate,
    )


def _rel_permutation() -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        order = rng.permutation(len(frame))
        shuffled = score(frame.iloc[order])
        out = np.empty(len(frame), dtype=float)
        out[order] = shuffled
        return out

    return Relation(
        name="batch permutation",
        kind="structural",
        statement="a flow's verdict does not depend on its position in the batch",
        rationale="batching is a transport detail; a per-row verdict must be a per-row function",
        evaluate=evaluate,
    )


def _rel_duplication() -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        doubled = score(pd.concat([frame, frame], ignore_index=True))
        return np.asarray(doubled[: len(frame)], dtype=float)

    return Relation(
        name="batch duplication",
        kind="structural",
        statement="doubling the batch does not change the verdicts already in it",
        rationale="batch size is a throughput choice, not an input to the decision",
        evaluate=evaluate,
    )


def _rel_single_vs_batch() -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        for i in range(min(cap, len(frame))):
            out[i] = float(score(frame.iloc[[i]])[0])
        return out

    return Relation(
        name="single vs batch",
        kind="structural",
        statement="a flow scored alone gets the score it gets inside a full batch",
        rationale=(
            "the offline evaluation always scores one large batch and the API always scores one "
            "flow, so any batch-dependent step is a divergence no offline metric can observe"
        ),
        evaluate=evaluate,
    )


def _rel_column_order() -> Relation:
    def evaluate(
        score: ScoreFn, frame: pd.DataFrame, rng: np.random.Generator, cap: int
    ) -> np.ndarray:
        cols = list(frame.columns)
        rng.shuffle(cols)
        return score(frame[cols])

    return Relation(
        name="column reorder",
        kind="structural",
        statement="the feature contract is by name, not by position",
        rationale=(
            "a caller that emits the same fields in a different order is a conforming caller; "
            "if order matters, the pipeline is not the thing enforcing the contract"
        ),
        evaluate=evaluate,
    )


def build_relations(cfg: MetamorphicConfig) -> list[Relation]:
    """The relation suite, in the order the report presents it."""
    relations = [_rel_clock(f) for f in cfg.clock_factors]
    relations.append(_rel_precision(cfg.significant_digits))
    relations.extend(
        [_rel_permutation(), _rel_duplication(), _rel_single_vs_batch(), _rel_column_order()]
    )
    return relations


# --------------------------------------------------------------------------------------
# The mutants: defects injected into the serving path, each a real production failure.
# --------------------------------------------------------------------------------------


@dataclass
class Mutant:
    """A defect injected between the request and the score."""

    name: str
    description: str
    wrap: Callable[[ScoreFn], ScoreFn]


def _mutant_none() -> Mutant:
    return Mutant("none (control)", "the deployed path, unmodified", lambda fn: fn)


def _matrix_mutant(
    name: str, description: str, transform: Callable[[np.ndarray], np.ndarray]
) -> Callable[[Pipeline, SupervisedClassifier, str], Mutant]:
    """A mutant that corrupts the feature matrix between the pipeline and the model."""

    def build(pipeline: Pipeline, model: SupervisedClassifier, benign: str) -> Mutant:
        def wrap(_: ScoreFn) -> ScoreFn:
            def score(frame: pd.DataFrame) -> np.ndarray:
                x = transform(np.asarray(pipeline.transform(frame), dtype=float))
                return attack_probability(
                    np.asarray(model.predict_proba(x)), model.classes_, benign
                )

            return score

        return Mutant(name, description, wrap)

    return build


def _swap_columns(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    if out.shape[1] >= 2:
        out[:, [0, 1]] = out[:, [1, 0]]
    return out


def _per_batch_scale(x: np.ndarray) -> np.ndarray:
    """Standardise using *this request's* statistics instead of the fitted training ones."""
    mean: np.ndarray = x.mean(axis=0, keepdims=True)
    std: np.ndarray = x.std(axis=0, keepdims=True)
    scaled: np.ndarray = (x - mean) / np.where(std < 1e-9, 1.0, std)
    return scaled


def _float16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float64)


def _roll(x: np.ndarray) -> np.ndarray:
    return np.roll(x, 1, axis=1)


def _shuffle_rows_unrestored(x: np.ndarray) -> np.ndarray:
    """Sort the batch for 'cache locality' and forget to invert the permutation."""
    if x.shape[1] == 0:
        return x
    return x[np.argsort(x[:, 0], kind="stable")]


# --------------------------------------------------------------------------------------
# Results.
# --------------------------------------------------------------------------------------


@dataclass
class RelationResult:
    """One relation checked against one scoring path."""

    relation: str
    kind: str
    statement: str
    rationale: str
    flip_rate: float
    max_delta: float

    @property
    def holds(self) -> bool:
        """Structural relations must be exact; semantic ones only have to preserve verdicts."""
        if self.kind == "structural":
            return self.max_delta == 0.0
        return self.flip_rate == 0.0


@dataclass
class MutantResult:
    """One mutant, put in front of all three oracles."""

    name: str
    description: str
    pr_auc: float
    pr_auc_delta: float
    caught_by_accuracy: bool
    caught_by_canary: bool
    canary_delta: float
    worst_flip_rate: float
    violated: list[str]

    @property
    def caught_by_metamorphic(self) -> bool:
        return bool(self.violated)

    @property
    def caught_by_any(self) -> bool:
        return self.caught_by_accuracy or self.caught_by_metamorphic or self.caught_by_canary


@dataclass
class MetamorphicStudy:
    """The relation suite on the deployed path, plus the three-oracle mutation kill matrix."""

    n_flows: int
    n_canary: int
    threshold: float
    primary_fpr: float
    baseline_pr_auc: float
    accuracy_tolerance: float
    canary_tolerance: float
    relations: list[RelationResult]
    mutants: list[MutantResult]

    @property
    def structural(self) -> list[RelationResult]:
        return [r for r in self.relations if r.kind == "structural"]

    @property
    def semantic(self) -> list[RelationResult]:
        return [r for r in self.relations if r.kind == "semantic"]


def check_relations(
    score: ScoreFn,
    frame: pd.DataFrame,
    relations: list[Relation],
    threshold: float,
    seed: int,
    single_cap: int,
) -> list[RelationResult]:
    """Score the frame once, then re-score it under each relation and compare."""
    base = np.asarray(score(frame), dtype=float)
    results = []
    for relation in relations:
        rng = np.random.default_rng(seed)
        after = np.asarray(relation.evaluate(score, frame, rng, single_cap), dtype=float)
        results.append(
            RelationResult(
                relation=relation.name,
                kind=relation.kind,
                statement=relation.statement,
                rationale=relation.rationale,
                flip_rate=verdict_flip_rate(base, after, threshold),
                max_delta=max_score_delta(base, after),
            )
        )
    return results


def run_metamorphic(settings: Settings) -> MetamorphicStudy:
    """Check the relation suite on the deployed path, then run the mutation study."""
    cfg: MetamorphicConfig = settings.metamorphic
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "multiclass"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    y_train = train[MULTICLASS_TARGET].to_numpy()
    x_val = np.asarray(pipeline.transform(val))
    model = SupervisedClassifier(variant).fit(
        x_train, y_train, eval_set=(x_val, val[MULTICLASS_TARGET].to_numpy())
    )

    def deployed(frame: pd.DataFrame) -> np.ndarray:
        x = np.asarray(pipeline.transform(frame))
        return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

    # The deployed operating point: the threshold this project would actually ship.
    threshold = threshold_at_fpr(
        val[BINARY_TARGET].to_numpy().astype(int),
        deployed(val),
        variant.thresholds.primary_fpr,
    )

    # Relations are checked on unlabelled traffic — the labels below are only for the
    # *other* oracle, and are never visible to the metamorphic suite.
    probe = feature_frame(test, variant)
    if len(probe) > cfg.max_rows:
        probe = probe.sample(n=cfg.max_rows, random_state=variant.seed).sort_index()
    y_probe = test.loc[probe.index, BINARY_TARGET].to_numpy().astype(int)

    relations = build_relations(cfg)
    relation_results = check_relations(
        deployed, probe, relations, threshold, variant.seed, cfg.max_single_rows
    )
    baseline_pr_auc = float(average_precision_score(y_probe, deployed(probe)))

    # The third oracle: a handful of flows whose scores were recorded from a trusted build, in
    # the shape of the serving canary this repo already ships. It needs no labels but it does
    # need a prior artefact to compare against -- which is exactly the dependency the other two
    # oracles do not have.
    canary_rows = probe.iloc[: cfg.canary_rows]
    canary_reference = deployed(canary_rows)

    # Only structural relations are usable as a bug oracle: a violation there is a defect in the
    # implementation, full stop. Semantic violations are a statement about the model.
    structural_names = {r.name for r in relations if r.kind == "structural"}

    mutants = _build_mutants(variant, pipeline, model, benign, x_train, y_train, x_val, val)
    mutant_results = []
    for mutant in mutants:
        score = mutant.wrap(deployed)
        pr_auc = float(average_precision_score(y_probe, score(probe)))
        checked = check_relations(
            score, probe, relations, threshold, variant.seed, cfg.max_single_rows
        )
        violated = [r.relation for r in checked if r.relation in structural_names and not r.holds]
        canary_delta = max_score_delta(canary_reference, score(canary_rows))
        mutant_results.append(
            MutantResult(
                name=mutant.name,
                description=mutant.description,
                pr_auc=pr_auc,
                pr_auc_delta=pr_auc - baseline_pr_auc,
                caught_by_accuracy=(baseline_pr_auc - pr_auc) > cfg.accuracy_tolerance,
                caught_by_canary=canary_delta > cfg.canary_tolerance,
                canary_delta=canary_delta,
                worst_flip_rate=max(
                    (r.flip_rate for r in checked if r.relation in structural_names), default=0.0
                ),
                violated=violated,
            )
        )

    return MetamorphicStudy(
        n_flows=len(probe),
        n_canary=len(canary_rows),
        threshold=threshold,
        primary_fpr=variant.thresholds.primary_fpr,
        baseline_pr_auc=baseline_pr_auc,
        accuracy_tolerance=cfg.accuracy_tolerance,
        canary_tolerance=cfg.canary_tolerance,
        relations=relation_results,
        mutants=mutant_results,
    )


def _build_mutants(
    settings: Settings,
    pipeline: Pipeline,
    model: SupervisedClassifier,
    benign: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    val: pd.DataFrame,
) -> list[Mutant]:
    """Every injected defect, each modelled on a failure that has shipped somewhere."""
    matrix_mutants = [
        _matrix_mutant(
            "swapped column pair",
            "two feature columns transposed between training and serving order",
            _swap_columns,
        ),
        _matrix_mutant(
            "per-request scaler",
            "standardised with the request's own mean/std instead of the fitted training ones",
            _per_batch_scale,
        ),
        _matrix_mutant(
            "float16 cast", "the feature matrix narrowed to half precision in transit", _float16
        ),
        _matrix_mutant(
            "rolled feature vector",
            "an off-by-one in the feature assembly shifts every column by one",
            _roll,
        ),
        _matrix_mutant(
            "uninverted sort",
            "the batch is sorted for locality and the permutation is never undone",
            _shuffle_rows_unrestored,
        ),
    ]
    mutants = [_mutant_none()]
    mutants += [build(pipeline, model, benign) for build in matrix_mutants]

    # The archetype of a defect no ranking metric can see: the serving path replaces the score
    # with its percentile inside the request. Average precision and ROC-AUC are invariant to any
    # monotone transform of the scores, so on one large offline batch this is *exactly* a no-op;
    # on a single-flow request the percentile is 0 or 1 and the verdict is arbitrary.
    def rank_normalise(inner: ScoreFn) -> ScoreFn:
        def score(frame: pd.DataFrame) -> np.ndarray:
            raw = np.asarray(inner(frame), dtype=float)
            order = np.argsort(np.argsort(raw, kind="stable"), kind="stable")
            return order / max(len(raw) - 1, 1)

        return score

    mutants.append(
        Mutant(
            "per-request rank normalisation",
            "the score is replaced by its percentile within the request",
            rank_normalise,
        )
    )

    # A frame-level mutant: the API fills a missing field with 0 instead of letting the fitted
    # imputer supply the training median.
    def zero_fill(_: ScoreFn) -> ScoreFn:
        def score(frame: pd.DataFrame) -> np.ndarray:
            x = np.asarray(pipeline.transform(frame.fillna(0.0)))
            return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

        return score

    mutants.append(
        Mutant(
            "zero-filled missing fields",
            "absent fields defaulted to 0 rather than imputed with the training median",
            zero_fill,
        )
    )

    # A unit slip in the exporter: milliseconds where the model was trained on microseconds.
    def unit_slip(_: ScoreFn) -> ScoreFn:
        def score(frame: pd.DataFrame) -> np.ndarray:
            x = np.asarray(pipeline.transform(rescale_clock(frame, 1e-3)))
            return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

        return score

    mutants.append(
        Mutant(
            "exporter unit slip",
            "the exporter switched to milliseconds; the model was trained on microseconds",
            unit_slip,
        )
    )

    # A genuinely worse but perfectly consistent model: last month's under-trained artefact.
    rng = np.random.default_rng(settings.seed)
    keep = rng.choice(
        len(x_train),
        size=max(50, int(len(x_train) * settings.metamorphic.stale_fraction)),
        replace=False,
    )
    stale = SupervisedClassifier(settings).fit(
        x_train[keep], y_train[keep], eval_set=(x_val, val[MULTICLASS_TARGET].to_numpy())
    )

    def stale_model(_: ScoreFn) -> ScoreFn:
        def score(frame: pd.DataFrame) -> np.ndarray:
            x = np.asarray(pipeline.transform(frame))
            return attack_probability(np.asarray(stale.predict_proba(x)), stale.classes_, benign)

        return score

    mutants.append(
        Mutant(
            "under-trained model",
            f"an artefact fit on {settings.metamorphic.stale_fraction:.0%} of the training rows",
            stale_model,
        )
    )
    return mutants


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def run_metamorphic_report(settings: Settings) -> Path:
    """Run the metamorphic study and write the report + figure."""
    study = run_metamorphic(settings)

    injected = [m for m in study.mutants if not m.name.startswith("none")]
    # How many of the three oracles each defect trips: the point is that no column is full.
    fig = plots.plot_barh(
        labels=[m.name for m in injected],
        values=[
            float(m.caught_by_accuracy + m.caught_by_metamorphic + m.caught_by_canary)
            for m in injected
        ],
        xlabel="oracles that catch the defect (accuracy / metamorphic / canary)",
        title="No single oracle catches every defect",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xmax=3.0,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote metamorphic report", extra={"path": str(out_path)})

    with track_run(settings, "metamorphic") as run:
        run.log_metrics(
            {
                "relations_held": float(sum(r.holds for r in study.relations)),
                "relations_total": float(len(study.relations)),
                "mutants_caught_metamorphic": float(sum(m.caught_by_metamorphic for m in injected)),
                "mutants_caught_accuracy": float(sum(m.caught_by_accuracy for m in injected)),
                "mutants_caught_canary": float(sum(m.caught_by_canary for m in injected)),
                "mutants_total": float(len(injected)),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _relation_table(results: list[RelationResult]) -> str:
    rows = [
        "| relation | the invariant | verdict flips | max score delta | holds? |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        mark = "yes" if r.holds else "**no**"
        rows.append(
            f"| {r.relation} | {r.statement} | {r.flip_rate:.2%} | {r.max_delta:.2e} | {mark} |"
        )
    return "\n".join(rows)


def _mutant_table(study: MetamorphicStudy) -> str:
    rows = [
        "| injected defect | PR-AUC (delta) | labelled accuracy | metamorphic | canary | "
        "structural relation broken |",
        "|---|---|---|---|---|---|",
    ]
    for m in study.mutants:
        acc = "caught" if m.caught_by_accuracy else "missed"
        meta = "caught" if m.caught_by_metamorphic else "missed"
        canary = "caught" if m.caught_by_canary else "missed"
        first = m.violated[0] if m.violated else "—"
        rows.append(
            f"| {m.name} | {m.pr_auc:.3f} ({m.pr_auc_delta:+.3f}) | {acc} | {meta} | {canary} "
            f"| {first} |"
        )
    return "\n".join(rows)


def _structural_read(study: MetamorphicStudy) -> str:
    broken = [r for r in study.structural if not r.holds]
    if not broken:
        return (
            f"**Every structural relation holds exactly.** Across {study.n_flows:,} unlabelled "
            "flows, permuting the batch, doubling it, splitting it down to single-flow requests "
            "and shuffling the column order all return **bit-identical** scores — max deviation "
            "0.00e+00, not merely a small one. That is a real property of this design rather "
            "than luck: preprocessing lives inside one fitted pipeline that is bundled with the "
            "model and selects its columns by name, so the serving path has no opportunity to "
            "re-derive a statistic from the request or to assume a position. The single-vs-batch "
            "result is the one worth keeping: it is the direct evidence that the API and the "
            "offline evaluation are computing the same function."
        )
    worst = max(broken, key=lambda r: r.max_delta)
    return (
        f"**{len(broken)} of {len(study.structural)} structural relations fail**, the worst being "
        f"*{worst.relation}* at a {worst.max_delta:.2e} score deviation and a "
        f"{worst.flip_rate:.2%} verdict-flip rate. A structural violation is unambiguous: the "
        "transformed input is the *same* input, so the implementation is not computing a "
        "per-flow function of the flow."
    )


def _semantic_read(study: MetamorphicStudy) -> str:
    broken = [r for r in study.semantic if not r.holds]
    if not broken:
        return (
            "No semantic relation fires either: re-timing a flow and rounding its fields leave "
            "every verdict intact, so the model is not keying on the exporter's clock or on "
            "digits a serialiser would drop."
        )
    worst = max(broken, key=lambda r: r.flip_rate)
    return (
        f"**The model is not invariant to its own exporter's clock.** Re-timing every flow by "
        f"{worst.relation.split('x')[-1].rstrip(')')} — durations up, rates correspondingly down, "
        f"not one byte or packet changed — flips {worst.flip_rate:.2%} of verdicts, with "
        f"individual scores moving as much as {worst.max_delta:.2f}. This is not a code defect; "
        "the implementation is doing exactly what it was asked. It is a **modelling** finding, "
        "and an uncomfortable one: some of what the detector calls attack behaviour is a "
        "statement about wall-clock pacing rather than about traffic, so two sites running "
        "exporters with different timing resolution would not receive the same verdicts on the "
        "same flows. Roughly one alert in "
        f"{round(1 / max(worst.flip_rate, 1e-9)):,} is decided by the clock. That number is "
        "exactly the kind of thing an accuracy metric cannot report, because both verdicts are "
        "scored against the same label and only one of them is ever computed."
    )


def _kill_read(study: MetamorphicStudy) -> str:
    injected = [m for m in study.mutants if not m.name.startswith("none")]
    meta_only = [m for m in injected if m.caught_by_metamorphic and not m.caught_by_accuracy]
    acc_only = [m for m in injected if m.caught_by_accuracy and not m.caught_by_metamorphic]
    canary_only = [
        m
        for m in injected
        if m.caught_by_canary and not (m.caught_by_metamorphic or m.caught_by_accuracy)
    ]
    missed = [m for m in injected if not m.caught_by_any]

    parts = [
        f"Of {len(injected)} injected defects, "
        f"{sum(m.caught_by_accuracy for m in injected)} are caught by the labelled accuracy "
        f"check, {sum(m.caught_by_metamorphic for m in injected)} by the metamorphic suite, and "
        f"{sum(m.caught_by_canary for m in injected)} by the canary. **No oracle catches them "
        "all, and each has a blind spot the others cover.**"
    ]
    if meta_only:
        m = meta_only[0]
        parts.append(
            f"*The labelled oracle misses* **{m.name}**, and the metamorphic suite catches it — "
            f"it moves PR-AUC by "
            f"{m.pr_auc_delta:+.3f}, and in the rank-normalisation case that is not an "
            "approximation but an identity: average precision and ROC-AUC are invariant to any "
            "monotone transform of the scores, so replacing a score with its percentile inside "
            "the batch is provably invisible to every ranking metric this project reports. The "
            "offline evaluation scores one batch of thousands; the API scores one flow, whose "
            f"percentile is 0 or 1. The *{m.violated[0]}* relation reports it immediately."
        )
    if acc_only:
        m = max(acc_only, key=lambda x: abs(x.pr_auc_delta))
        parts.append(
            f"*The metamorphic suite misses* **{m.name}**: it costs {abs(m.pr_auc_delta):.3f} "
            "PR-AUC "
            "while breaking no relation at all, because it is a perfectly consistent function of "
            "its input — just a worse one. This is the boundary of the technique, and it is worth "
            "stating sharply: metamorphic testing establishes that a system is *self-consistent*, "
            "never that it is *right*. A uniformly wrong model satisfies every invariance here."
        )
    if canary_only:
        m = canary_only[0]
        parts.append(
            f"*Only the canary catches* **{m.name}** ({m.canary_delta:.2e} deviation on "
            f"{study.n_canary} pinned flows): a consistent, accuracy-neutral corruption is "
            "invisible to both of the others, and can only be found by comparing against scores "
            "recorded from a build that was already trusted. That is the third oracle's whole "
            "value, and also its cost — it is the only one of the three that needs a prior "
            "artefact to compare against."
        )
    if missed:
        names = ", ".join(f"**{m.name}**" for m in missed)
        parts.append(
            f"{names} slips past all three, which is worth stating plainly: this is a lower "
            "bound on defect detection, not a proof of correctness."
        )
    return " ".join(parts)


def _render(study: MetamorphicStudy, fig: Path) -> str:
    injected = [m for m in study.mutants if not m.name.startswith("none")]
    covered = sum(m.caught_by_any for m in injected)
    return f"""# NetSentry — Metamorphic Testing: a Correctness Oracle Without Labels

_Synthetic stand-in. Honest temporal split; {study.n_flows:,} test flows used **unlabelled** for
the relation checks. Verdicts taken at the deployed threshold ({study.threshold:.4f}, the
{study.primary_fpr:.1%}-FPR operating point picked on validation). Baseline PR-AUC
{study.baseline_pr_auc:.3f}._

## Why this report exists

Every other number in this repo is produced by comparing a prediction to a label. Production
has no labels, so a deployed detector can only notice the defects that happen to move an
aggregate — and a large class of real defect does not. A step that normalises per request, a
serialiser that drops digits, an exporter that switches units: each of these can be *correct on
average* and wrong on the specific request an analyst is looking at.

Metamorphic testing (Chen, Cheung & Yiu 1998; Xie et al. 2011 for ML) removes the label from the
loop by testing **relations between outputs** rather than outputs themselves. If a transformation
of the input cannot change the right answer, then the two answers must agree — and that is
checkable on traffic nobody has labelled. Each relation below is grounded in something an
exporter or a serving path actually does.

Two kinds of relation are checked, and conflating them is the mistake that makes this technique
look either trivial or unreliable. A **structural** relation transforms the input into *the same
input* — a different batch position, a different column order — so the scores must be
bit-identical and any deviation is a code defect. A **semantic** relation transforms the input
into a *different record of the same behaviour* — a different exporter clock, a lossier
serialiser — so a deviation is not a bug in the code but a statement about what the model has
learned to depend on.

## Structural relations: is the implementation computing a per-flow function?

{_relation_table(study.structural)}

{_structural_read(study)}

## Semantic relations: what has the model learned to depend on?

{_relation_table(study.semantic)}

{_semantic_read(study)}

## The mutation study: what does each oracle actually catch?

A relation suite that never fires proves nothing on its own — it could be checking invariants no
realistic bug would break. So {len(injected)} defects are injected into the serving path, each
modelled on a failure that has shipped somewhere, and each is put in front of **three** oracles:

1. **labelled accuracy** — the check this project uses everywhere else; caught when PR-AUC drops
   by more than {study.accuracy_tolerance:.3f}. Needs labels, so it cannot run in production.
2. **the metamorphic suite** — caught when any *structural* relation is violated. Needs nothing
   but traffic, so it can run continuously against the live stream.
3. **the canary** — the load-time behavioural self-test this repo already ships: scores on
   {study.n_canary} pinned flows compared against a recorded reference, caught above
   {study.canary_tolerance:.0e}. Needs a prior trusted artefact.

{_mutant_table(study)}

{_kill_read(study)}

![Oracles that catch each defect](../figures/{fig.name})

Between them the three oracles cover {covered}/{len(injected)} defects, and the reason to run all
three is visible in the table rather than argued: they fail on **disjoint** kinds of bug. Labels
find a model that is worse. Invariants find an implementation that is inconsistent. A recorded
reference finds a change that is neither — consistent, accuracy-neutral, and wrong. A deployment
that only ships the first of the three is blind to the other two categories for the whole time
it is running, which is precisely when it matters.

## Scope

Metamorphic relations are **necessary, not sufficient**, and the kill matrix shows exactly where
the boundary sits: they constrain self-consistency, never correctness. The clock-rescale relation
assumes re-timing is semantics-preserving, which holds for a change of exporter resolution and
would *not* hold for a large dilation — a flow slowed tenfold is arguably a different flow — so
the factors are deliberately kept near unity. The natural next family is directional rather than
invariant (adding bytes must not lower suspicion), which the [monotone-constraint
study](monotonic.md) enforces structurally instead of testing after the fact. The
[sensor-failure study](degradation.md) covers the complementary case where the input is genuinely
corrupt rather than merely re-expressed, and the [predictive-multiplicity
study](multiplicity.md) asks the adjacent question of how much the verdict depends on *which*
equally-good model was fitted rather than on how it was called."""
