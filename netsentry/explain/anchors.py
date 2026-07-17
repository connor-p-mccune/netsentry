"""Anchors: high-precision IF-THEN rules that explain a verdict with a guarantee (Ribeiro 2018).

The explainability suite answers many questions but not the one a SOC analyst asks out loud:
"give me a *rule* I can trust." SHAP attributes a verdict additively across features;
partial dependence draws each feature's response shape; the counterfactual finds the smallest
change that would flip the verdict; exemplars point at similar past flows. None of them state a
**sufficient condition**: a compact set of predicates such that, whenever they hold, the model
almost always returns this same verdict.

That is exactly an *anchor* (Ribeiro, Singh & Guestrin, AAAI 2018, from the authors of LIME).
For a flagged flow, an anchor is a conjunction of feature predicates ``A`` — e.g.
"Flow Packets/s >= p AND Flow Duration <= q" — with high **precision**: among flows that
satisfy ``A``, the model agrees with this verdict at least ``tau`` of the time. Of the many
high-precision rules, the useful one also has high **coverage**: it applies to as much of the
traffic as possible, so it generalises beyond the single flow. Precision is the guarantee;
coverage is what makes the guarantee worth stating.

The construction here is faithful to the paper's core on tabular data: each candidate feature
is discretised into quantile bins, a predicate pins a feature to the flagged flow's own bin,
and a greedy search grows the conjunction — adding at each step the predicate that most raises
precision, estimated on a background sample of real flows that satisfy the current rule — until
a lower confidence bound on precision clears ``tau`` or the rule reaches its length budget.
Every reported anchor's precision is then **re-measured on a held-out background** it was not
grown against, so the guarantee is validated, not just fit — the same measure-then-check
discipline the anomaly-attribution and data-valuation studies use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "anchors.md"
FIGURE_NAME = "anchors.png"


@dataclass
class Anchor:
    """One flow's anchor: the predicate features, its precision, and its coverage."""

    features: list[int]  # column indices pinned to the flow's bin, in the order chosen
    precision: float  # model-agreement among background rows satisfying the rule
    precision_lcb: float  # lower confidence bound on precision (the guarantee side)
    coverage: float  # fraction of background satisfying the rule
    trajectory: list[tuple[int, float, float]] = field(default_factory=list)  # (len, prec, cov)


def _precision_lcb(hits: int, n: int, z: float) -> float:
    """Normal lower confidence bound on a proportion (the anchor's precision guarantee side)."""
    if n == 0:
        return 0.0
    p = hits / n
    return max(0.0, p - z * float(np.sqrt(p * (1.0 - p) / n)))


def greedy_anchor(
    x_bins: np.ndarray,
    background_bins: np.ndarray,
    background_class: np.ndarray,
    x_class: int,
    *,
    tau: float,
    max_predicates: int,
    min_match: int,
    z: float,
) -> Anchor:
    """Grow a high-precision anchor greedily for one flow (pure; operates on discretised bins).

    ``x_bins`` is the flow's per-feature bin id; ``background_bins`` is ``(n, n_features)`` of
    the same discretisation over a reference sample; ``background_class`` is the model's
    predicted class for each background row. At each step every not-yet-used feature is trialled
    by pinning it to the flow's bin, precision is estimated on the rows that satisfy the growing
    rule, and the feature with the highest precision (subject to ``min_match`` supporting rows)
    is added. Stops when the precision lower bound clears ``tau`` or the rule is full.
    """
    n_features = background_bins.shape[1]
    chosen: list[int] = []
    mask = np.ones(len(background_bins), dtype=bool)  # rows satisfying the current rule
    base = float(np.mean(background_class == x_class)) if len(background_class) else 0.0
    coverage = float(np.mean(mask))
    trajectory: list[tuple[int, float, float]] = [(0, base, coverage)]
    precision, precision_lcb = base, _precision_lcb(
        int(np.sum(background_class == x_class)), len(background_class), z
    )

    while len(chosen) < max_predicates:
        best: tuple[float, int, np.ndarray] | None = None
        for f in range(n_features):
            if f in chosen:
                continue
            cand = mask & (background_bins[:, f] == x_bins[f])
            n_match = int(np.sum(cand))
            if n_match < min_match:
                continue
            prec = float(np.mean(background_class[cand] == x_class))
            if best is None or prec > best[0]:
                best = (prec, f, cand)
        if best is None:  # no feature has enough support to refine further
            break
        precision, f, mask = best
        chosen.append(f)
        hits = int(np.sum(background_class[mask] == x_class))
        n_match = int(np.sum(mask))
        precision_lcb = _precision_lcb(hits, n_match, z)
        coverage = n_match / len(background_bins)
        trajectory.append((len(chosen), precision, coverage))
        if precision_lcb >= tau:
            break

    return Anchor(chosen, precision, precision_lcb, coverage, trajectory)


@dataclass
class AnchorExample:
    """A rendered anchor for one flagged flow, with the held-out precision check."""

    label: str  # the flow's true class (for context only; never used to build the rule)
    predicates: list[str]  # human-readable "lo <= feature <= hi" clauses
    precision: float
    precision_lcb: float
    coverage: float
    holdout_precision: float


@dataclass
class AnchorsStudy:
    """The anchors study: per-flow rules, the growth trajectory, and aggregate guarantees."""

    tau: float
    n_bins: int
    examples: list[AnchorExample]
    mean_precision: float
    mean_coverage: float
    mean_predicates: float
    mean_holdout_precision: float
    traj_precision: list[float]  # mean precision vs rule length
    traj_coverage: list[float]  # mean coverage vs rule length


def _finite_median(values: np.ndarray) -> float:
    """Median over the finite entries (CIC-IDS2017's Flow Bytes/s carries Inf/NaN)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else 0.0


def _clean(values: np.ndarray, median: float) -> np.ndarray:
    """Replace non-finite values with the feature's finite median (mirrors the pipeline impute)."""
    v = np.asarray(values, dtype=float)
    return np.where(np.isfinite(v), v, median)


def _digitize(
    frame: pd.DataFrame,
    features: list[str],
    edges: dict[str, np.ndarray],
    medians: dict[str, float],
) -> np.ndarray:
    """Map each row/feature to a quantile-bin id under shared edges (fit on the reference)."""
    cols = [
        np.digitize(_clean(frame[f].to_numpy(dtype=float), medians[f]), edges[f][1:-1])
        for f in features
    ]
    return np.column_stack(cols)


def _predicate_text(feature: str, bin_id: int, edges: np.ndarray) -> str:
    """Render 'feature in bin' as a raw-unit range, using open ends for the tail bins."""
    lo = edges[bin_id]
    hi = edges[bin_id + 1]
    lo_ok = np.isfinite(lo)
    hi_ok = np.isfinite(hi)
    if lo_ok and hi_ok:
        return f"{lo:.3g} <= {feature} <= {hi:.3g}"
    if hi_ok:
        return f"{feature} <= {hi:.3g}"
    return f"{feature} >= {lo:.3g}"


def _fit(
    settings: Settings,
) -> tuple[SupervisedClassifier, object, pd.DataFrame, pd.DataFrame]:
    """Fit the stratified/binary model; return it and the raw train/test frames.

    Anchors explain the classifier's own decision, so the study runs on the exchangeable
    stratified split where the decision boundary is well-populated and the held-out
    background is exchangeable with the reference — the same reason the data-valuation and
    membership studies live there. The temporal split's headline is the operating-point
    story; explaining the model's decision structure is a distributional question.
    """
    seed_everything(settings.seed)
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test").reset_index(drop=True)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    return model, pipeline, train, test


def _hard_class(
    model: SupervisedClassifier,
    pipeline: object,
    frame: pd.DataFrame,
    benign: str,
) -> np.ndarray:
    """The model's natural verdict (1 attack / 0 benign) at its decision boundary on raw rows."""
    x = np.asarray(pipeline.transform(frame))  # type: ignore[attr-defined]
    classes = np.asarray(model.model.classes_)
    scores = attack_probability(model.predict_proba(x), classes, benign)
    return (scores >= 0.5).astype(int)


def run_anchors(settings: Settings) -> AnchorsStudy:
    """Build and validate high-precision anchors for a sample of flagged flows."""
    cfg = settings.anchors
    model, pipeline, train, test = _fit(settings)
    benign = settings.labels.benign_label

    importances = getattr(model.model, "feature_importances_", None)
    transformed_names = list(pipeline.named_steps["features"].get_feature_names_out())  # type: ignore[attr-defined]
    from netsentry.features.feature_sets import display_feature_name

    ranked = (
        pd.Series(np.asarray(importances, dtype=float), index=transformed_names)
        if importances is not None
        else pd.Series(np.ones(len(transformed_names)), index=transformed_names)
    )
    raw_by_imp: list[str] = []
    for name in ranked.sort_values(ascending=False).index:
        raw = display_feature_name(name)
        if raw in train.columns and raw not in raw_by_imp:
            raw_by_imp.append(raw)
        if len(raw_by_imp) >= cfg.top_k_features:
            break
    features = raw_by_imp

    rng = np.random.default_rng(settings.seed)
    ref = train.sample(n=min(cfg.background_rows, len(train)), random_state=settings.seed)
    hold = test.sample(n=min(cfg.background_rows, len(test)), random_state=settings.seed + 1)

    medians = {f: _finite_median(ref[f].to_numpy(dtype=float)) for f in features}
    edges = {
        f: np.concatenate(
            [
                [-np.inf],
                np.quantile(
                    _clean(ref[f].to_numpy(dtype=float), medians[f]),
                    np.linspace(0, 1, cfg.n_bins + 1)[1:-1],
                ),
                [np.inf],
            ]
        )
        for f in features
    }
    ref_bins = _digitize(ref, features, edges, medians)
    hold_bins = _digitize(hold, features, edges, medians)
    ref_class = _hard_class(model, pipeline, ref, benign)
    hold_class = _hard_class(model, pipeline, hold, benign)

    # Explain flagged (predicted-attack) flows from the held-out test split.
    test_class = _hard_class(model, pipeline, test, benign)
    flagged = np.where(test_class == 1)[0]
    rng.shuffle(flagged)
    flagged = flagged[: cfg.n_explained]
    test_bins = _digitize(test.iloc[flagged], features, edges, medians)

    examples: list[AnchorExample] = []
    traj_prec: dict[int, list[float]] = {}
    traj_cov: dict[int, list[float]] = {}
    for i, row_idx in enumerate(flagged):
        anchor = greedy_anchor(
            test_bins[i],
            ref_bins,
            ref_class,
            x_class=1,
            tau=cfg.precision_threshold,
            max_predicates=cfg.max_predicates,
            min_match=cfg.min_match,
            z=cfg.confidence_z,
        )
        for length, prec, cov in anchor.trajectory:
            traj_prec.setdefault(length, []).append(prec)
            traj_cov.setdefault(length, []).append(cov)
        hold_mask = np.ones(len(hold_bins), dtype=bool)
        for f in anchor.features:
            hold_mask &= hold_bins[:, f] == test_bins[i][f]
        hold_prec = (
            float(np.mean(hold_class[hold_mask] == 1)) if int(np.sum(hold_mask)) else float("nan")
        )
        predicates = [
            _predicate_text(features[f], int(test_bins[i][f]), edges[features[f]])
            for f in anchor.features
        ]
        examples.append(
            AnchorExample(
                label=str(test.iloc[row_idx][MULTICLASS_TARGET]),
                predicates=predicates or ["(no rule cleared the support threshold)"],
                precision=anchor.precision,
                precision_lcb=anchor.precision_lcb,
                coverage=anchor.coverage,
                holdout_precision=hold_prec,
            )
        )

    lengths = sorted(traj_prec)
    valid = [e for e in examples if e.predicates and e.predicates[0].startswith("(") is False]
    hold_vals = [e.holdout_precision for e in valid if np.isfinite(e.holdout_precision)]
    study = AnchorsStudy(
        tau=cfg.precision_threshold,
        n_bins=cfg.n_bins,
        examples=examples,
        mean_precision=float(np.mean([e.precision for e in valid])) if valid else 0.0,
        mean_coverage=float(np.mean([e.coverage for e in valid])) if valid else 0.0,
        mean_predicates=float(np.mean([len(e.predicates) for e in valid])) if valid else 0.0,
        mean_holdout_precision=float(np.mean(hold_vals)) if hold_vals else 0.0,
        traj_precision=[float(np.mean(traj_prec[length])) for length in lengths],
        traj_coverage=[float(np.mean(traj_cov[length])) for length in lengths],
    )
    logger.info(
        "Anchors study",
        extra={
            "mean_precision": round(study.mean_precision, 3),
            "mean_coverage": round(study.mean_coverage, 4),
            "mean_holdout_precision": round(study.mean_holdout_precision, 3),
        },
    )
    return study


def run_anchors_report(settings: Settings) -> Path:
    """Build anchors for flagged flows, plot the precision/coverage trade-off, write the report."""
    study = run_anchors(settings)

    lengths = np.arange(len(study.traj_precision), dtype=float)
    fig = plots.plot_lines(
        {
            "precision (model agreement)": (lengths, np.array(study.traj_precision)),
            "coverage (share of traffic)": (lengths, np.array(study.traj_coverage)),
        },
        xlabel="predicates in the anchor",
        ylabel="rate",
        title="Anchors: precision rises and coverage falls as the rule grows",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote anchors report", extra={"path": str(out_path)})

    with track_run(settings, "anchors") as run:
        run.log_metrics(
            {
                "mean_precision": study.mean_precision,
                "mean_coverage": study.mean_coverage,
                "mean_predicates": study.mean_predicates,
                "mean_holdout_precision": study.mean_holdout_precision,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _examples_block(study: AnchorsStudy, limit: int = 4) -> str:
    blocks = []
    seen: set[str] = set()
    for e in study.examples:
        rule = " **AND** ".join(e.predicates)
        if rule in seen:  # distinct flows can share a rule; show each rule once
            continue
        seen.add(rule)
        hold = "n/a" if not np.isfinite(e.holdout_precision) else f"{e.holdout_precision:.0%}"
        blocks.append(
            f"- **IF** {rule} **THEN** attack — precision {e.precision:.0%} "
            f"(LCB {e.precision_lcb:.0%}), coverage {e.coverage:.1%}, held-out precision {hold} "
            f"_(this flow's true label: {e.label})_"
        )
        if len(blocks) >= limit:
            break
    return "\n".join(blocks)


def _read(study: AnchorsStudy) -> str:
    faithful = abs(study.mean_precision - study.mean_holdout_precision) <= 0.1
    faithful_clause = (
        f"The guarantee holds out of sample: anchors precise to {study.mean_precision:.0%} on the "
        f"reference background stay at {study.mean_holdout_precision:.0%} on a held-out background "
        "they were never grown against — the rules generalise, they are not overfit to the sample "
        "used to find them. "
        if faithful
        else (
            f"On this stand-in the held-out precision ({study.mean_holdout_precision:.0%}) drops "
            f"noticeably from the in-sample estimate ({study.mean_precision:.0%}), a reminder that "
            "a greedy rule fit on a finite sample can overstate its own precision — reported "
            "plainly. "
        )
    )
    return (
        f"Across the anchored flows the average rule needs **{study.mean_predicates:.1f} "
        f"predicates** to reach **{study.mean_precision:.0%}** precision while still covering "
        f"**{study.mean_coverage:.1%}** of traffic — a compact sufficient condition, not a "
        f"per-feature attribution. The trade-off is the whole point and is visible in the figure: "
        "each predicate added raises precision and shrinks coverage, and the search stops as soon "
        "as the precision lower bound clears the target. " + faithful_clause
    )


def _render(study: AnchorsStudy, fig: Path) -> str:
    return f"""# NetSentry — Anchor Explanations (high-precision IF-THEN rules)

_Synthetic stand-in. Stratified/binary model at its natural decision boundary; flagged
(predicted-attack) test flows explained by anchors grown on a training-background sample and
re-validated on a held-out test background. Features discretised into {study.n_bins} quantile
bins; target precision tau = {study.tau:g}._

## Why this report exists

SHAP attributes a verdict across features, the counterfactual finds the smallest clearing
change, exemplars point at similar cases — but none states a **sufficient condition** an analyst
can act on. An anchor (Ribeiro, Singh & Guestrin, AAAI 2018) does: a short conjunction of
feature predicates such that, whenever they hold, the model returns this verdict with high
**precision**. Among high-precision rules the useful one has high **coverage**, so it explains a
region of traffic rather than a single flow. Precision is the guarantee; coverage is what makes
it worth stating.

## Example anchors

{_examples_block(study)}

![Precision/coverage trade-off as the anchor grows](../figures/{fig.name})

{_read(study)}

## Scope

Anchors are built by discretising each candidate feature into quantile bins and greedily pinning
the flow to its own bins — a faithful tabular rendering of the paper's algorithm, using a
lower-confidence-bound stopping rule in place of its KL-LUCB bandit, and a real background sample
as the perturbation distribution (so the rule respects the feature correlations the PDP study
warns a synthetic perturbation would break). The precision guarantee is *conditional on the
perturbation distribution*: it says the model is stable on real flows matching the rule, not on
adversarially crafted ones — that stronger, worst-case statement is the certified-robustness
study's job. Anchors complement the additive (SHAP), contrastive (counterfactual), and
case-based (exemplar) views with the sufficient-condition one the suite was missing."""
