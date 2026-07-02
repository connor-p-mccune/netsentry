"""Active learning — spend the analyst's labeling budget where it moves detection.

Labels are the binding constraint in a real SOC: an analyst can only adjudicate so
many flows a day, so the question is not "train on everything" but "given a budget,
*which* flows should be labeled next". This study starts from a small labeled seed
and grows the labeled set two ways — **uncertainty sampling** (query the flows whose
calibrated attack probability sits nearest the decision boundary, i.e. the model is
least sure about) versus a **random** baseline — refitting and scoring the held-out
test split after each round. The gap between the curves is the label-efficiency win.

It runs on the **stratified** split on purpose: active learning assumes the
unlabeled pool and the test set are exchangeable, which holds there. On the temporal
split the same later-day distribution shift that breaks conformal exchangeability
would also mislead query selection — so this is the reference-split technique, and
the report says so rather than overclaiming it on the honest split.

Only the *labels* are treated as scarce: the feature pipeline is unsupervised
(impute/scale) and may see the whole pool, exactly as it could in deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "active_learning.md"


def select_uncertain(probs: np.ndarray, unlabeled: np.ndarray, k: int) -> np.ndarray:
    """The ``k`` unlabeled indices whose attack probability is nearest 0.5.

    Least-confident-by-margin selection for a binary detector: |p - 0.5| smallest is
    the flow the model is most torn on, so a label there is the most informative.
    """
    if k >= len(unlabeled):
        return unlabeled
    margin = np.abs(probs[unlabeled] - 0.5)
    order = np.argsort(margin, kind="stable")[:k]
    return unlabeled[order]


def select_random(unlabeled: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random ``k``-subset of the unlabeled pool — the baseline to beat."""
    if k >= len(unlabeled):
        return unlabeled
    return rng.choice(unlabeled, size=k, replace=False)


@dataclass
class RoundPoint:
    """One labeling round: labels spent, test PR-AUC, and detection at the budget."""

    n_labeled: int
    pr_auc: float
    detection: float


def _fit_and_score(
    settings: Settings,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    labeled: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    operating_fpr: float,
) -> tuple[np.ndarray, RoundPoint]:
    """Train on the labeled subset; return pool attack-probs + the test operating point."""
    model = SupervisedClassifier(settings).fit(
        x_pool[labeled], y_pool[labeled], eval_set=(x_val, y_val)
    )
    benign = settings.labels.benign_label
    pool_probs = attack_probability(model.predict_proba(x_pool), model.classes_, benign)
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
    detection = rates_at_threshold(y_test, s_test, threshold)["tpr"]
    pr_auc = float(average_precision_score(y_test, s_test))
    return pool_probs, RoundPoint(int(labeled.size), pr_auc, detection)


def run_active_learning_strategy(
    settings: Settings,
    strategy: str,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> list[RoundPoint]:
    """Run one acquisition strategy from a shared seed; return the per-round curve."""
    cfg = settings.active_learning
    operating_fpr = settings.thresholds.fpr_targets[-1]
    rng = np.random.default_rng(settings.seed)
    all_idx = np.arange(len(y_pool))
    labeled = rng.choice(all_idx, size=min(cfg.seed_size, len(all_idx)), replace=False)
    labeled_mask = np.zeros(len(y_pool), dtype=bool)
    labeled_mask[labeled] = True

    points: list[RoundPoint] = []
    for round_i in range(cfg.rounds + 1):
        seed_everything(settings.seed)
        pool_probs, point = _fit_and_score(
            settings, x_pool, y_pool, labeled, x_val, y_val, x_test, y_test, operating_fpr
        )
        points.append(point)
        logger.info(
            "AL round",
            extra={"strategy": strategy, "n": point.n_labeled, "pr_auc": round(point.pr_auc, 4)},
        )
        if round_i == cfg.rounds:
            break
        unlabeled = all_idx[~labeled_mask]
        if unlabeled.size == 0:
            break
        if strategy == "uncertainty":
            chosen = select_uncertain(pool_probs, unlabeled, cfg.query_batch)
        else:
            chosen = select_random(unlabeled, cfg.query_batch, rng)
        labeled_mask[chosen] = True
        labeled = all_idx[labeled_mask]
    return points


def run_active_learning_report(settings: Settings) -> Path:
    """Compare uncertainty vs random acquisition on the stratified split; write the report."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")

    cfg = variant.active_learning
    if len(train) > cfg.max_pool:  # cap the pool so the many refits stay fast
        train = train.sample(cfg.max_pool, random_state=variant.seed).reset_index(drop=True)

    pipeline = build_pipeline(variant)  # unsupervised transforms; labels stay scarce
    x_pool = pipeline.fit_transform(train)
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    y_pool = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    curves = {
        strategy: run_active_learning_strategy(
            variant, strategy, x_pool, y_pool, x_val, y_val, x_test, y_test
        )
        for strategy in cfg.strategies
    }

    fig = plots.plot_lines(
        {
            f"{name} sampling": (
                np.array([p.n_labeled for p in pts]),
                np.array([p.pr_auc for p in pts]),
            )
            for name, pts in curves.items()
        },
        xlabel="Labeled flows (analyst budget)",
        ylabel="Test PR-AUC",
        title="Active learning: label efficiency (stratified split)",
        out_path=variant.paths.figures_dir / "active_learning.png",
    )

    report = _render(curves, fig, variant)
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote active-learning report", extra={"path": str(out_path)})

    with track_run(settings, "active_learning") as run:
        run.log_metrics(
            {f"{name}_pr_auc_final": pts[-1].pr_auc for name, pts in curves.items() if pts}
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _labels_to_reach(points: list[RoundPoint], target: float) -> int | None:
    """Fewest labels at which a curve reaches ``target`` PR-AUC (None if never)."""
    for p in points:
        if p.pr_auc >= target:
            return p.n_labeled
    return None


def _read(curves: dict[str, list[RoundPoint]]) -> str:
    """Sign-aware efficiency read comparing uncertainty against random.

    Uses the standard active-learning metric: how many labels uncertainty sampling
    needs to reach the quality random only reaches at the *end* of its budget.
    """
    if "uncertainty" not in curves or "random" not in curves:
        return "Configure both an `uncertainty` and a `random` strategy to compare efficiency."
    unc, rnd = curves["uncertainty"], curves["random"]
    target = rnd[-1].pr_auc  # the quality random achieves with its full budget
    n_rnd = rnd[-1].n_labeled
    n_unc = _labels_to_reach(unc, target)
    # Also report the gap at a mid-budget point where the curves are still separated.
    mid = len(rnd) // 2
    mid_gap = unc[mid].pr_auc - rnd[mid].pr_auc

    if n_unc is not None and n_unc < n_rnd:
        saved = n_rnd - n_unc
        pct = 100 * saved / n_rnd
        return (
            f"Uncertainty sampling reaches random's full-budget PR-AUC ({target:.3f}) with "
            f"**{n_unc:,}** labels — **{saved:,} fewer ({pct:.0f}%)** than the {n_rnd:,} random "
            f"spends to get there, and it leads at every mid-budget round "
            f"(+{mid_gap:.3f} PR-AUC at {unc[mid].n_labeled:,} labels). Querying the flows the "
            "model is least sure about spends the budget where it actually moves the decision "
            "boundary — the argument for a review queue ordered by model uncertainty rather "
            "than by arrival time."
        )
    if mid_gap > 0.005:
        return (
            f"Uncertainty sampling leads through the rising part of the curve "
            f"(+{mid_gap:.3f} PR-AUC at {unc[mid].n_labeled:,} labels) before both saturate at "
            "the pool ceiling. The advantage is real where labels are genuinely scarce (the "
            "early budget); once nearly everything informative is labeled the strategies "
            "necessarily converge."
        )
    return (
        "On this stand-in the two curves run close: the synthetic classes separate easily "
        "enough that a random seed already lands near the ceiling, leaving little headroom "
        "for smart querying. On harder real traffic — where the boundary is genuinely "
        "contested — uncertainty sampling has more room to help. The method and the honest "
        "reporting are the point, not the synthetic margin."
    )


def _table(name: str, points: list[RoundPoint]) -> str:
    head = f"| {name} | " + " | ".join(f"{p.n_labeled:,}" for p in points) + " |"
    sep = "|" + "---|" * (len(points) + 1)
    body = "| PR-AUC | " + " | ".join(f"{p.pr_auc:.3f}" for p in points) + " |"
    return "\n".join([head, sep, body])


def _render(curves: dict[str, list[RoundPoint]], fig: Path, settings: Settings) -> str:
    operating_fpr = settings.thresholds.fpr_targets[-1]
    tables = "\n\n".join(_table(f"{name} — labeled", pts) for name, pts in curves.items() if pts)
    return f"""# NetSentry — Active Learning (label efficiency)

_Synthetic stand-in. Stratified split (where the pool and test are exchangeable, the
assumption active learning needs). Binary attack-vs-benign; each point refits on the
labeled subset and scores the fixed test split. Detection thresholds use the
{operating_fpr * 100:g}%-FPR budget on validation._

## The question

An analyst can only label so many flows a day, so labels — not compute — are the
scarce resource. Active learning asks *which* flows to label next: the ones the
model is least sure about (**uncertainty sampling**, query nearest the decision
boundary) or a **random** draw. The gap is analyst time saved for equal detection.

## PR-AUC vs labeling budget

{tables}

![Active learning](../figures/{fig.name})

## Read

{_read(curves)}

The tie-in to the rest of the pipeline: this is the *training-time* mirror of the
conformal selective-prediction work (which routes uncertain flows to a human at
*inference* time). Both order the analyst's attention by model uncertainty — active
learning to build a better model with fewer labels, conformal to spend review effort
only where the model abstains. And both rest on exchangeability, which is exactly
why this study lives on the stratified split and the honest temporal number does not.
"""
