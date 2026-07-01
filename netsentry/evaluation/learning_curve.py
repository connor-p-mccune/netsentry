"""Learning curves — does more training data help, or is the model saturated?

Trains on increasing fractions of the training split (fit fresh each time, evaluate
on the fixed test split) and plots PR-AUC vs training size for both the honest
temporal and the optimistic stratified splits. A curve still rising at 100% argues
for collecting more data; a flat one says the ceiling is the features/model, not the
sample size — the standard bias/variance read, and a useful "what would I do next".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
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

REPORT_NAME = "learning_curve.md"


def subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified subsample of the training frame by attack/benign, seeded."""
    if fraction >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby(BINARY_TARGET):
        n = max(1, round(len(group) * fraction))
        parts.append(group.iloc[rng.permutation(len(group))[:n]])
    return pd.concat(parts).sort_index()


def _pr_auc_on_test(
    settings: Settings, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> float:
    """Fit the leakage-safe pipeline + model on ``train``; PR-AUC on ``test``."""
    pipeline = build_pipeline(settings)
    x_train = pipeline.fit_transform(train)
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    scores = attack_probability(
        model.predict_proba(x_test), model.classes_, settings.labels.benign_label
    )
    return float(average_precision_score(test[BINARY_TARGET].to_numpy(), scores))


def compute_learning_curve(
    settings: Settings, strategy: str, fractions: list[float]
) -> list[tuple[int, float]]:
    """Return (n_train, PR-AUC) points for a split strategy across train fractions."""
    variant = settings.model_copy(deep=True)
    variant.supervised.task = "binary"
    train = load_split(variant, strategy, "train")
    val = load_split(variant, strategy, "val")
    test = load_split(variant, strategy, "test")

    points: list[tuple[int, float]] = []
    for frac in fractions:
        seed_everything(variant.seed)
        sub = subsample(train, frac, variant.seed)
        pr = _pr_auc_on_test(variant, sub, val, test)
        points.append((len(sub), pr))
        logger.info(
            "Learning-curve point",
            extra={"strategy": strategy, "n": len(sub), "pr_auc": round(pr, 4)},
        )
    return points


@dataclass
class LearningCurves:
    """Learning-curve points for the temporal and stratified splits."""

    temporal: list[tuple[int, float]]
    stratified: list[tuple[int, float]]


def run_learning_curve_report(settings: Settings) -> Path:
    """Compute learning curves for both splits, plot them, write the report."""
    fractions = settings.evaluation.learning_curve_fractions
    curves = LearningCurves(
        temporal=compute_learning_curve(settings, "temporal", fractions),
        stratified=compute_learning_curve(settings, "stratified", fractions),
    )
    fig = plots.plot_lines(
        {
            "temporal (honest)": (
                np.array([n for n, _ in curves.temporal]),
                np.array([p for _, p in curves.temporal]),
            ),
            "stratified (optimistic)": (
                np.array([n for n, _ in curves.stratified]),
                np.array([p for _, p in curves.stratified]),
            ),
        },
        xlabel="Training examples",
        ylabel="Test PR-AUC",
        title="Learning curve (PR-AUC vs training size)",
        out_path=settings.paths.figures_dir / "learning_curve.png",
    )

    report = _render(curves, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote learning-curve report", extra={"path": str(out_path)})

    with track_run(settings, "learning_curve") as run:
        run.log_metrics(
            {
                "temporal_pr_auc_full": curves.temporal[-1][1],
                "stratified_pr_auc_full": curves.stratified[-1][1],
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _trend(points: list[tuple[int, float]]) -> float:
    """PR-AUC gained going from the smallest to the largest training size."""
    return points[-1][1] - points[0][1]


def _table(points: list[tuple[int, float]]) -> str:
    head = "| train examples | " + " | ".join(f"{n:,}" for n, _ in points) + " |"
    sep = "|" + "---|" * (len(points) + 1)
    body = "| PR-AUC | " + " | ".join(f"{p:.3f}" for _, p in points) + " |"
    return "\n".join([head, sep, body])


def _render(curves: LearningCurves, fig: Path) -> str:
    t_gain, s_gain = _trend(curves.temporal), _trend(curves.stratified)
    saturated = abs(t_gain) < 0.02
    read = (
        "The temporal curve is nearly flat at full size, so more data of the same kind "
        "would help little — the ceiling is the features/model and the cross-day shift, "
        "not the sample count."
        if saturated
        else "The temporal curve is still rising at full size, so more (or more recent) "
        "labelled data would likely improve honest detection — a concrete next step."
    )
    return f"""# NetSentry — Learning Curves

_Synthetic stand-in. Each point fits the full pipeline on a stratified subsample of
the training split and scores the fixed test split (binary attack vs benign)._

## Temporal (honest) split

{_table(curves.temporal)}

PR-AUC change from smallest to largest training size: **{t_gain:+.3f}**.

## Stratified (optimistic) split

{_table(curves.stratified)}

PR-AUC change: **{s_gain:+.3f}**.

![Learning curve](../figures/{fig.name})

## Read

{read} The gap between the two curves at every training size is the same
over-optimism the headline temporal-vs-stratified comparison exposes — it does not
close with more data, because it is a *validation-protocol* effect, not a sample-size one.
"""
