"""Partial dependence + ICE: the response-curve shape SHAP and ablation don't show.

The explainability suite already answers *which* features matter (SHAP global
importance), *how much a family is worth* (ablation), *whether the ranking is
trustworthy* (importance stability), and *what would clear this flow*
(counterfactual). None of them answer the question a modeller asks next: **what is
the shape of the response** — as a feature sweeps its range, does the attack
probability rise, fall, saturate, or turn over? That is exactly what a partial
dependence plot (Friedman) shows, with individual conditional expectation (ICE)
curves layered underneath to expose heterogeneity the average hides.

The computation is done honestly in **raw feature space**: a feature is swept across
a grid of its own data quantiles while every other column is held at its real value,
and each perturbed frame is pushed through the *fitted pipeline + model* — the same
transform the API applies — so the x-axis is in interpretable raw units and there is
no train/serve skew. The report states the standard PDP caveat plainly: it assumes
the swept feature is independent of the others, so where features are correlated the
curve extrapolates into traffic that does not occur, and the ICE spread is the
honest signal of that. It is a diagnostic of the model's learned response, not a
causal claim — the causal reading is the ablation study's job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "partial_dependence.md"
FIGURE_NAME = "partial_dependence.png"

ScoreFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class PartialDependence:
    """One feature's partial-dependence and ICE curves over its value grid."""

    feature: str  # raw feature name (human-readable)
    grid: np.ndarray  # grid values, in raw units
    average: np.ndarray  # PDP: mean model score at each grid value
    ice: np.ndarray  # (n_ice, len(grid)) sample of individual curves
    importance: float  # model importance (used only for feature selection/ordering)

    @property
    def effect(self) -> float:
        """Peak-to-trough swing of the PDP — the feature's marginal effect size."""
        return float(self.average.max() - self.average.min())

    @property
    def direction(self) -> str:
        """Monotone up/down, or non-monotone (turns over) — the coarse shape."""
        diffs = np.diff(self.average)
        if np.all(diffs >= -1e-9):
            return "increasing"
        if np.all(diffs <= 1e-9):
            return "decreasing"
        return "non-monotone"


def _grid_for(values: np.ndarray, *, points: int, trim_quantile: float) -> np.ndarray:
    """A monotone grid over a feature's central mass (trimmed to avoid outlier tails)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0])
    lo, hi = np.quantile(finite, [trim_quantile, 1.0 - trim_quantile])
    if hi <= lo:  # (near-)constant feature: nothing to sweep
        return np.array([float(lo)])
    return np.linspace(float(lo), float(hi), points)


def partial_dependence_1d(
    score_fn: ScoreFn,
    base: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
    *,
    n_ice: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PDP average and an ICE sample for one feature (pure, given a score function).

    For each grid value the feature column is overwritten across ``base`` and the
    frame is re-scored; the PDP is the mean over rows, the ICE is the per-row curve
    for the first ``n_ice`` rows.
    """
    n_ice = min(n_ice, len(base))
    scores = np.empty((len(grid), len(base)), dtype=float)
    for g, value in enumerate(grid):
        perturbed = base.copy()
        perturbed[feature] = value
        scores[g] = score_fn(perturbed)
    average = scores.mean(axis=1)
    ice = scores[:, :n_ice].T  # (n_ice, len(grid))
    return average, ice


def _fit_scorer(settings: Settings) -> tuple[ScoreFn, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit the honest temporal/binary pipeline+model; return a raw-frame score function.

    The scorer takes a raw flow frame and returns calibrated-free attack probability
    through the exact fitted pipeline, so partial dependence is measured on the model
    the rest of the project evaluates.
    """
    seed_everything(settings.seed)
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))  # FIT ON TRAIN ONLY
    x_val = np.asarray(pipeline.transform(val))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.model.classes_)

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        transformed = np.asarray(pipeline.transform(frame))
        return positive_scores(model.predict_proba(transformed), classes)

    importances = getattr(model.model, "feature_importances_", None)
    transformed_names = list(pipeline.named_steps["features"].get_feature_names_out())
    imp = (
        np.asarray(importances, dtype=float)
        if importances is not None
        else np.ones(len(transformed_names))
    )
    ranking = pd.Series(imp, index=[display_feature_name(n) for n in transformed_names])
    return score_fn, train, val, ranking


def compute_partial_dependence(settings: Settings) -> list[PartialDependence]:
    """Compute PDP + ICE for the top model features on the honest temporal model."""
    cfg = settings.partial_dependence
    score_fn, train, val, ranking = _fit_scorer(settings)

    # Rank raw features by model importance; a raw feature may map from several
    # transformed columns (e.g. a one-hot), so take the max importance per raw name.
    per_feature = ranking.groupby(level=0).max().sort_values(ascending=False)
    features = [f for f in per_feature.index if f in train.columns][: cfg.top_k]

    sample = val.sample(n=min(cfg.sample_rows, len(val)), random_state=settings.seed)
    results: list[PartialDependence] = []
    for feature in features:
        grid = _grid_for(
            train[feature].to_numpy(dtype=float),
            points=cfg.grid_points,
            trim_quantile=cfg.grid_trim_quantile,
        )
        if grid.size < 2:  # constant feature — no response to plot
            continue
        average, ice = partial_dependence_1d(score_fn, sample, feature, grid, n_ice=cfg.ice_samples)
        results.append(
            PartialDependence(
                feature=feature,
                grid=grid,
                average=average,
                ice=ice,
                importance=float(per_feature[feature]),
            )
        )
    return results


def run_partial_dependence_report(settings: Settings) -> Path:
    """Compute PDP/ICE for the top features, plot the grid, and write the report."""
    results = compute_partial_dependence(settings)
    panels = [(r.feature, r.grid, r.average, r.ice) for r in results]
    fig = plots.plot_pdp_grid(panels, out_path=settings.paths.figures_dir / FIGURE_NAME, ncols=2)

    report = _render(results, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote partial-dependence report",
        extra={"path": str(out_path), "features": len(results)},
    )

    with track_run(settings, "partial_dependence") as run:
        run.log_metrics({f"effect_{r.feature[:24]}": r.effect for r in results})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(results: list[PartialDependence], fig: Path) -> str:
    rows = ["| feature | shape | marginal effect (Δp) | model importance |", "|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r.feature} | {r.direction} | {r.effect:.3f} | {r.importance:.4g} |")
    body = (
        chr(10).join(rows) if results else "_No non-constant top features to profile on this data._"
    )
    return f"""# NetSentry - Partial Dependence & ICE

_Synthetic stand-in. Partial dependence (Friedman) for the top model features on the
honest **temporal / binary** model. Each feature is swept across a grid of its own
data quantiles while every other column stays at its real value, and every perturbed
flow is scored through the **fitted pipeline + model** — the same transform the API
applies, so the x-axis is in raw units and there is no train/serve skew._

## Marginal response of the top features

{body}

![Partial dependence and ICE](../figures/{fig.name})

The bold line is the **partial dependence** (mean predicted attack probability as the
feature sweeps its range); the faint lines are **individual conditional expectation
(ICE)** curves — one flow each — whose spread is the heterogeneity the average hides.

## How to read this (and what it is not)

A partial dependence plot shows the *shape* of the model's learned response — rising,
falling, saturating, or turning over — which the SHAP global summary (a single
importance number) and the ablation study (a family's causal value) do not. The
features with the steepest curves are the ones the model's score is most sensitive to,
and they line up with the attacker-controllable features the evasion and recourse
studies exploit — the response shape *is* the surface an adversary shapes traffic along.

**The standard caveat, stated plainly:** PDP assumes the swept feature is independent of
the others. Where features are correlated (flow rates, packet counts, and byte totals
move together here), sweeping one alone pushes the frame into traffic that does not
occur, and the curve extrapolates. The **ICE spread** is the honest signal of that: when
the individual curves fan out or cross, the average is hiding interaction, and the PDP
should be read as the model's *marginal* response, not a causal effect — the causal
reading is the feature-group ablation's job.
"""
