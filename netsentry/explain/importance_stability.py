"""Explanation-trust audit: are the model's feature importances stable across refits?

The API returns SHAP top-features as a product contract, and the evaluation report
shows a global importance ranking. But a ranking from a *single* fit can be an
artifact of one lucky training sample — and shipping an unstable explanation is worse
than shipping none, because it invites false confidence. This refits the model on
bootstrap resamples of the training data, recomputes global feature importance each
time, and measures how much the ranking moves.

Two honest summary numbers: the mean pairwise **Spearman rank correlation** of the
importance vectors across refits (≈1 means the whole ranking is stable), and the mean
pairwise **Jaccard overlap** of the top-k sets (how often the same features lead). If
both are high, the explanations the API ships are trustworthy; if they are low, the
attributions are noise and should be reported as such. The pure metric computation is
separated from the refitting so it unit-tests cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "importance_stability.md"


@dataclass
class FeatureStability:
    """One feature's importance behaviour across the bootstrap refits."""

    feature: str
    mean_importance: float
    mean_rank: float
    rank_std: float
    topk_frequency: float  # fraction of refits in which it landed in the top-k


@dataclass
class StabilityResult:
    """Aggregate explanation-stability across refits."""

    features: list[FeatureStability]  # sorted by mean importance, descending
    rank_correlation: float  # mean pairwise Spearman of importances across refits
    topk_jaccard: float  # mean pairwise Jaccard of the top-k sets
    n_runs: int
    top_k: int


def stability_metrics(
    importances: np.ndarray, feature_names: list[str], top_k: int
) -> StabilityResult:
    """Summarise an (n_runs x n_features) importance matrix into stability metrics."""
    matrix = np.asarray(importances, dtype=float)
    n_runs, n_features = matrix.shape
    top_k = min(top_k, n_features)

    # Rank 1 == most important, per run (ties broken by argsort order — immaterial here).
    order = np.argsort(-matrix, axis=1)
    ranks = np.empty_like(matrix)
    rows = np.arange(n_runs)[:, None]
    ranks[rows, order] = np.arange(1, n_features + 1)[None, :]

    mean_importance = matrix.mean(axis=0)
    mean_rank = ranks.mean(axis=0)
    rank_std = ranks.std(axis=0)
    topk_sets = [set(order[i, :top_k].tolist()) for i in range(n_runs)]
    topk_frequency = np.array([sum(f in s for s in topk_sets) / n_runs for f in range(n_features)])

    correlations: list[float] = []
    jaccards: list[float] = []
    for i, j in combinations(range(n_runs), 2):
        # Spearman is undefined when either importance vector is constant (a degenerate
        # refit); skip those pairs rather than warn and discard a NaN afterwards.
        if matrix[i].std() > 0 and matrix[j].std() > 0:
            rho = spearmanr(matrix[i], matrix[j]).correlation
            if not np.isnan(rho):
                correlations.append(float(rho))
        union = topk_sets[i] | topk_sets[j]
        jaccards.append(len(topk_sets[i] & topk_sets[j]) / len(union) if union else 1.0)

    features = [
        FeatureStability(
            feature=feature_names[f],
            mean_importance=float(mean_importance[f]),
            mean_rank=float(mean_rank[f]),
            rank_std=float(rank_std[f]),
            topk_frequency=float(topk_frequency[f]),
        )
        for f in range(n_features)
    ]
    features.sort(key=lambda x: x.mean_importance, reverse=True)
    return StabilityResult(
        features=features,
        rank_correlation=float(np.mean(correlations)) if correlations else 1.0,
        topk_jaccard=float(np.mean(jaccards)) if jaccards else 1.0,
        n_runs=n_runs,
        top_k=top_k,
    )


def _run_importances(
    model: SupervisedClassifier, x_val: np.ndarray, y_val: np.ndarray, s: Settings
) -> np.ndarray:
    """Global feature importance for one fitted model (gain if available, else permutation)."""
    estimator: Any = model.model
    gain = getattr(estimator, "feature_importances_", None)
    if gain is not None:
        return np.asarray(gain, dtype=float)
    # HistGradientBoosting (the no-LightGBM fallback) exposes no gain importances, so
    # fall back to model-agnostic permutation importance on the validation split.
    from sklearn.inspection import permutation_importance

    cfg = s.importance_stability
    n = min(len(x_val), cfg.max_val_rows)
    result = permutation_importance(
        estimator,
        x_val[:n],
        y_val[:n],
        n_repeats=cfg.permutation_repeats,
        random_state=s.seed,
        scoring="average_precision",
    )
    return np.asarray(result.importances_mean, dtype=float)


def compute_importance_matrix(settings: Settings) -> tuple[np.ndarray, list[str]]:
    """Bootstrap-refit the temporal model and stack each refit's global importance."""
    seed_everything(settings.seed)
    train = load_split(settings, "temporal", "train")
    val = load_split(settings, "temporal", "val")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(settings)
    x_train = np.asarray(pipeline.fit_transform(train))  # FIT ON TRAIN ONLY
    x_val = np.asarray(pipeline.transform(val))
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())

    cfg = settings.importance_stability
    rng = np.random.default_rng(settings.seed)
    rows: list[np.ndarray] = []
    for _ in range(cfg.n_bootstrap):
        idx = rng.choice(len(x_train), len(x_train), replace=True)  # bootstrap resample
        model = SupervisedClassifier(settings).fit(
            x_train[idx], y_train[idx], eval_set=(x_val, y_val)
        )
        rows.append(_run_importances(model, x_val, y_val, settings))
    return np.vstack(rows), feature_names


def run_importance_stability_report(settings: Settings) -> Path:
    """Audit explanation stability across bootstrap refits and write the report."""
    matrix, feature_names = compute_importance_matrix(settings)
    result = stability_metrics(matrix, feature_names, settings.importance_stability.top_k)

    top = result.features[: result.top_k]
    fig = plots.plot_barh(
        [f.feature for f in top],
        [f.topk_frequency for f in top],
        xlabel=f"Top-{result.top_k} frequency across {result.n_runs} refits",
        title="Feature-importance stability",
        out_path=settings.paths.figures_dir / "importance_stability.png",
    )

    report = _render(result, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote importance-stability report",
        extra={"path": str(out_path), "rank_correlation": round(result.rank_correlation, 3)},
    )

    with track_run(settings, "importance_stability") as run:
        run.log_metrics(
            {
                "rank_correlation": result.rank_correlation,
                "topk_jaccard": result.topk_jaccard,
                "n_runs": float(result.n_runs),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _verdict(result: StabilityResult) -> str:
    rho = result.rank_correlation
    jac = result.topk_jaccard
    if rho >= 0.9:
        return (
            f"At **{rho:.2f}** the full ranking is highly stable across refits — the "
            "explanations the API ships are trustworthy, not an artifact of one sample."
        )
    if jac >= 0.55 and jac - rho >= 0.1:
        # The common, important case: the tail is noise but the head holds — the top-k
        # is both reasonably stable and clearly steadier than the full ranking.
        return (
            f"The full ranking is only moderately reproducible (Spearman **{rho:.2f}**), but "
            f"the **top-{result.top_k}** leaders are comparatively stable (Jaccard **{jac:.2f}**): "
            "the headline drivers a SOC reads off an explanation are reliable, while the long "
            "tail of near-zero importances reshuffles between refits. The honest read is *trust "
            "the head, not the tail* — and it is why the API returns only the top few features."
        )
    return (
        f"At Spearman **{rho:.2f}** / top-{result.top_k} Jaccard **{jac:.2f}** the importances "
        "are noisy across refits — individual attributions should not be over-interpreted here. "
        "Reported plainly rather than dressed up, exactly the discipline this audit exists for."
    )


def _render(result: StabilityResult, fig: Path) -> str:
    rows = [
        "| feature | mean importance | mean rank | rank std | top-k freq |",
        "|---|---|---|---|---|",
    ]
    for f in result.features[: result.top_k]:
        rows.append(
            f"| {f.feature} | {f.mean_importance:.4g} | {f.mean_rank:.1f} | "
            f"{f.rank_std:.1f} | {f.topk_frequency * 100:.0f}% |"
        )
    return f"""# NetSentry - Feature-Importance Stability

_Synthetic stand-in. The model is refit on **{result.n_runs}** bootstrap resamples of
the temporal training split; global feature importance is recomputed each time, and the
ranking's movement is measured. Explainability is a product contract here (the API
returns SHAP top-features), so whether those attributions are **stable** is a question
worth answering, not assuming._

## Stability summary

- **Rank correlation (mean pairwise Spearman): {result.rank_correlation:.3f}** — how much
  the whole importance ordering agrees across refits (1.0 = identical).
- **Top-{result.top_k} Jaccard overlap: {result.topk_jaccard:.3f}** — how often the same
  features lead across refits.

{_verdict(result)}

## Top features (by mean importance across refits)

{chr(10).join(rows)}

![Feature-importance stability](../figures/{fig.name})

## Why this matters

A single fit yields one importance ranking; resampling the training data reveals how
much of that ranking is signal versus sampling noise. Features that sit in the top-k in
**every** refit are the ones a SOC analyst can trust when the API explains a flagged
flow; features whose rank swings between refits are attributions to hedge on. This is
the companion to the SHAP global summary (which explains one model) and the feature-group
ablation (which measures each family's causal value): here we audit whether the
explanation itself is reproducible.
"""
