"""SHAP-based explanations for the supervised model.

A flagged flow is only actionable with a reason, so explanations are a product
requirement, not a nicety. Uses SHAP ``TreeExplainer`` (exact, fast on trees) for
both a global importance summary and top-k per-prediction contributions. When
SHAP is unavailable it degrades to the model's feature importances so the API
still returns ``top_features`` (a documented approximation).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.features.feature_sets import display_feature_name
from netsentry.log import get_logger
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)


def _reduce_to_row_contributions(shap_output: Any, row: int = 0) -> np.ndarray:
    """Reduce SHAP output (list / 2D / 3D, across versions) to one row's vector."""
    values = shap_output
    if isinstance(values, list):  # older API: one array per class -> positive class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes) -> positive/last class
        values = values[:, :, -1]
    return np.asarray(values[row])


class ShapExplainer:
    """Explains a fitted :class:`ModelBundle` globally and per-prediction."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        # Analyst-facing names: the pipeline's numeric__ branch prefix is plumbing.
        self.feature_names = [display_feature_name(n) for n in bundle.feature_names()]
        # The raw tree estimator lives on the wrapper's `.model`; fall back to the
        # wrapper itself for estimators without one.
        self._estimator: Any = getattr(bundle.model, "model", bundle.model)
        self._explainer: Any = None
        self.mode = "importance"
        if is_available("shap"):
            try:
                import shap

                self._explainer = shap.TreeExplainer(self._estimator)
                self.mode = "shap"
            except Exception as exc:  # SHAP can reject some estimators; degrade gracefully
                logger.warning("SHAP TreeExplainer unavailable (%s); using importances", exc)

    def explain_row(self, flow: pd.DataFrame, k: int) -> list[tuple[str, float]]:
        """Top-k (feature, signed contribution) pairs for a single flow row."""
        x = self.bundle.pipeline.transform(flow)
        if self.mode == "shap":
            contributions = _reduce_to_row_contributions(self._explainer.shap_values(x), row=0)
        else:
            contributions = self._importances() * np.asarray(x)[0]
        order = np.argsort(np.abs(contributions))[::-1][:k]
        return [(self.feature_names[i], float(contributions[i])) for i in order]

    def _importances(self) -> np.ndarray:
        importances = getattr(self._estimator, "feature_importances_", None)
        if importances is None:
            return np.ones(len(self.feature_names)) / len(self.feature_names)
        return np.asarray(importances, dtype="float64")

    def global_importance(self, background: np.ndarray, top_n: int = 20) -> list[tuple[str, float]]:
        """Mean |SHAP| (or model importance) over a background sample, top-n."""
        if self.mode == "shap":
            raw = self._explainer.shap_values(background)
            if isinstance(raw, list):
                raw = raw[-1]
            raw = np.asarray(raw)
            if raw.ndim == 3:
                raw = raw[:, :, -1]
            importance = np.abs(raw).mean(axis=0)
        else:
            importance = self._importances()
        order = np.argsort(importance)[::-1][:top_n]
        return [(self.feature_names[i], float(importance[i])) for i in order]

    def plot_global(self, background: np.ndarray, out_path: Path, top_n: int = 20) -> Path:
        """Horizontal bar chart of the top global feature importances."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranked = self.global_importance(background, top_n=top_n)
        names = [n for n, _ in ranked][::-1]
        values = [v for _, v in ranked][::-1]
        fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.3)))
        ax.barh(names, values, color="#3b7dd8")
        label = "mean |SHAP value|" if self.mode == "shap" else "model feature importance"
        ax.set(xlabel=label, title="Global feature importance")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote global importance figure", extra={"path": str(out_path)})
        return out_path


def top_feature_contributions(
    bundle: ModelBundle, flow: pd.DataFrame, k: int
) -> list[tuple[str, float]]:
    """Convenience wrapper: top-k contributions for one flow via a fresh explainer."""
    return ShapExplainer(bundle).explain_row(flow, k)
