"""SHAP-based explanations for the supervised model.

Global importance summaries plus top-k per-prediction feature contributions for
the API. Falls back to permutation importance when SHAP is not installed.
Implemented in Phase 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def top_feature_contributions(
    model: object, x: np.ndarray, feature_names: list[str], k: int
) -> list[tuple[str, float]]:
    """Return the top-k (feature, signed contribution) pairs for one flow."""
    raise NotImplementedError("Implemented in Phase 7")
