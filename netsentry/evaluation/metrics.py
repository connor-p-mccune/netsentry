"""Operational metrics: PR-AUC, per-class P/R/F1, TPR@fixed-FPR, alerts/day.

Accuracy is deliberately not a headline metric. Implemented in Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Return (threshold, TPR) at the largest FPR not exceeding ``target_fpr``."""
    raise NotImplementedError("Implemented in Phase 5")
