"""Evaluation figures: confusion matrix, PR/ROC curves, threshold curve.

Matplotlib is imported lazily so importing this module stays cheap. Implemented
in Phase 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> Path:
    """Render a precision-recall curve to ``out_path``."""
    raise NotImplementedError("Implemented in Phase 5")
