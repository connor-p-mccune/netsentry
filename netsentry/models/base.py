"""Common model interface shared by supervised and anomaly models.

A thin contract (``fit`` / ``predict`` / ``predict_proba`` / ``save`` / ``load``)
so training, evaluation, and serving treat models uniformly and persistence is
consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast

import joblib

if TYPE_CHECKING:
    import numpy as np

    EvalSet = tuple[np.ndarray, np.ndarray]

#: The probability at which a score becomes a hard label under sklearn's convention.
#:
#: This is **not** an operating point and must never be used as one: every deployed decision
#: in this project comes from a threshold calibrated at a false-positive budget
#: (``settings.thresholds``). The constant exists so that the places genuinely reproducing
#: `predict()`'s convention -- a label-only query oracle, a correctness mask for influence
#: attribution -- say so by name rather than by an anonymous `0.5` that reads identically to
#: a threshold somebody chose. Named after the static-analysis pass found the bare literals.
HARD_LABEL_CUT: float = 0.5


class BaseModel(ABC):
    """Abstract estimator with a uniform fit/predict/persist contract."""

    #: Class labels, populated after ``fit`` (classifier models).
    classes_: np.ndarray

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, *, eval_set: EvalSet | None = None) -> BaseModel:
        """Fit the model, optionally using a validation set for early stopping."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (n_samples, n_classes)."""

    def save(self, path: Path) -> Path:
        """Persist this model to ``path`` with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        """Load a previously saved model."""
        return cast("BaseModel", joblib.load(path))
