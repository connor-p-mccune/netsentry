"""Post-hoc probability calibration for the attack score.

Gradient-boosted trees rank attacks well but their raw score is not ``P(attack)``
(``ml.md`` §4). A threshold or probability the system *reports* should mean what
it says, so we fit a monotonic calibrator — isotonic regression or Platt/sigmoid
scaling — on the **validation** attack scores and apply it everywhere the attack
probability is consumed (the operating-point thresholds and the served number).

Because the calibration map is monotonic it preserves the *ordering* of flows, so
the model's discriminative power is untouched — calibration improves the meaning
of the probability, not the ranking. Platt/sigmoid scaling is strictly monotone
(ranking metrics exactly invariant); isotonic is monotone up to ties, so a
recomputed operating point can move by a negligible amount. Both are asserted in
the tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from netsentry.config import Settings

_METHODS = ("isotonic", "sigmoid")


class ProbabilityCalibrator:
    """A fitted 1-D monotonic map from a raw attack score to a probability."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in _METHODS:
            raise ValueError(f"Unknown calibration method {method!r}; choose from {_METHODS}.")
        self.method = method
        self._model: Any = None

    def fit(self, scores: np.ndarray, y_binary: np.ndarray) -> ProbabilityCalibrator:
        """Fit the calibrator on (raw attack score, attack indicator) pairs."""
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y_binary, dtype=int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(scores, y)
        else:  # Platt scaling: a 1-D logistic fit on the scores.
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1e6, solver="lbfgs")
            model.fit(scores.reshape(-1, 1), y)
        self._model = model
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``."""
        if self._model is None:
            raise RuntimeError("ProbabilityCalibrator.fit must be called before transform().")
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            calibrated = self._model.predict(scores)
        else:
            calibrated = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        out: np.ndarray = np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
        return out


def fit_calibrator(
    settings: Settings, scores: np.ndarray, y_binary: np.ndarray
) -> ProbabilityCalibrator | None:
    """Fit a calibrator when enabled in config and both classes are present.

    Returns ``None`` (the no-op, raw-score path) if calibration is disabled or the
    validation slice is single-class, so callers can fold the result in uniformly.
    """
    if not settings.thresholds.calibrate:
        return None
    if len(np.unique(np.asarray(y_binary, dtype=int))) < 2:
        return None
    return ProbabilityCalibrator(settings.thresholds.calibration_method).fit(scores, y_binary)
