"""Calibration metrics and the probability calibrator.

The high-value property: calibration must improve the *meaning* of the score
(lower Brier/ECE) while preserving the *ranking* (so PR-AUC/TPR@FPR are untouched).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from netsentry.config import Settings
from netsentry.evaluation.calibration import (
    brier_score,
    calibration_summary,
    expected_calibration_error,
    reliability_curve,
)
from netsentry.models.calibration import ProbabilityCalibrator, fit_calibrator


def test_reliability_curve_lengths_and_weights_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    prob = rng.uniform(size=500)
    y = (rng.uniform(size=500) < prob).astype(int)
    mean_pred, observed, weight = reliability_curve(y, prob, n_bins=10)
    assert len(mean_pred) == len(observed) == len(weight)
    assert np.isclose(weight.sum(), 1.0)


def test_ece_zero_for_calibrated_constant() -> None:
    # Half positives at a constant 0.5 prediction is perfectly calibrated.
    y = np.array([0, 1] * 50)
    prob = np.full(100, 0.5)
    assert expected_calibration_error(y, prob, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence() -> None:
    # Predict 0.9 everywhere but nothing is an attack -> gap of 0.9.
    y = np.zeros(100, dtype=int)
    prob = np.full(100, 0.9)
    assert expected_calibration_error(y, prob, n_bins=10) == pytest.approx(0.9, abs=1e-9)


def test_brier_score_known_values() -> None:
    assert brier_score(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert brier_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_calibration_summary_keys() -> None:
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.2, 0.8, 0.3, 0.7])
    summary = calibration_summary(y, prob)
    assert set(summary) == {"brier", "ece", "mce"}


def _miscalibrated_sample(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Overconfident scores: true attack rate is score**2, not score."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(size=4000)
    y = (rng.uniform(size=4000) < scores**2).astype(int)
    return scores, y


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_is_monotone_and_bounded(method: str) -> None:
    scores, y = _miscalibrated_sample()
    out = ProbabilityCalibrator(method).fit(scores, y).transform(scores)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Sorting by the raw score, the calibrated score is non-decreasing: the map
    # preserves the ordering of flows (it only re-scales the axis).
    ordered = out[np.argsort(scores)]
    assert np.all(np.diff(ordered) >= -1e-9)


def test_sigmoid_preserves_pr_auc_exactly() -> None:
    # Platt scaling is strictly monotone, so ranking metrics are exactly invariant.
    scores, y = _miscalibrated_sample()
    out = ProbabilityCalibrator("sigmoid").fit(scores, y).transform(scores)
    assert average_precision_score(y, out) == pytest.approx(average_precision_score(y, scores))


def test_isotonic_preserves_pr_auc_up_to_ties() -> None:
    # Isotonic is monotone *up to ties*, so PR-AUC moves only negligibly.
    scores, y = _miscalibrated_sample()
    out = ProbabilityCalibrator("isotonic").fit(scores, y).transform(scores)
    assert average_precision_score(y, out) == pytest.approx(
        average_precision_score(y, scores), abs=0.02
    )


def test_isotonic_improves_calibration_on_fit_data() -> None:
    scores, y = _miscalibrated_sample()
    cal = ProbabilityCalibrator("isotonic").fit(scores, y)
    calibrated = cal.transform(scores)
    # Isotonic is the L2-optimal monotone fit, so Brier cannot get worse than the
    # raw (identity is also monotone), and ECE should drop on an overconfident set.
    assert brier_score(y, calibrated) <= brier_score(y, scores) + 1e-9
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, scores)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        ProbabilityCalibrator("isotonic").transform(np.array([0.5]))


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="calibration method"):
        ProbabilityCalibrator("nope")


def test_fit_calibrator_respects_config(settings: Settings) -> None:
    scores, y = _miscalibrated_sample()
    settings.thresholds.calibrate = True
    assert isinstance(fit_calibrator(settings, scores, y), ProbabilityCalibrator)

    settings.thresholds.calibrate = False
    assert fit_calibrator(settings, scores, y) is None

    settings.thresholds.calibrate = True
    single_class = np.zeros(len(scores), dtype=int)
    assert fit_calibrator(settings, scores, single_class) is None
