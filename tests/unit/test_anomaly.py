"""Anomaly detectors: anomalies score higher, and the FPR budget is respected."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config import Settings
from netsentry.models.anomaly import (
    AutoencoderDetector,
    IsolationForestDetector,
    build_anomaly_detector,
)
from netsentry.utils.optional import is_available


def _benign_and_anomalies() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, size=(500, 8))
    anomalies = rng.normal(5.0, 1.0, size=(50, 8))  # clearly shifted
    return benign, anomalies


def test_iforest_scores_anomalies_higher(settings: Settings) -> None:
    benign, anomalies = _benign_and_anomalies()
    detector = IsolationForestDetector(settings).fit(benign)
    assert detector.score(anomalies).mean() > detector.score(benign).mean()


def test_threshold_respects_fpr_budget(settings: Settings) -> None:
    benign, _ = _benign_and_anomalies()
    detector = IsolationForestDetector(settings).fit(benign)
    detector.calibrate_threshold(benign, target_fpr=0.05)
    achieved = float(np.mean(detector.score(benign) >= detector.threshold))
    assert achieved == pytest.approx(0.05, abs=0.03)


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_autoencoder_detects_anomalies(settings: Settings) -> None:
    settings.anomaly.autoencoder.epochs = 15
    benign, anomalies = _benign_and_anomalies()
    detector = AutoencoderDetector(settings).fit(benign)
    assert detector.score(anomalies).mean() > detector.score(benign).mean()


def test_factory(settings: Settings) -> None:
    assert isinstance(build_anomaly_detector(settings, "iforest"), IsolationForestDetector)
    with pytest.raises(ValueError, match="Unknown anomaly detector"):
        build_anomaly_detector(settings, "nope")
