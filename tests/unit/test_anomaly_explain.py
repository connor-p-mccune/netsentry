"""Tests for the anomaly-flag attribution (occlusion + faithfulness).

The attribution logic is pure and detector-agnostic, so it is tested against a stub
detector with a *known* anomaly rule: the top-attributed feature must be the one the
rule actually keys on, and the deletion check must reward the true drivers.
"""

from __future__ import annotations

import numpy as np

from netsentry.explain.anomaly_explain import (
    faithfulness_check,
    occlusion_attributions,
)
from netsentry.models.anomaly import AnomalyDetector


class _RuleDetector(AnomalyDetector):
    """A detector whose anomaly score is the squared deviation on chosen features.

    Score = sum over ``driver_features`` of (x[:, j] - benign_value)^2. Benign is 0,
    so a flow is anomalous exactly to the extent its driver features are far from 0 —
    a known ground truth the attribution must recover.
    """

    def __init__(self, driver_features: list[int]) -> None:
        self.driver_features = driver_features

    def fit(self, x_benign: np.ndarray) -> _RuleDetector:
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.asarray(np.sum(x[:, self.driver_features] ** 2, axis=1))


def _anomalous_batch(n: int = 50, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=3.0, size=(n, d))


def test_occlusion_attributes_the_true_driver_features() -> None:
    drivers = [2, 5]
    detector = _RuleDetector(drivers)
    x = _anomalous_batch()
    benign_reference = np.zeros(x.shape[1])  # benign == 0 for this rule

    contrib = occlusion_attributions(detector, x, benign_reference)
    mean_contrib = contrib.mean(axis=0)
    top_two = set(np.argsort(-mean_contrib)[:2].tolist())
    assert top_two == set(drivers)  # the driver features rank highest
    # Non-driver features carry ~zero contribution (occluding them changes nothing).
    non_drivers = [j for j in range(x.shape[1]) if j not in drivers]
    assert np.allclose(contrib[:, non_drivers], 0.0)


def test_occluding_a_driver_reduces_the_score() -> None:
    detector = _RuleDetector([1])
    x = _anomalous_batch()
    benign_reference = np.zeros(x.shape[1])
    base = detector.score(x)
    occluded = x.copy()
    occluded[:, 1] = benign_reference[1]
    assert np.all(detector.score(occluded) <= base + 1e-9)  # resetting a driver can only help


def test_faithfulness_top_beats_random() -> None:
    drivers = [0, 3, 6]
    detector = _RuleDetector(drivers)
    x = _anomalous_batch(n=80, d=10)
    benign_reference = np.zeros(x.shape[1])
    contrib = occlusion_attributions(detector, x, benign_reference)

    top_drop, rand_drop = faithfulness_check(detector, x, benign_reference, contrib, k=3, seed=1)
    # The named features carry essentially all the score, random ones almost none.
    assert top_drop > rand_drop
    assert top_drop / max(rand_drop, 1e-9) > 2.0


def test_attribution_shape_matches_input() -> None:
    detector = _RuleDetector([0])
    x = _anomalous_batch(n=12, d=5)
    contrib = occlusion_attributions(detector, x, np.zeros(5))
    assert contrib.shape == x.shape
