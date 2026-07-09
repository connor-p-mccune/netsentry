"""Leaderboard: family construction, labels, and the shared-protocol evaluator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from netsentry.config import Settings
from netsentry.evaluation.leaderboard import build_family, evaluate_family, family_label
from netsentry.models.supervised import SupervisedClassifier


def test_build_family_covers_every_configured_name(settings: Settings) -> None:
    for name in settings.leaderboard.families:
        assert build_family(name, settings) is not None


def test_build_family_types_and_seeding(settings: Settings) -> None:
    assert isinstance(build_family("majority", settings), DummyClassifier)
    rf = build_family("random_forest", settings)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.random_state == settings.seed
    assert rf.n_estimators == settings.leaderboard.rf_n_estimators
    assert isinstance(build_family("gbdt", settings), SupervisedClassifier)
    with pytest.raises(KeyError):
        build_family("perceptron", settings)


def test_family_label_names_the_deployed_backend(settings: Settings) -> None:
    label = family_label("gbdt", settings)
    assert "deployed" in label
    assert "LightGBM" in label or "HistGradientBoosting" in label


def test_evaluate_family_separable_data_scores_high(settings: Settings) -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] > 0).astype(int)
    x[:, 0] += y * 3.0  # cleanly separable on feature 0
    half = n // 2
    outcome = evaluate_family(
        "logistic",
        settings,
        "temporal",
        x[:half],
        y[:half],
        x[half:],
        y[half:],
        x[half:],
        y[half:],
    )
    assert outcome.split == "temporal"
    assert outcome.pr_auc > 0.95
    assert outcome.fit_seconds >= 0.0
    assert 0.0 <= outcome.tpr_primary <= outcome.tpr_secondary <= 1.0


def test_evaluate_family_majority_prior_is_the_floor(settings: Settings) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3))
    y = (rng.random(300) < 0.3).astype(int)
    outcome = evaluate_family(
        "majority", settings, "stratified", x[:150], y[:150], x[150:], y[150:], x[150:], y[150:]
    )
    # A constant score ranks nothing: PR-AUC collapses to ~prevalence, detection to ~0.
    assert outcome.pr_auc == pytest.approx(float(y[150:].mean()), abs=0.02)
