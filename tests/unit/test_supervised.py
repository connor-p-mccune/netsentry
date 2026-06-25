"""Supervised model: shape contract, determinism, and beating the baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.config import Settings
from netsentry.data.clean import BINARY_TARGET
from netsentry.features.pipeline import build_pipeline
from netsentry.models.supervised import SupervisedClassifier, build_baselines, resolve_backend
from netsentry.training.train_supervised import quick_metrics


def _features_and_target(
    clean_synth: pd.DataFrame, settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    pipeline = build_pipeline(settings)
    x = pipeline.fit_transform(clean_synth)
    y = clean_synth[BINARY_TARGET].to_numpy()
    return x, y


def test_predict_proba_shape_and_normalisation(
    clean_synth: pd.DataFrame, settings: Settings
) -> None:
    settings.supervised.n_estimators = 80
    x, y = _features_and_target(clean_synth, settings)
    model = SupervisedClassifier(settings).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_training_is_deterministic_with_seed(clean_synth: pd.DataFrame, settings: Settings) -> None:
    settings.supervised.n_estimators = 80
    settings.supervised.n_jobs = 1  # single thread => bit-reproducible
    x, y = _features_and_target(clean_synth, settings)
    first = SupervisedClassifier(settings).fit(x, y).predict_proba(x)
    second = SupervisedClassifier(settings).fit(x, y).predict_proba(x)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_model_beats_majority_baseline(clean_synth: pd.DataFrame, settings: Settings) -> None:
    settings.supervised.n_estimators = 120
    x, y = _features_and_target(clean_synth, settings)
    model = SupervisedClassifier(settings).fit(x, y)
    majority = build_baselines(settings)["majority"].fit(x, y)

    model_ap = quick_metrics(y, model.predict_proba(x), model.classes_, "binary")["pr_auc"]
    base_ap = quick_metrics(y, majority.predict_proba(x), np.asarray(majority.classes_), "binary")[
        "pr_auc"
    ]
    assert model_ap > base_ap


def test_backend_resolution(settings: Settings) -> None:
    settings.supervised.backend = "hist_gbdt"
    assert resolve_backend(settings) == "hist_gbdt"
    settings.supervised.backend = "auto"
    assert resolve_backend(settings) in {"lightgbm", "hist_gbdt"}
