"""Adversarial-evasion robustness: perturbation geometry + end-to-end curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings, load_settings
from netsentry.robustness.evasion import controllable_indices, mimicry_perturb


def test_controllable_indices_maps_names_to_columns() -> None:
    names = ["Flow Duration", "SYN Flag Count", "Flow Bytes/s"]
    idx = controllable_indices(names, ["Flow Duration", "Flow Bytes/s", "Not Present"])
    assert idx.tolist() == [0, 2]


def test_mimicry_endpoints_and_midpoint() -> None:
    x = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    centroid = np.array([0.0, 0.0, 0.0])
    ctrl = np.array([0, 2])  # leave column 1 fixed

    unchanged = mimicry_perturb(x, centroid, ctrl, 0.0)
    np.testing.assert_array_equal(unchanged, x)

    full = mimicry_perturb(x, centroid, ctrl, 1.0)
    assert full[:, 0].tolist() == [0.0, 0.0]  # moved to centroid
    assert full[:, 2].tolist() == [0.0, 0.0]
    np.testing.assert_array_equal(full[:, 1], x[:, 1])  # non-controllable untouched

    half = mimicry_perturb(x, centroid, ctrl, 0.5)
    assert half[0, 0] == pytest.approx(5.0)


def test_mimicry_no_controllable_features_is_noop() -> None:
    x = np.array([[1.0, 2.0]])
    out = mimicry_perturb(x, np.zeros(2), np.array([], dtype=int), 1.0)
    np.testing.assert_array_equal(out, x)


@pytest.mark.slow
def test_evasion_study_curves_are_monotone(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
    from netsentry.data.split import make_splits
    from netsentry.models.registry import load_bundle
    from netsentry.robustness.evasion import run_evasion_study
    from netsentry.serving.bundle import build_serving_bundle

    settings: Settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.robustness.mimicry_fractions = [0.0, 0.5, 1.0]
    settings.robustness.search_budgets = [0.0, 1.0, 3.0]
    settings.robustness.search_iterations = 20
    settings.robustness.max_attack_samples = 400

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    bundle = load_bundle(build_serving_bundle(settings))

    train = pd.read_parquet(settings.paths.data_processed / "splits/stratified/train.parquet")
    test = pd.read_parquet(settings.paths.data_processed / "splits/stratified/test.parquet")
    attack = test[test[BINARY_TARGET] == 1]
    benign_ref = train[train[MULTICLASS_TARGET] == settings.labels.benign_label]

    study = run_evasion_study(settings, bundle, attack, benign_ref)

    assert 0.0 <= study.baseline_detection <= 1.0
    # Fraction/budget 0 reproduces the un-attacked detection rate.
    assert study.mimicry_detection[0] == pytest.approx(study.baseline_detection)
    assert study.search_detection[0] == pytest.approx(study.baseline_detection)
    # Mimicry toward benign cannot, on average, *raise* detection.
    assert study.mimicry_detection[-1] <= study.baseline_detection + 1e-9
    # The query search only ever keeps a lower score, so detection is non-increasing.
    assert all(np.diff(study.search_detection) <= 1e-9)
    assert study.top_exploitable
