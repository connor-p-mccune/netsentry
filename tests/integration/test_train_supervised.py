"""End-to-end supervised training: splits -> fit -> bundle -> reload -> predict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import load_settings
from netsentry.data.split import make_splits
from netsentry.models.registry import load_bundle
from netsentry.training.train_supervised import train_supervised


@pytest.mark.slow
def test_train_supervised_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False  # exercise the local-file tracker
    settings.supervised.n_estimators = 80

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    result = train_supervised(settings)

    # A real PR-AUC was produced and the model beats the majority baseline.
    assert 0.0 <= result["metrics"]["pr_auc"] <= 1.0
    assert result["metrics"]["pr_auc"] > result["baselines"]["majority"]["pr_auc"]

    # The bundle persisted and round-trips: reload and predict on raw rows.
    bundle_path = Path(result["bundle"])
    assert bundle_path.exists()
    bundle = load_bundle(bundle_path)
    proba = bundle.predict_proba(clean_synth.head(5))
    assert proba.shape[0] == 5

    # The run was logged locally (MLflow disabled in this test).
    assert (settings.paths.mlruns_dir / "local").exists()
