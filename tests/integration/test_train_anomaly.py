"""End-to-end anomaly training: leave-one-attack-out + ensemble + report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import load_settings
from netsentry.data.split import make_splits
from netsentry.training.train_anomaly import train_anomaly


@pytest.mark.slow
def test_train_anomaly_reports_loao(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.anomaly.detectors = ["iforest"]  # keep the test fast
    settings.anomaly.loao_min_samples = 20
    settings.supervised.n_estimators = 60

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    result = train_anomaly(settings)

    # Leave-one-attack-out produced per-attack detection for at least one class.
    assert result["loao"]["iforest"]
    for metrics in result["loao"]["iforest"].values():
        assert 0.0 <= metrics["detection_at_fpr"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
    # The ensemble comparison ran on the temporal split.
    assert set(result["ensemble"]) == {"supervised_only", "anomaly_only", "ensemble"}
    assert Path(result["report"]).exists()
