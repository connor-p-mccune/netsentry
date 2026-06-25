"""End-to-end evaluation: splits -> fit variants -> figures + comparison report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import load_settings
from netsentry.data.split import make_splits
from netsentry.evaluation.report import run_evaluation


@pytest.mark.slow
def test_run_evaluation_produces_report_and_figures(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.figures_dir = tmp_path / "figures"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    report_path = run_evaluation(settings)

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    # Leads with operational metrics and the honest gap, not accuracy.
    assert "PR-AUC" in text
    assert "gap" in text.lower()
    assert "accuracy" not in text.split("## Headline")[1].split("##")[0].lower()
    for figure in ("pr_curve.png", "roc_curve.png", "threshold_curve.png", "confusion_matrix.png"):
        assert (settings.paths.figures_dir / figure).exists()
