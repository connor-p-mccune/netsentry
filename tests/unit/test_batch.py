"""Batch file scoring: row conversion (fast) and an end-to-end file score (slow)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.serving.batch import OUTPUT_COLUMNS, _to_row
from netsentry.serving.schemas import FeatureContribution, PredictionResponse


def _response(**kw: object) -> PredictionResponse:
    base = dict(
        predicted_class="DoS Hulk",
        is_attack=True,
        attack_probability=0.912345678,
        anomaly_score=0.4,
        is_anomaly=False,
        top_features=[FeatureContribution(feature="Flow Duration", contribution=0.2)],
        model_version="0.1.0",
        threshold_profile="fpr_1pct",
        prediction_set=["attack"],
        recommended_action="auto_alert",
        mitre={"tactic": "Impact", "technique_id": "T1499", "technique_name": "DoS", "url": "x"},
    )
    base.update(kw)
    return PredictionResponse(**base)  # type: ignore[arg-type]


def test_to_row_extracts_flat_fields() -> None:
    row = _to_row(_response())
    assert set(row) == set(OUTPUT_COLUMNS)
    assert row["predicted_class"] == "DoS Hulk"
    assert row["attack_probability"] == pytest.approx(0.912346)  # rounded to 6 dp
    assert row["mitre_technique"] == "T1499"
    assert row["top_feature"] == "Flow Duration"


def test_to_row_handles_benign_without_mitre() -> None:
    row = _to_row(_response(is_attack=False, predicted_class="BENIGN", mitre=None, top_features=[]))
    assert row["mitre_technique"] is None
    assert row["top_feature"] is None


@pytest.mark.slow
def test_score_file_end_to_end(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.serving.batch import score_file
    from netsentry.serving.bundle import build_serving_bundle

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    flows = clean_synth.head(50)
    in_path = tmp_path / "flows.csv"
    out_path = tmp_path / "scored.csv"
    flows.to_csv(in_path, index=False)

    stats = score_file(settings, in_path, out_path)
    assert stats["scored"] == 50
    result = pd.read_csv(out_path)
    assert len(result) == 50
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["attack_probability"].between(0.0, 1.0).all()
