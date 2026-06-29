"""ONNX export: numerically faithful to sklearn; quantization is a safe no-op for trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings, load_settings
from netsentry.data.split import make_splits
from netsentry.training.train_supervised import fit_supervised


def _prepared_settings(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> Settings:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    return settings


@pytest.mark.slow
def test_onnx_export_matches_sklearn(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    from netsentry.serving.onnx_export import OnnxScorer, export_to_onnx, quantize_onnx

    settings = _prepared_settings(repo_root, tmp_path, clean_synth)
    bundle = fit_supervised(settings).bundle

    x = np.asarray(bundle.pipeline.transform(clean_synth.head(64)), dtype=np.float32)
    onnx_path = export_to_onnx(bundle, tmp_path / "model.onnx")
    assert onnx_path.exists()

    scorer = OnnxScorer(onnx_path)
    sk = np.asarray(bundle.model.predict_proba(x))
    ox = scorer.predict_proba(x)
    assert (sk.argmax(1) == ox.argmax(1)).mean() == 1.0  # identical decisions
    assert float(np.max(np.abs(sk - ox))) < 1e-3  # identical probabilities

    quant = quantize_onnx(onnx_path, tmp_path / "model.quant.onnx")
    assert quant is None or quant.exists()  # runs without crashing (no-op for trees)


@pytest.mark.slow
def test_run_onnx_export_writes_report(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    from netsentry.serving.onnx_export import run_onnx_export

    settings = _prepared_settings(repo_root, tmp_path, clean_synth)
    bundle = fit_supervised(settings).bundle
    out = run_onnx_export(settings, bundle=bundle, repeats=3)
    assert out.exists()
    assert "ONNX" in out.read_text(encoding="utf-8")
