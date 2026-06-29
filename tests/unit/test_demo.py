"""Streamlit demo: sample flows are valid CIC features and score through the engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import load_settings
from netsentry.data import schema
from netsentry.data.split import make_splits
from netsentry.demo.core import EDITABLE_FEATURES, SAMPLE_FLOWS, predict_flow
from netsentry.serving.bundle import build_serving_bundle
from netsentry.serving.inference import InferenceEngine

_ALLOWED = set(schema.FEATURE_COLUMNS)


def _bundle_settings(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> object:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    return settings


def test_sample_flows_use_known_features() -> None:
    assert SAMPLE_FLOWS
    for name, flow in SAMPLE_FLOWS.items():
        assert not (set(flow) - _ALLOWED), f"{name} references unknown features"
    assert set(EDITABLE_FEATURES) <= _ALLOWED


@pytest.mark.slow
def test_presets_score_through_engine(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    settings = _bundle_settings(repo_root, tmp_path, clean_synth)
    engine = InferenceEngine(settings, bundle_path=build_serving_bundle(settings))  # type: ignore[arg-type]
    for name, flow in SAMPLE_FLOWS.items():
        resp = predict_flow(engine, flow)
        assert 0.0 <= resp.attack_probability <= 1.0, name
        assert resp.top_features


@pytest.mark.slow
def test_dashboard_renders_via_apptest(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    settings = _bundle_settings(repo_root, tmp_path, clean_synth)
    bundle_path = build_serving_bundle(settings)  # type: ignore[arg-type]
    monkeypatch.setenv("NETSENTRY_SERVING__ARTIFACT_PATH", str(bundle_path))

    dashboard = repo_root / "netsentry" / "demo" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=60)
    assert not app.exception
    assert any("NetSentry" in (title.value or "") for title in app.title)
