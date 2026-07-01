"""Serving contract: valid flow -> full response schema; malformed -> 422."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import load_settings
from netsentry.data.split import make_splits
from netsentry.serving.app import create_app
from netsentry.serving.bundle import build_serving_bundle

SAMPLE_FLOW = {"Flow Duration": 1200.0, "Total Fwd Packets": 8.0, "Flow Packets/s": 50.0}


@pytest.fixture
def client(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    return TestClient(create_app(settings))


@pytest.mark.slow
def test_health(client) -> None:  # type: ignore[no-untyped-def]
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]


@pytest.mark.slow
def test_predict_returns_full_contract(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post("/predict", json={"flow": SAMPLE_FLOW})
    assert response.status_code == 200
    body = response.json()
    for key in (
        "predicted_class",
        "is_attack",
        "attack_probability",
        "anomaly_score",
        "is_anomaly",
        "top_features",
        "model_version",
        "threshold_profile",
    ):
        assert key in body
    assert isinstance(body["top_features"], list) and body["top_features"]
    assert set(body["top_features"][0]) == {"feature", "contribution"}
    assert 0.0 <= body["attack_probability"] <= 1.0
    # Conformal selective-prediction outputs are part of the contract.
    assert body["recommended_action"] in {"auto_alert", "auto_clear", "review"}
    assert isinstance(body["prediction_set"], list)
    assert set(body["prediction_set"]) <= {"BENIGN", "attack"}
    # MITRE enrichment: present + well-formed when flagged, null when benign.
    if body["is_attack"]:
        assert body["mitre"] is not None
        assert {"tactic", "technique_id", "technique_name"} <= set(body["mitre"])
    else:
        assert body["mitre"] is None


@pytest.mark.slow
def test_cost_optimal_profile_is_selectable(client) -> None:  # type: ignore[no-untyped-def]
    # The serving bundle carries a cost-optimal threshold profile alongside the FPR ones.
    response = client.post("/predict?profile=cost_optimal", json={"flow": SAMPLE_FLOW})
    assert response.status_code == 200
    assert response.json()["threshold_profile"] == "cost_optimal"


@pytest.mark.slow
def test_malformed_requests_return_422(client) -> None:  # type: ignore[no-untyped-def]
    # Unknown feature column.
    assert client.post("/predict", json={"flow": {"NotAFeature": 1.0}}).status_code == 422
    # Non-numeric value.
    assert client.post("/predict", json={"flow": {"Flow Duration": "abc"}}).status_code == 422
    # Missing the required `flow` key.
    assert client.post("/predict", json={"oops": {}}).status_code == 422


@pytest.mark.slow
def test_batch_and_metrics(client) -> None:  # type: ignore[no-untyped-def]
    batch = client.post("/predict/batch", json={"flows": [SAMPLE_FLOW, SAMPLE_FLOW]})
    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "netsentry_request_latency_seconds" in metrics.text


@pytest.mark.slow
def test_model_behaviour_metrics_emitted(client) -> None:  # type: ignore[no-untyped-def]
    # Scoring a flow should populate the model-behaviour collectors the Grafana
    # dashboard reads (predictions by decision + attack-probability histogram).
    client.post("/predict", json={"flow": SAMPLE_FLOW})
    text = client.get("/metrics").text
    assert "netsentry_predictions_total" in text
    assert "netsentry_attack_probability" in text
