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
    settings.subgroups.min_support = 20  # small fixture: let services qualify for thresholds
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
def test_explain_opt_out_skips_shap_only(client) -> None:  # type: ignore[no-untyped-def]
    """?explain=false is the fast path: identical verdict fields, empty top_features."""
    fast = client.post("/predict?explain=false", json={"flow": SAMPLE_FLOW}).json()
    full = client.post("/predict", json={"flow": SAMPLE_FLOW}).json()
    assert fast["top_features"] == []
    assert full["top_features"]  # the default keeps the explanation contract
    # Only the explanation is skipped; every decision output is unchanged.
    assert fast["attack_probability"] == full["attack_probability"]
    assert fast["is_attack"] == full["is_attack"]
    assert fast["recommended_action"] == full["recommended_action"]

    batch = client.post("/predict/batch?explain=false", json={"flows": [SAMPLE_FLOW]}).json()
    assert batch["predictions"][0]["top_features"] == []


@pytest.mark.slow
def test_exemplars_opt_in_returns_similar_flows(client) -> None:  # type: ignore[no-untyped-def]
    default = client.post("/predict", json={"flow": SAMPLE_FLOW}).json()
    assert default["similar_flows"] is None  # opt-in: absent unless requested

    body = client.post("/predict?exemplars=true", json={"flow": SAMPLE_FLOW}).json()
    assert body["similar_flows"]  # the freshly built bundle embeds an index
    case = body["similar_flows"][0]
    assert set(case) == {"label", "day", "distance"}
    assert case["distance"] >= 0.0
    distances = [c["distance"] for c in body["similar_flows"]]
    assert distances == sorted(distances)  # nearest first
    # Retrieval is evidence, not a decision input: verdict fields are unchanged.
    assert body["is_attack"] == default["is_attack"]
    assert body["attack_probability"] == default["attack_probability"]

    batch = client.post("/predict/batch?exemplars=true", json={"flows": [SAMPLE_FLOW]}).json()
    assert batch["predictions"][0]["similar_flows"]


# A wildly out-of-distribution flow the benign-fit anomaly detector should flag.
ANOMALOUS_FLOW = {
    "Flow Duration": 1.0,
    "Total Fwd Packets": 100000.0,
    "Total Backward Packets": 100000.0,
    "Flow Bytes/s": 1e8,
    "Flow Packets/s": 1e6,
    "Fwd Packet Length Max": 65535.0,
    "Max Packet Length": 65535.0,
}


@pytest.mark.slow
def test_anomaly_explain_opt_in_returns_features(client) -> None:  # type: ignore[no-untyped-def]
    default = client.post("/predict", json={"flow": ANOMALOUS_FLOW}).json()
    assert default["anomaly_features"] is None  # opt-in: absent unless requested

    body = client.post("/predict?anomaly_explain=true", json={"flow": ANOMALOUS_FLOW}).json()
    # The engine reproduces the same verdict; the attribution is evidence, not a decision.
    assert body["is_attack"] == default["is_attack"]
    assert body["anomaly_score"] == default["anomaly_score"]
    if body["is_anomaly"]:  # only a flagged flow carries an explanation
        assert body["anomaly_features"]
        item = body["anomaly_features"][0]
        assert set(item) == {"feature", "contribution"}
        # Contributions are ranked by magnitude, most-driving first.
        mags = [abs(c["contribution"]) for c in body["anomaly_features"]]
        assert mags == sorted(mags, reverse=True)

    # A benign, in-distribution flow requests the field but earns no explanation.
    benign = client.post("/predict?anomaly_explain=true", json={"flow": SAMPLE_FLOW}).json()
    if not benign["is_anomaly"]:
        assert benign["anomaly_features"] is None


@pytest.mark.slow
def test_cost_optimal_profile_is_selectable(client) -> None:  # type: ignore[no-untyped-def]
    # The serving bundle carries a cost-optimal threshold profile alongside the FPR ones.
    response = client.post("/predict?profile=cost_optimal", json={"flow": SAMPLE_FLOW})
    assert response.status_code == 200
    assert response.json()["threshold_profile"] == "cost_optimal"


@pytest.mark.slow
def test_per_service_profile_routes_by_destination_port(client) -> None:  # type: ignore[no-untyped-def]
    # The port rides in the flow mapping as metadata (it is never a model feature);
    # the per_service profile judges the flow at its service's calibrated threshold.
    flow = dict(SAMPLE_FLOW) | {"Destination Port": 22.0}
    response = client.post("/predict?profile=per_service", json={"flow": flow})
    assert response.status_code == 200
    assert response.json()["threshold_profile"] == "per_service"
    # A flow that omits the port still works: it falls back to the global cut.
    response = client.post("/predict?profile=per_service", json={"flow": SAMPLE_FLOW})
    assert response.status_code == 200
    assert response.json()["threshold_profile"] == "per_service"


@pytest.mark.slow
def test_api_key_and_rate_limit(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> None:
    from fastapi.testclient import TestClient

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    # --- API-key auth (rate limit disabled) ---
    settings.serving.api_key = "s3cret"
    auth = TestClient(create_app(settings))
    assert auth.get("/health").status_code == 200  # health is unauthenticated
    assert auth.post("/predict", json={"flow": SAMPLE_FLOW}).status_code == 401  # no key
    assert (
        auth.post(
            "/predict", json={"flow": SAMPLE_FLOW}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    ok = auth.post("/predict", json={"flow": SAMPLE_FLOW}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200

    # --- Fixed-window rate limit (auth disabled) ---
    settings.serving.api_key = None
    settings.serving.rate_limit_per_minute = 3
    limited = TestClient(create_app(settings))
    codes = [limited.post("/predict", json={"flow": SAMPLE_FLOW}).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429  # fourth request in the window is throttled


@pytest.mark.slow
def test_hot_reload_is_disabled_by_default(client) -> None:  # type: ignore[no-untyped-def]
    # The admin surface is opt-in: without serving.reload_enabled it 404s.
    response = client.post("/admin/reload", json={"bundle": "serving_bundle.joblib"})
    assert response.status_code == 404


@pytest.mark.slow
def test_hot_reload_swaps_a_canary_passing_bundle(  # type: ignore[no-untyped-def]
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    import re
    import shutil

    from fastapi.testclient import TestClient

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.serving.reload_enabled = True
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    bundle_path = build_serving_bundle(settings)

    # A valid candidate: a copy of the freshly-built bundle under another name.
    candidate = tmp_path / "models" / "candidate.joblib"
    shutil.copy2(bundle_path, candidate)
    client = TestClient(create_app(settings))

    before = client.get("/health").json()["model_version"]
    response = client.post("/admin/reload", json={"bundle": "candidate.joblib"})
    assert response.status_code == 200
    body = response.json()
    assert body["reloaded"] is True
    assert body["canary"]["ok"] is True  # the swap is canary-gated
    # The service keeps serving predictions after the swap.
    assert client.post("/predict", json={"flow": SAMPLE_FLOW}).status_code == 200
    assert client.get("/health").json()["model_version"] == before  # same model, new file

    metrics = client.get("/metrics").text
    promoted = re.search(r'netsentry_model_reloads_total\{outcome="promoted"\} (\S+)', metrics)
    assert promoted is not None and float(promoted.group(1)) >= 1.0


@pytest.mark.slow
def test_hot_reload_rejects_missing_and_escaping_bundles(  # type: ignore[no-untyped-def]
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from fastapi.testclient import TestClient

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.serving.reload_enabled = True
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)
    client = TestClient(create_app(settings))

    # A path that escapes the models dir is refused before any load.
    assert client.post("/admin/reload", json={"bundle": "../secrets.joblib"}).status_code == 400
    # A missing bundle under the models dir is a 404.
    assert client.post("/admin/reload", json={"bundle": "nope.joblib"}).status_code == 404


@pytest.mark.slow
def test_watch_spool_emits_ecs_alerts(  # type: ignore[no-untyped-def]
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    import json

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

    from netsentry.serving.watch import run_watch

    spool = tmp_path / "spool"
    spool.mkdir()
    # A flow file with capture-identity metadata columns (as pcap --flows-out emits).
    flows = clean_synth.head(40).copy()
    flows["Src IP"] = "10.0.0.5"
    flows["Dst IP"] = "10.0.0.9"
    flows["Dst Port"] = 80
    flows.to_csv(spool / "flows_001.csv", index=False)

    alerts_out = tmp_path / "alerts.ndjson"
    state_path = tmp_path / "state.json"
    totals = run_watch(
        settings,
        spool=spool,
        alerts_out=alerts_out,
        state_path=state_path,
        once=True,
        emit_all=True,  # deterministic output regardless of the operating threshold
    )
    assert totals["files"] == 1
    assert totals["alerts"] == 40  # one ECS doc per flow under emit_all
    lines = alerts_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 40
    doc = json.loads(lines[0])
    assert doc["ecs"]["version"] == "8.11"
    assert doc["event"]["kind"] == "alert"
    assert doc["source"]["ip"] == "10.0.0.5"  # metadata rode into ECS network fields
    assert doc["destination"]["port"] == 80

    # A second pass processes nothing (the file is already recorded in state).
    again = run_watch(
        settings, spool=spool, alerts_out=alerts_out, state_path=state_path, once=True
    )
    assert again["files"] == 0


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


@pytest.mark.slow
def test_health_reports_a_passing_canary(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/health").json()
    # The freshly built bundle embeds canaries; the same runtime must reproduce them.
    assert body["canary"] is not None
    assert body["canary"]["ok"] is True
    assert body["canary"]["n"] > 0
    assert body["canary"]["max_delta"] <= body["canary"]["tolerance"]
    assert body["status"] == "ok"


@pytest.mark.slow
def test_shadow_challenger_scores_silently(  # type: ignore[no-untyped-def]
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    import re
    import shutil

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
    bundle_path = build_serving_bundle(settings)

    # The shadow is the identical bundle under another name: deltas must be zero.
    shadow_path = tmp_path / "models" / "shadow.joblib"
    shutil.copy2(bundle_path, shadow_path)
    settings.serving.artifact_path = bundle_path
    settings.serving.shadow_artifact_path = shadow_path
    client = TestClient(create_app(settings))

    assert client.get("/health").json()["shadow_model_version"] is not None
    for _ in range(3):
        assert client.post("/predict", json={"flow": SAMPLE_FLOW}).status_code == 200

    metrics = client.get("/metrics").text
    scored = re.search(r"^netsentry_shadow_scored_total (\S+)", metrics, re.MULTILINE)
    assert scored is not None and float(scored.group(1)) >= 3.0
    # An identical shadow cannot disagree with the champion at the same profile.
    disagreements = re.search(r"^netsentry_shadow_disagreements_total (\S+)", metrics, re.MULTILINE)
    assert disagreements is not None and float(disagreements.group(1)) == 0.0
