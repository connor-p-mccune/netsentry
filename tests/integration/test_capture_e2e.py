"""Packet-to-verdict integration: demo pcap -> flows -> scored predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from netsentry.capture.demo import write_demo_pcap
from netsentry.capture.flows import FLOW_META_COLUMNS
from netsentry.capture.score import score_capture
from netsentry.serving.batch import OUTPUT_COLUMNS


@pytest.mark.slow
def test_demo_capture_scores_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
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

    capture = write_demo_pcap(tmp_path / "demo.pcap", seed=11)
    out = tmp_path / "scored.csv"
    flows_out = tmp_path / "flows.csv"
    summary = score_capture(settings, capture, out, flows_out=flows_out)

    assert summary["parsed"] == summary["packets"]
    assert summary["flows"] > 40

    scored = pd.read_csv(out)
    assert list(scored.columns) == FLOW_META_COLUMNS + OUTPUT_COLUMNS
    assert len(scored) == summary["flows"]
    assert scored["attack_probability"].between(0.0, 1.0).all()
    # Flow identity must ride along so an analyst can act on a verdict.
    assert {"203.0.113.66", "198.51.100.7"} <= set(scored["Src IP"])

    flows = pd.read_csv(flows_out)
    assert len(flows) == summary["flows"]
    assert "Flow Duration" in flows.columns
