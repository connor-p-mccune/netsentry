"""End-to-end smoke tests for the analysis report generators.

The pure logic is unit-tested elsewhere; these guard the orchestration (fitting,
plotting, markdown rendering, file output) so the report commands cannot bitrot.
All paths are redirected to a tmp dir so the committed reports/figures are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from netsentry.config import Settings, load_settings
from netsentry.data.split import make_splits


@pytest.fixture
def prepared(repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame) -> Settings:
    """Default settings with tmp paths and persisted splits ready for reports."""
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.figures_dir = tmp_path / "figures"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    return settings


@pytest.mark.slow
def test_cost_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.cost import run_cost_report

    out = run_cost_report(prepared)
    assert out.exists() and "expected cost" in out.read_text(encoding="utf-8").lower()


@pytest.mark.slow
def test_conformal_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.conformal import run_conformal_report

    out = run_conformal_report(prepared)
    assert out.exists() and "coverage" in out.read_text(encoding="utf-8").lower()


@pytest.mark.slow
def test_active_learning_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.active_learning import run_active_learning_report

    prepared.active_learning.seed_size = 200
    prepared.active_learning.query_batch = 200
    prepared.active_learning.rounds = 2
    prepared.active_learning.max_pool = 2000
    out = run_active_learning_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "uncertainty" in text and "label" in text


@pytest.mark.slow
def test_provenance_writes_sbom_manifest_and_verifies(prepared: Settings) -> None:
    from netsentry.governance.provenance import (
        MANIFEST_NAME,
        SBOM_NAME,
        run_provenance_report,
        verify_manifest,
    )

    out = run_provenance_report(prepared)
    reports = prepared.paths.reports_dir
    assert out.exists()
    sbom = json.loads((reports / SBOM_NAME).read_text(encoding="utf-8"))
    assert sbom["bomFormat"] == "CycloneDX" and sbom["components"]
    # The manifest names the freshly-built bundle, and verify agrees it is intact.
    manifest = json.loads((reports / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["bundle"]["sha256"]
    bundle_path = prepared.paths.models_dir / manifest["bundle"]["name"]
    assert verify_manifest(reports / MANIFEST_NAME, bundle_path).ok


@pytest.mark.slow
def test_rules_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.rules import run_rules_report

    out = run_rules_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "matched" in text and "per-rule performance" in text


@pytest.mark.slow
def test_poisoning_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.poisoning import run_poisoning_report

    prepared.poisoning.label_flip_rates = [0.0, 0.25]
    prepared.poisoning.contamination_rates = [0.0, 0.1]
    out = run_poisoning_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "poisoning" in text and "contamination" in text


@pytest.mark.slow
def test_robustness_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.report import run_robustness_report

    prepared.robustness.mimicry_fractions = [0.0, 0.5, 1.0]
    prepared.robustness.search_budgets = [0.0, 1.0]
    prepared.robustness.search_iterations = 10
    prepared.robustness.max_attack_samples = 300
    out = run_robustness_report(prepared)
    assert out.exists() and "evasion" in out.read_text(encoding="utf-8").lower()
