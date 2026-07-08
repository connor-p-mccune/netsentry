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
def test_ablation_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.ablation import run_ablation_report

    out = run_ablation_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "ablation" in text and "family removed" in text


@pytest.mark.slow
def test_subgroups_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.subgroups import run_subgroups_report

    prepared.subgroups.min_support = 20  # the 6k-row fixture has small service buckets
    out = run_subgroups_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "parity" in text and "service" in text


@pytest.mark.slow
def test_novelty_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.novelty import run_novelty_report

    prepared.novelty.n_bins = 3
    prepared.novelty.max_queries = 500
    out = run_novelty_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "novelty" in text and "distance bin" in text


@pytest.mark.slow
def test_label_audit_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.label_audit import run_label_audit_report

    prepared.label_audit.folds = 2
    prepared.label_audit.max_rows = 2500
    out = run_label_audit_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "planted" in text and "recovery" in text


@pytest.mark.slow
def test_lodo_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.lodo import run_lodo_report

    out = run_lodo_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "held-out day" in text and "monday" in text


@pytest.mark.slow
def test_streaming_report_is_written(prepared: Settings) -> None:
    from netsentry.monitoring.streaming import run_streaming_report

    prepared.streaming.n_batches = 3
    out = run_streaming_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "prequential" in text and "retrain" in text


@pytest.mark.slow
def test_retrain_policy_report_is_written(prepared: Settings) -> None:
    from netsentry.monitoring.retrain_policy import run_retrain_policy_report

    prepared.retrain_policy.n_batches = 3
    prepared.retrain_policy.periodic_every = 2
    prepared.retrain_policy.cooldown_batches = 1
    out = run_retrain_policy_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "frontier" in text and "drift-triggered" in text


@pytest.mark.slow
def test_selftrain_report_is_written(prepared: Settings) -> None:
    from netsentry.training.selftrain import run_selftrain_report

    out = run_selftrain_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "pseudo-label audit" in text and "oracle" in text
    # The three-model comparison table is present with the operating columns.
    assert "self-trained (pseudo-labels)" in text and "detection @ threshold" in text


@pytest.mark.slow
def test_poisoning_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.poisoning import run_poisoning_report

    prepared.poisoning.label_flip_rates = [0.0, 0.25]
    prepared.poisoning.contamination_rates = [0.0, 0.1]
    out = run_poisoning_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "poisoning" in text and "contamination" in text


@pytest.mark.slow
def test_distill_report_is_written(prepared: Settings) -> None:
    from netsentry.explain.distill import run_distill_report

    prepared.distill.depths = [2, 3]
    prepared.distill.report_depth = 3
    prepared.distill.min_samples_leaf = 20
    out = run_distill_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "fidelity" in text and "leaves" in text


@pytest.mark.slow
def test_seed_variance_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.seed_variance import run_seed_variance_report

    prepared.seed_variance.n_seeds = 2
    prepared.evaluation.bootstrap_samples = 50
    out = run_seed_variance_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "reproducibility" in text and "noise" in text


@pytest.mark.slow
def test_gate_report_verdict_and_exit_state(prepared: Settings) -> None:
    from netsentry.evaluation.gate import run_gate

    # Metric floors are relaxed for the tiny fixture; the structural checks are the point.
    prepared.gate.min_tpr_at_primary_fpr = 0.0
    prepared.gate.max_ece = 1.0
    out, result = run_gate(prepared)
    text = out.read_text(encoding="utf-8")
    assert out.exists() and "leakage firewall" in text
    assert result.ok and "**PASS**" in text

    # An impossible floor must flip the verdict — the CI-enforcement path.
    prepared.gate.min_pr_auc_lift = 100.0
    _, failing = run_gate(prepared)
    assert not failing.ok


@pytest.mark.slow
def test_promotion_lifecycle_bootstrap_hold_and_rollforward(prepared: Settings) -> None:
    from netsentry.models.promotion import CHAMPION_POINTER, HISTORY_NAME, run_promotion
    from netsentry.training.train_supervised import train_supervised

    prepared.split.strategy = "temporal"
    prepared.supervised.task = "binary"
    prepared.promotion.n_boot = 50
    train_supervised(prepared)

    # Every persisted training bundle carries behavioral canaries the same runtime
    # must reproduce — the deployable artifact is attestable, not just the serving one.
    from netsentry.models.registry import load_bundle
    from netsentry.serving.canary import run_canary

    trained = prepared.paths.models_dir / "supervised_binary_temporal.joblib"
    replay = run_canary(load_bundle(trained))
    assert replay.present and replay.ok

    # 1) Empty registry: the first candidate seeds the champion.
    out, first = run_promotion(prepared)
    assert first.promote and first.champion is None
    assert (prepared.paths.models_dir / CHAMPION_POINTER).exists()

    # 2) Superiority policy vs an identical challenger: parity is not evidence.
    prepared.promotion.policy = "superiority"
    _, held = run_promotion(prepared)
    assert not held.promote and held.pr_delta is not None
    assert held.pr_delta.diff == 0.0  # identical bundle scores identically

    # 3) Non-inferiority policy: parity rolls forward (freshness wins under drift).
    prepared.promotion.policy = "non_inferiority"
    out, rolled = run_promotion(prepared)
    assert rolled.promote
    history = (prepared.paths.models_dir / HISTORY_NAME).read_text(encoding="utf-8")
    assert len(history.strip().splitlines()) == 3  # every decision is on the record
    assert "PROMOTE" in out.read_text(encoding="utf-8")


@pytest.mark.slow
def test_robustness_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.report import run_robustness_report

    prepared.robustness.mimicry_fractions = [0.0, 0.5, 1.0]
    prepared.robustness.search_budgets = [0.0, 1.0]
    prepared.robustness.search_iterations = 10
    prepared.robustness.max_attack_samples = 300
    out = run_robustness_report(prepared)
    assert out.exists() and "evasion" in out.read_text(encoding="utf-8").lower()
