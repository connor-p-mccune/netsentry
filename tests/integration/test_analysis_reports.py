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
def test_campaigns_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.campaigns import run_campaigns_report

    out = run_campaigns_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "campaign" in text and "first alert" in text


@pytest.mark.slow
def test_leaderboard_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.leaderboard import run_leaderboard_report

    prepared.leaderboard.families = ["majority", "logistic", "gbdt"]
    out = run_leaderboard_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "temporal split" in text and "stratified split" in text
    assert "logistic regression" in text and "gap" in text


@pytest.mark.slow
def test_selftrain_report_is_written(prepared: Settings) -> None:
    from netsentry.training.selftrain import run_selftrain_report

    out = run_selftrain_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "pseudo-label audit" in text and "oracle" in text
    # The three-model comparison table is present with the operating columns.
    assert "self-trained (pseudo-labels)" in text and "detection @ threshold" in text


@pytest.mark.slow
def test_backdoor_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.backdoor import run_backdoor_report

    prepared.backdoor.poison_rates = [0.005, 0.02]
    out = run_backdoor_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "attack success rate" in text and "spectral" in text
    assert "trigger" in text and "badnets" in text


@pytest.mark.slow
def test_weak_supervision_report_is_written(prepared: Settings) -> None:
    from netsentry.training.weak_supervision import run_weak_supervision_report

    out = run_weak_supervision_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "labeling function" in text and "zero labels" in text
    # Both tables render: the per-signature precision audit and the detector comparison.
    assert "model precision (no labels)" in text and "matched volume" in text
    assert "supervised ceiling" in text


@pytest.mark.slow
def test_poisoning_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.poisoning import run_poisoning_report

    prepared.poisoning.label_flip_rates = [0.0, 0.25]
    prepared.poisoning.contamination_rates = [0.0, 0.1]
    out = run_poisoning_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "poisoning" in text and "contamination" in text


@pytest.mark.slow
def test_poisoning_defense_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.sanitize import run_sanitize_report

    prepared.sanitize.flip_rates = [0.0, 0.5]
    prepared.sanitize.max_rows = 3000
    prepared.label_audit.folds = 2
    out = run_sanitize_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "sanitiz" in text and "undefended" in text
    assert "flips caught" in text and "clean rows lost" in text


@pytest.mark.slow
def test_socsim_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.socsim import run_socsim_report

    prepared.socsim.n_runs = 3
    prepared.socsim.arrivals_per_shift = 120
    prepared.socsim.analyst_counts = [2, 4]
    out = run_socsim_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "fifo" in text and "score-priority" in text
    assert "attack-sla" in text and "offered load" in text


@pytest.mark.slow
def test_threshold_transfer_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.transfer import run_transfer_report
    from netsentry.serving.bundle import build_serving_bundle

    build_serving_bundle(prepared)
    prepared.crossdata.rows = 3000
    prepared.transfer.label_budgets = [50, 500]
    prepared.transfer.n_resamples = 5
    out = run_transfer_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "transplant" in text and "oracle" in text
    assert "budget held" in text


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
def test_data_value_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.data_value import run_data_value_report

    prepared.data_value.reference_rows = 1200
    prepared.data_value.query_rows = 600
    prepared.data_value.prune_fractions = [0.1]
    out = run_data_value_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "knn-shapley" in text and "mislabel" in text
    assert "planted" in text and "pruning" in text


@pytest.mark.slow
def test_anchors_report_is_written(prepared: Settings) -> None:
    from netsentry.explain.anchors import run_anchors_report

    prepared.anchors.background_rows = 2000
    prepared.anchors.n_explained = 6
    prepared.anchors.min_match = 15
    out = run_anchors_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "anchor" in text and "precision" in text
    assert "coverage" in text and "if" in text


@pytest.mark.slow
def test_hmeasure_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.hmeasure import run_hmeasure_report

    prepared.hmeasure.grid_points = 500
    out = run_hmeasure_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "h-measure" in text and "roc-auc" in text
    assert "coherent" in text and "prior" in text


@pytest.mark.slow
def test_exchangeability_report_is_written(prepared: Settings) -> None:
    from netsentry.monitoring.exchangeability import run_exchangeability_report

    prepared.exchangeability.stream_len = 400
    prepared.exchangeability.change_point = 200
    prepared.exchangeability.n_null_streams = 5
    out = run_exchangeability_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "martingale" in text and "exchangeab" in text
    assert "ville" in text and "false-alarm" in text


@pytest.mark.slow
def test_ppi_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.ppi import run_ppi_report

    prepared.ppi.label_budgets = [100, 400]
    prepared.ppi.n_trials = 40
    out = run_ppi_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "prediction-powered" in text and "rectifier" in text
    assert "coverage" in text and "prevalence" in text


@pytest.mark.slow
def test_influence_report_is_written(prepared: Settings) -> None:
    from netsentry.explain.influence import run_influence_report

    prepared.influence.max_train = 1500
    prepared.influence.loo_sample = 12
    prepared.influence.n_explained = 2
    out = run_influence_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "influence" in text and "leave-one-out" in text
    assert "self-influence" in text and "training flow" in text


@pytest.mark.slow
def test_label_shift_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.label_shift import run_label_shift_report

    prepared.label_shift.target_priors = [0.05, 0.4]
    prepared.label_shift.n_trials = 8
    prepared.label_shift.target_size = 1500
    out = run_label_shift_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "bbse" in text and "mlls" in text
    assert "calibration" in text and "pr-auc" in text


@pytest.mark.slow
def test_leakage_report_is_written(prepared: Settings) -> None:
    from netsentry.evaluation.leakage import run_leakage_report

    prepared.leakage.max_rows = 3000
    out = run_leakage_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "leakage" in text and "shuffled split" in text
    # The ladder must reproduce inflation: the leaky end beats the honest baseline.
    assert "session identifier" in text and "manufactur" in text


@pytest.mark.slow
def test_membership_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.membership import run_membership_report

    prepared.membership.target_train_rows = 1500
    prepared.membership.eval_rows = 800
    prepared.membership.shadow_rows = 1500
    prepared.membership.n_shadow = 3
    out = run_membership_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "membership" in text and "shadow" in text
    assert "advantage" in text and "overfit reference" in text


@pytest.mark.slow
def test_anomaly_explain_report_is_written(prepared: Settings) -> None:
    from netsentry.explain.anomaly_explain import run_anomaly_explain_report

    prepared.anomaly.detectors = ["iforest"]  # no Torch assumption in the test path
    prepared.anomaly_explain.max_explained = 100
    prepared.anomaly_explain.min_class_flags = 3
    out = run_anomaly_explain_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "anomaly" in text and "occlusion" in text
    assert "faithfulness" in text and "iforest" in text


@pytest.mark.slow
def test_dp_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.dp import run_dp_report

    prepared.dp.noise_multipliers = [0.0, 1.0, 4.0]
    prepared.dp.target_train_rows = 1500
    prepared.dp.eval_rows = 800
    prepared.dp.epochs = 20
    out = run_dp_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "differential privacy" in text and "frontier" in text
    # The non-private reference and a private (finite-epsilon) row are both present.
    assert "non-private" in text and "membership auc" in text


@pytest.mark.slow
def test_interactions_report_is_written(prepared: Settings) -> None:
    from netsentry.explain.interactions import run_interactions_report

    prepared.interactions.top_k = 4
    prepared.interactions.sample_rows = 60
    prepared.supervised.n_estimators = 40
    out = run_interactions_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "h-statistic" in text and "interaction" in text
    assert "additive" in text


@pytest.mark.slow
def test_extraction_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.extraction import run_extraction_report

    prepared.extraction.query_budgets = [200, 600]
    prepared.extraction.max_eval_rows = 800
    prepared.extraction.transfer_iterations = 15
    prepared.extraction.max_attack_samples = 300
    out = run_extraction_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "model extraction" in text and "fidelity" in text
    # The defense axis and the transfer attack are both rendered.
    assert "top-1 label only" in text and "transfer" in text


@pytest.mark.slow
def test_certify_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.certify import run_certify_report

    prepared.certify.sigmas = [0.25, 0.5]
    prepared.certify.n_samples = 200
    prepared.certify.max_flows = 40
    out = run_certify_report(prepared)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "certified" in text and "randomized smoothing" in text
    assert "radius" in text and "sigma" in text


@pytest.mark.slow
def test_robustness_report_is_written(prepared: Settings) -> None:
    from netsentry.robustness.report import run_robustness_report

    prepared.robustness.mimicry_fractions = [0.0, 0.5, 1.0]
    prepared.robustness.search_budgets = [0.0, 1.0]
    prepared.robustness.search_iterations = 10
    prepared.robustness.max_attack_samples = 300
    out = run_robustness_report(prepared)
    assert out.exists() and "evasion" in out.read_text(encoding="utf-8").lower()
