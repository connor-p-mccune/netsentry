"""NetSentry command-line interface.

A single Typer app exposing the pipeline stages: ``download``, ``prep``,
``train`` (``supervised``/``anomaly``), ``eval``, ``serve``, and ``benchmark``.
Each subcommand is a thin wrapper that loads config, configures logging, and
calls into the package — no business logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from netsentry import __version__
from netsentry.config import Settings, load_settings
from netsentry.log import configure_logging, get_logger

logger = get_logger("netsentry.cli")

app = typer.Typer(
    name="netsentry",
    help="Leakage-safe ML network intrusion detection (CIC-IDS2017).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
train_app = typer.Typer(help="Train models.", no_args_is_help=True)
app.add_typer(train_app, name="train")

DEFAULT_BENCH_URL = "http://127.0.0.1:8000"

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Base config YAML (default: configs/default.yaml)."),
]
OverrideOpt = Annotated[
    list[Path] | None,
    typer.Option("--override", "-o", help="Override YAML(s), merged in order."),
]


def _load(config: Path | None, override: list[Path] | None) -> Settings:
    """Resolve settings from the optional base config and overrides."""
    settings = load_settings(config, overrides=override)
    logger.debug("Loaded settings", extra={"seed": settings.seed})
    return settings


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"netsentry {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Logging level (DEBUG/INFO/WARNING).")] = "INFO",
    json_logs: Annotated[bool, typer.Option(help="Emit structured JSON logs.")] = False,
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(log_level, json_logs=json_logs)


@app.command()
def download(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    force: Annotated[bool, typer.Option(help="Re-download even if files exist.")] = False,
) -> None:
    """Fetch/locate the CIC-IDS2017 CSVs into data/raw and verify them."""
    from netsentry.data.download import download_dataset

    settings = _load(config, override)
    paths = download_dataset(settings, force=force)
    logger.info("Dataset ready", extra={"files": len(paths)})


@app.command()
def prep(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Clean the raw data and produce persisted, honest train/val/test splits."""
    from netsentry.data.clean import clean_raw
    from netsentry.data.split import make_splits

    settings = _load(config, override)
    processed = clean_raw(settings)
    make_splits(settings)
    logger.info("Prep complete", extra={"processed": str(processed)})


@app.command()
def validate(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    data: Annotated[
        Path | None, typer.Option(help="Dataset to validate (default: cleaned parquet).")
    ] = None,
) -> None:
    """Run data-quality gates against a dataset; exit non-zero on a structural failure."""
    from netsentry.data.validation import run_validation

    settings = _load(config, override)
    out, report = run_validation(settings, data)
    logger.info(
        "Data quality report ready",
        extra={"path": str(out), "ok": report.ok, "warnings": report.n_warn},
    )
    if not report.ok:
        raise typer.Exit(code=1)


@train_app.command("supervised")
def train_supervised_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Train the LightGBM/HistGB classifier on the temporal split; log to MLflow."""
    from netsentry.training.train_supervised import train_supervised

    settings = _load(config, override)
    train_supervised(settings)


@train_app.command("tune")
def tune_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    trials: Annotated[
        int | None, typer.Option(help="Number of search trials (overrides config).")
    ] = None,
    out: Annotated[
        Path, typer.Option(help="Where to write the tuned-params YAML override.")
    ] = Path("configs/tuned.yaml"),
) -> None:
    """Search supervised hyperparameters on validation (Optuna; test untouched)."""
    from netsentry.training.tune import tune_supervised, write_tuned_config

    settings = _load(config, override)
    if trials is not None:
        settings.supervised.tune_trials = trials
    result = tune_supervised(settings)
    write_tuned_config(result, out)
    logger.info(
        "Tuning done",
        extra={"best_val_pr_auc": round(result.best_value, 4), "out": str(out)},
    )


@train_app.command("anomaly")
def train_anomaly_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Train benign-only anomaly detectors; evaluate leave-one-attack-out."""
    from netsentry.training.train_anomaly import train_anomaly

    settings = _load(config, override)
    train_anomaly(settings)


@app.command("eval")
def evaluate(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Generate the operational metrics report and figures."""
    from netsentry.evaluation.report import run_evaluation

    settings = _load(config, override)
    out = run_evaluation(settings)
    logger.info("Evaluation report ready", extra={"path": str(out)})


@app.command()
def streaming(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Replay later-day flows as a stream; compare a static vs a retrained model."""
    from netsentry.monitoring.streaming import run_streaming_report

    settings = _load(config, override)
    out = run_streaming_report(settings)
    logger.info("Streaming report ready", extra={"path": str(out)})


@app.command()
def selftrain(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Self-training on the unlabeled stream: pseudo-labels vs the labeled ceiling."""
    from netsentry.training.selftrain import run_selftrain_report

    settings = _load(config, override)
    out = run_selftrain_report(settings)
    logger.info("Self-training report ready", extra={"path": str(out)})


@app.command()
def weaksup(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Train the detector from the signature rules alone (data programming, zero labels)."""
    from netsentry.training.weak_supervision import run_weak_supervision_report

    settings = _load(config, override)
    out = run_weak_supervision_report(settings)
    logger.info("Weak-supervision report ready", extra={"path": str(out)})


@app.command()
def refresh(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Price threshold refresh (frozen model, re-chosen cut) against retraining."""
    from netsentry.monitoring.refresh import run_refresh_report

    settings = _load(config, override)
    out = run_refresh_report(settings)
    logger.info("Refresh report ready", extra={"path": str(out)})


@app.command("retrainpolicy")
def retrain_policy_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Price retrain triggers on the stream: never / periodic / drift-triggered / every batch."""
    from netsentry.monitoring.retrain_policy import run_retrain_policy_report

    settings = _load(config, override)
    out = run_retrain_policy_report(settings)
    logger.info("Retrain-policy report ready", extra={"path": str(out)})


@app.command()
def drift(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    reference: Annotated[
        Path | None,
        typer.Option(help="Reference dataset (parquet/csv); default: temporal train split."),
    ] = None,
    current: Annotated[
        Path | None,
        typer.Option(help="Current dataset (parquet/csv); default: temporal test split."),
    ] = None,
) -> None:
    """Report feature/score drift (PSI) of a current dataset against a reference."""
    from netsentry.monitoring.report import run_drift_report

    settings = _load(config, override)
    out = run_drift_report(settings, reference_path=reference, current_path=current)
    logger.info("Drift report ready", extra={"path": str(out)})


@app.command()
def driftscan(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    reference: Annotated[
        Path | None,
        typer.Option(help="Reference dataset (parquet/csv); default: temporal train split."),
    ] = None,
    current: Annotated[
        Path | None,
        typer.Option(help="Current dataset (parquet/csv); default: temporal test split."),
    ] = None,
) -> None:
    """Significance-tested drift: per-feature KS+FDR plus online Page-Hinkley/DDM."""
    from netsentry.monitoring.report import run_drift_tests_report

    settings = _load(config, override)
    out = run_drift_tests_report(settings, reference_path=reference, current_path=current)
    logger.info("Statistical drift report ready", extra={"path": str(out)})


@app.command("exchangeability")
def exchangeability_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Anytime-valid drift detection: a conformal test martingale with a Ville false-alarm bound."""
    from netsentry.monitoring.exchangeability import run_exchangeability_report

    settings = _load(config, override)
    out = run_exchangeability_report(settings)
    logger.info("Exchangeability report ready", extra={"path": str(out)})


@app.command()
def robustness(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Evaluate adversarial-evasion robustness (mimicry + adaptive query search)."""
    from netsentry.robustness.report import run_robustness_report

    settings = _load(config, override)
    out = run_robustness_report(settings)
    logger.info("Robustness report ready", extra={"path": str(out)})


@app.command()
def poisoning(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Measure detection decay under training-set poisoning (label flips + contamination)."""
    from netsentry.robustness.poisoning import run_poisoning_report

    settings = _load(config, override)
    out = run_poisoning_report(settings)
    logger.info("Poisoning report ready", extra={"path": str(out)})


@app.command()
def harden(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Adversarially train against mimicry evasion, then re-measure robustness."""
    from netsentry.robustness.hardening import run_hardening_report

    settings = _load(config, override)
    out = run_hardening_report(settings)
    logger.info("Hardening report ready", extra={"path": str(out)})


@app.command()
def privacy(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit membership-inference leakage (does the model memorise its training data?)."""
    from netsentry.robustness.membership import run_membership_report

    settings = _load(config, override)
    out = run_membership_report(settings)
    logger.info("Membership-inference report ready", extra={"path": str(out)})


@app.command()
def dp(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Price the differential-privacy frontier: detection & leakage vs the epsilon budget."""
    from netsentry.robustness.dp import run_dp_report

    settings = _load(config, override)
    out = run_dp_report(settings)
    logger.info("Differential-privacy report ready", extra={"path": str(out)})


@app.command()
def extraction(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Steal the model by query (model extraction): fidelity, stolen detection, transfer evasion."""
    from netsentry.robustness.extraction import run_extraction_report

    settings = _load(config, override)
    out = run_extraction_report(settings)
    logger.info("Model-extraction report ready", extra={"path": str(out)})


@app.command()
def certify(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Certify robustness (randomized smoothing): a provable L2 radius per flow vs sigma."""
    from netsentry.robustness.certify import run_certify_report

    settings = _load(config, override)
    out = run_certify_report(settings)
    logger.info("Certification report ready", extra={"path": str(out)})


@app.command()
def backdoor(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Plant a trigger backdoor (BadNets), then defend it with spectral signatures."""
    from netsentry.robustness.backdoor import run_backdoor_report

    settings = _load(config, override)
    out = run_backdoor_report(settings)
    logger.info("Backdoor report ready", extra={"path": str(out)})


@app.command()
def sanitize(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit-and-drop poisoned training labels, then re-measure the decay curve."""
    from netsentry.robustness.sanitize import run_sanitize_report

    settings = _load(config, override)
    out = run_sanitize_report(settings)
    logger.info("Poisoning-defense report ready", extra={"path": str(out)})


@app.command()
def cost(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Find the cost-optimal decision threshold (SOC economics) and write the report."""
    from netsentry.evaluation.cost import run_cost_report

    settings = _load(config, override)
    out = run_cost_report(settings)
    logger.info("Cost report ready", extra={"path": str(out)})


@app.command()
def baserate(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Sweep the operating points across production base rates (the base-rate fallacy)."""
    from netsentry.evaluation.baserate import run_base_rate_report

    settings = _load(config, override)
    out = run_base_rate_report(settings)
    logger.info("Base-rate report ready", extra={"path": str(out)})


@app.command("alertqueue")
def alert_queue_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Capacity-constrained triage: detection vs analyst budget, and lift over random."""
    from netsentry.evaluation.alert_queue import run_alert_queue_report

    settings = _load(config, override)
    out = run_alert_queue_report(settings)
    logger.info("Alert-queue report ready", extra={"path": str(out)})


@app.command()
def socsim(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Simulate the analyst queue: FIFO vs score-priority attack-SLA under load."""
    from netsentry.evaluation.socsim import run_socsim_report

    settings = _load(config, override)
    out = run_socsim_report(settings)
    logger.info("SOC-sim report ready", extra={"path": str(out)})


@app.command()
def conformal(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Build conformal prediction sets with a coverage guarantee and selective alerting."""
    from netsentry.evaluation.conformal import run_conformal_report

    settings = _load(config, override)
    out = run_conformal_report(settings)
    logger.info("Conformal report ready", extra={"path": str(out)})


@app.command("adaptiveconformal")
def adaptive_conformal_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Restore conformal coverage online under drift (adaptive alpha, ACI)."""
    from netsentry.evaluation.adaptive_conformal import run_adaptive_conformal_report

    settings = _load(config, override)
    out = run_adaptive_conformal_report(settings)
    logger.info("Adaptive-conformal report ready", extra={"path": str(out)})


@app.command()
def recourse(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Counterfactual recourse for flagged flows: minimal change that would clear them."""
    from netsentry.explain.counterfactual import run_recourse_report

    settings = _load(config, override)
    out = run_recourse_report(settings)
    logger.info("Recourse report ready", extra={"path": str(out)})


@app.command()
def provenance(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Write the supply-chain SBOM + model-integrity manifest and provenance report."""
    from netsentry.governance.provenance import run_provenance_report

    settings = _load(config, override)
    out = run_provenance_report(settings)
    logger.info("Provenance report ready", extra={"path": str(out)})


@app.command()
def verify(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    manifest: Annotated[
        Path | None, typer.Option(help="Manifest path (default: reports_dir/model_manifest.json).")
    ] = None,
    bundle: Annotated[
        Path | None, typer.Option(help="Bundle to verify (default: the one named in the manifest).")
    ] = None,
) -> None:
    """Verify a deployed bundle against its provenance manifest (integrity gate)."""
    import json as _json

    from netsentry.governance.provenance import MANIFEST_NAME, verify_manifest

    settings = _load(config, override)
    manifest_path = manifest or (settings.paths.reports_dir / MANIFEST_NAME)
    if not manifest_path.exists():
        logger.error("No manifest at %s; run `netsentry provenance` first.", manifest_path)
        raise typer.Exit(code=2)
    # The manifest records only the bundle's *name* (portable, no absolute paths), so
    # resolve it against the models directory where bundles actually live.
    if bundle is None:
        name = _json.loads(manifest_path.read_text(encoding="utf-8")).get("bundle", {}).get("name")
        if name:
            bundle = settings.paths.models_dir / name
    result = verify_manifest(manifest_path, bundle)
    for name, passed, detail in result.checks:
        logger.info("verify: %s %s (%s)", name, "OK" if passed else "FAIL", detail)
    if not result.ok:
        raise typer.Exit(code=1)
    logger.info("Bundle integrity verified", extra={"manifest": str(manifest_path)})


@app.command()
def canary(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    bundle: Annotated[
        Path | None, typer.Option(help="Bundle to check (default: the served bundle).")
    ] = None,
) -> None:
    """Replay the bundle's behavioral canaries; non-zero exit if this runtime skews them."""
    from netsentry.models.registry import latest_bundle, load_bundle
    from netsentry.serving.canary import run_canary

    settings = _load(config, override)
    path = bundle or settings.serving.artifact_path or latest_bundle(settings)
    if path is None or not Path(path).exists():
        logger.error("No model bundle found; build one first (train / serve).")
        raise typer.Exit(code=2)
    result = run_canary(load_bundle(Path(path)))
    logger.info(
        "Canary check",
        extra={
            "bundle": Path(path).name,
            "present": result.present,
            "ok": result.ok,
            "max_delta": result.max_delta,
            "message": result.message,
        },
    )
    if not result.present:
        raise typer.Exit(code=2)  # distinct from behavioral failure: nothing to check
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("modelcard")
def modelcard_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Auto-generate the model-card spec sheet from the deployed bundle."""
    from netsentry.evaluation.model_card import generate_model_card

    settings = _load(config, override)
    out = generate_model_card(settings)
    logger.info("Model card ready", extra={"path": str(out)})


@app.command("activelearning")
def active_learning_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Compare uncertainty vs random labeling (analyst-budget label efficiency)."""
    from netsentry.evaluation.active_learning import run_active_learning_report

    settings = _load(config, override)
    out = run_active_learning_report(settings)
    logger.info("Active-learning report ready", extra={"path": str(out)})


@app.command("learningcurve")
def learning_curve_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Plot PR-AUC vs training size (does more data help?) for both splits."""
    from netsentry.evaluation.learning_curve import run_learning_curve_report

    settings = _load(config, override)
    out = run_learning_curve_report(settings)
    logger.info("Learning-curve report ready", extra={"path": str(out)})


@app.command()
def ablation(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Leave-one-feature-family-out ablation: which behavioural families carry detection."""
    from netsentry.evaluation.ablation import run_ablation_report

    settings = _load(config, override)
    out = run_ablation_report(settings)
    logger.info("Ablation report ready", extra={"path": str(out)})


@app.command()
def distill(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Distill the model into an auditable tree; report fidelity and its cost."""
    from netsentry.explain.distill import run_distill_report

    settings = _load(config, override)
    out = run_distill_report(settings)
    logger.info("Distillation report ready", extra={"path": str(out)})


@app.command()
def exemplars(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit case-based explanations: do the nearest known flows vouch for alerts?"""
    from netsentry.explain.exemplars import run_exemplars_report

    settings = _load(config, override)
    out = run_exemplars_report(settings)
    logger.info("Exemplars report ready", extra={"path": str(out)})


@app.command("anomexplain")
def anomaly_explain_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Attribute anomaly flags to features (why is a flow abnormal?) + a faithfulness check."""
    from netsentry.explain.anomaly_explain import run_anomaly_explain_report

    settings = _load(config, override)
    out = run_anomaly_explain_report(settings)
    logger.info("Anomaly-explanation report ready", extra={"path": str(out)})


@app.command("pdp")
def partial_dependence_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Plot partial dependence + ICE for the top features (the model's response shape)."""
    from netsentry.explain.partial_dependence import run_partial_dependence_report

    settings = _load(config, override)
    out = run_partial_dependence_report(settings)
    logger.info("Partial-dependence report ready", extra={"path": str(out)})


@app.command()
def interactions(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Measure feature interactions (Friedman's H): which features the model has entangled."""
    from netsentry.explain.interactions import run_interactions_report

    settings = _load(config, override)
    out = run_interactions_report(settings)
    logger.info("Interactions report ready", extra={"path": str(out)})


@app.command()
def anchors(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Explain flagged flows with high-precision IF-THEN anchor rules (Ribeiro et al. 2018)."""
    from netsentry.explain.anchors import run_anchors_report

    settings = _load(config, override)
    out = run_anchors_report(settings)
    logger.info("Anchors report ready", extra={"path": str(out)})


@app.command("importance")
def importance_stability_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit feature-importance stability across bootstrap refits (are explanations stable?)."""
    from netsentry.explain.importance_stability import run_importance_stability_report

    settings = _load(config, override)
    out = run_importance_stability_report(settings)
    logger.info("Importance-stability report ready", extra={"path": str(out)})


@app.command()
def promote(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    challenger: Annotated[
        Path | None,
        typer.Option(help="Challenger bundle (default: the trained temporal bundle)."),
    ] = None,
    champion: Annotated[
        Path | None,
        typer.Option(help="Champion bundle (default: the registry's champion pointer)."),
    ] = None,
) -> None:
    """Champion/challenger promotion: paired-bootstrap deltas; non-zero exit on HOLD."""
    from netsentry.models.promotion import run_promotion

    settings = _load(config, override)
    out, decision = run_promotion(settings, challenger_path=challenger, champion_path=champion)
    logger.info(
        "Promotion report ready",
        extra={
            "path": str(out),
            "decision": "promote" if decision.promote else "hold",
            "reason": decision.reason,
        },
    )
    if not decision.promote:
        raise typer.Exit(code=1)


@app.command()
def gate(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Release quality gate: honesty invariants + metric floors; non-zero exit on failure."""
    from netsentry.evaluation.gate import run_gate

    settings = _load(config, override)
    out, result = run_gate(settings)
    logger.info(
        "Gate report ready",
        extra={"path": str(out), "ok": result.ok, "failed": result.n_failed},
    )
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def seeds(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Refit across seeds: the training-noise floor under every reported metric."""
    from netsentry.evaluation.seed_variance import run_seed_variance_report

    settings = _load(config, override)
    out = run_seed_variance_report(settings)
    logger.info("Seed-variance report ready", extra={"path": str(out)})


@app.command()
def leaderboard(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Benchmark model families under the identical honest protocol (both splits)."""
    from netsentry.evaluation.leaderboard import run_leaderboard_report

    settings = _load(config, override)
    out = run_leaderboard_report(settings)
    logger.info("Leaderboard report ready", extra={"path": str(out)})


@app.command("datavalue")
def data_value_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Value each training flow (KNN-Shapley): mislabel detection + value-guided pruning."""
    from netsentry.evaluation.data_value import run_data_value_report

    settings = _load(config, override)
    out = run_data_value_report(settings)
    logger.info("Data-valuation report ready", extra={"path": str(out)})


@app.command()
def ppi(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Estimate attack prevalence with prediction-powered inference (valid CIs from few labels)."""
    from netsentry.evaluation.ppi import run_ppi_report

    settings = _load(config, override)
    out = run_ppi_report(settings)
    logger.info("PPI report ready", extra={"path": str(out)})


@app.command("labelshift")
def label_shift_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Estimate + correct the deployment prior with zero labels (BBSE + MLLS/EM)."""
    from netsentry.evaluation.label_shift import run_label_shift_report

    settings = _load(config, override)
    out = run_label_shift_report(settings)
    logger.info("Label-shift report ready", extra={"path": str(out)})


@app.command()
def influence(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Explain a verdict by its most influential training flows (Koh & Liang 2017)."""
    from netsentry.explain.influence import run_influence_report

    settings = _load(config, override)
    out = run_influence_report(settings)
    logger.info("Influence report ready", extra={"path": str(out)})


@app.command()
def hmeasure(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Report the H-measure (Hand 2009): a coherent, cost-explicit alternative to ROC-AUC."""
    from netsentry.evaluation.hmeasure import run_hmeasure_report

    settings = _load(config, override)
    out = run_hmeasure_report(settings)
    logger.info("H-measure report ready", extra={"path": str(out)})


@app.command()
def leakage(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Reproduce the field's ~99% and attribute the inflation to each leakage source."""
    from netsentry.evaluation.leakage import run_leakage_report

    settings = _load(config, override)
    out = run_leakage_report(settings)
    logger.info("Leakage-attribution report ready", extra={"path": str(out)})


@app.command()
def slices(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Report per-attack-class detection on the temporal split (known vs novel)."""
    from netsentry.evaluation.slices import run_slices_report

    settings = _load(config, override)
    out = run_slices_report(settings)
    logger.info("Slices report ready", extra={"path": str(out)})


@app.command()
def campaigns(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Campaign-level detection: which (day, class) operations raise an alert, and when."""
    from netsentry.evaluation.campaigns import run_campaigns_report

    settings = _load(config, override)
    out = run_campaigns_report(settings)
    logger.info("Campaigns report ready", extra={"path": str(out)})


@app.command()
def subgroups(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit per-service detection/false-alarm parity at the single global threshold."""
    from netsentry.evaluation.subgroups import run_subgroups_report

    settings = _load(config, override)
    out = run_subgroups_report(settings)
    logger.info("Subgroups report ready", extra={"path": str(out)})


@app.command()
def novelty(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Profile detection vs distance-to-training (why shuffled splits flatter)."""
    from netsentry.evaluation.novelty import run_novelty_report

    settings = _load(config, override)
    out = run_novelty_report(settings)
    logger.info("Novelty report ready", extra={"path": str(out)})


@app.command("labelaudit")
def label_audit_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Audit training labels for likely errors (confident-learning style, self-validated)."""
    from netsentry.evaluation.label_audit import run_label_audit_report

    settings = _load(config, override)
    out = run_label_audit_report(settings)
    logger.info("Label-audit report ready", extra={"path": str(out)})


@app.command()
def lodo(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Leave-one-day-out: every day takes a turn as the held-out future."""
    from netsentry.evaluation.lodo import run_lodo_report

    settings = _load(config, override)
    out = run_lodo_report(settings)
    logger.info("LODO report ready", extra={"path": str(out)})


@app.command()
def rules(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Benchmark the hand-written signature ruleset against the model (matched FPR)."""
    from netsentry.evaluation.rules import run_rules_report

    settings = _load(config, override)
    out = run_rules_report(settings)
    logger.info("Rules report ready", extra={"path": str(out)})


@app.command()
def intel(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Write the MITRE ATT&CK coverage report for the detected attack classes."""
    from netsentry.intel.report import run_mitre_report

    settings = _load(config, override)
    out = run_mitre_report(settings)
    logger.info("MITRE ATT&CK report ready", extra={"path": str(out)})


@app.command()
def navigator(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Export detection coverage as a MITRE ATT&CK Navigator layer (scored by recall)."""
    from netsentry.intel.navigator import run_navigator_export

    settings = _load(config, override)
    out = run_navigator_export(settings)
    logger.info("ATT&CK Navigator layer ready", extra={"path": str(out)})


@app.command()
def sigma(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    out: Annotated[
        Path | None, typer.Option(help="Output directory (default: reports_dir/sigma).")
    ] = None,
) -> None:
    """Export the signature ruleset as portable Sigma detection rules for any SIEM."""
    from netsentry.intel.sigma import export_sigma_rules

    settings = _load(config, override)
    target = export_sigma_rules(settings, out)
    logger.info("Sigma rules ready", extra={"dir": str(target)})


@app.command()
def analyze(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Run the full analysis suite (eval, cost, conformal, robustness, drift) + index."""
    from netsentry.evaluation.analyze import run_full_analysis

    settings = _load(config, override)
    out = run_full_analysis(settings)
    logger.info("Analysis index ready", extra={"path": str(out)})


@app.command("crosseval")
def crosseval(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Score a foreign-schema dataset with the trained model (cross-dataset study)."""
    from netsentry.evaluation.cross_dataset import run_cross_dataset_eval

    settings = _load(config, override)
    out = run_cross_dataset_eval(settings)
    logger.info("Cross-dataset report ready", extra={"path": str(out)})


@app.command()
def transfer(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Price re-buying the FPR budget on a foreign dataset: quantile vs k labels vs oracle."""
    from netsentry.evaluation.transfer import run_transfer_report

    settings = _load(config, override)
    out = run_transfer_report(settings)
    logger.info("Threshold-transfer report ready", extra={"path": str(out)})


@app.command()
def triage(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    findings: Annotated[Path, typer.Option(help="JSON file of vulnpipe findings.")] = Path(
        "examples/vulnpipe_findings.json"
    ),
    out: Annotated[
        Path | None, typer.Option(help="Where to write the triaged report (markdown).")
    ] = None,
) -> None:
    """Re-rank vulnpipe findings by NetSentry traffic risk (severity + attack + anomaly)."""
    from netsentry.integrations.vulnpipe import (
        load_findings,
        render_triage_markdown,
        triage_findings,
    )
    from netsentry.models.registry import latest_bundle, load_bundle
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    bundle_path = settings.serving.artifact_path or latest_bundle(settings)
    if bundle_path is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        bundle_path = build_serving_bundle(settings)
    triaged = triage_findings(load_findings(findings), load_bundle(Path(bundle_path)), settings)

    out_path = out or (settings.paths.reports_dir / "triage.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_triage_markdown(triaged), encoding="utf-8")
    logger.info("Triaged findings", extra={"count": len(triaged), "path": str(out_path)})


@app.command()
def serve(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    host: Annotated[str | None, typer.Option(help="Bind host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port (overrides config).")] = None,
) -> None:
    """Run the FastAPI inference service."""
    import uvicorn

    from netsentry.models.registry import latest_bundle
    from netsentry.serving.app import create_app
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    app_obj = create_app(settings)
    uvicorn.run(
        app_obj,
        host=host or settings.serving.host,
        port=port or settings.serving.port,
    )


@app.command()
def score(
    input: Annotated[Path, typer.Option("--input", "-i", help="Flow file (CSV/Parquet).")],
    output: Annotated[Path, typer.Option("--output", help="Where to write predictions.")] = Path(
        "scored.csv"
    ),
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
) -> None:
    """Score a CSV/Parquet of flows to a predictions file (offline batch inference)."""
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.batch import score_file
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    stats = score_file(settings, input, output, profile=profile)
    logger.info("Scored file", extra={**stats, "output": str(output)})


@app.command()
def zeek(
    input: Annotated[
        Path, typer.Option("--input", "-i", help="Zeek conn.log (TSV or JSON lines).")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Where to write scored connections.")
    ] = Path("zeek_scored.csv"),
    flows_out: Annotated[
        Path | None, typer.Option(help="Also write the mapped CIC feature rows (CSV/Parquet).")
    ] = None,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
) -> None:
    """Score a Zeek conn.log: map connections into CIC features, run the model."""
    from netsentry.integrations.zeek import score_zeek_log
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    stats = score_zeek_log(settings, input, output, flows_out=flows_out, profile=profile)
    logger.info("Zeek log scored", extra={**stats, "output": str(output)})


@app.command()
def incident(
    input: Annotated[Path, typer.Option("--input", "-i", help="Flow file (CSV/Parquet).")],
    output: Annotated[
        Path, typer.Option("--output", help="Where to write the incident report.")
    ] = Path("incident_report.md"),
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
) -> None:
    """Score a flow file and fold the alerts into an analyst-ready incident report."""
    from netsentry.intel.incident import build_incident_report
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    stats = build_incident_report(settings, input, output, profile=profile)
    logger.info("Incident report ready", extra={**stats, "output": str(output)})


@app.command()
def beacon(
    input: Annotated[
        Path | None, typer.Option("--input", "-i", help="Flow file with identity + timestamp.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Where to write the beacon report.")
    ] = None,
    demo: Annotated[
        bool, typer.Option(help="Analyze a synthetic capture with a planted beacon.")
    ] = False,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Rank talker pairs by beacon-like periodicity (C2 the per-flow model can't see)."""
    from netsentry.intel.beacon import run_beacon_report

    settings = _load(config, override)
    if input is None and not demo:
        logger.error("Provide a flow file with --input, or use --demo.")
        raise typer.Exit(code=2)
    out = run_beacon_report(settings, input_path=input, output_path=output, demo=demo)
    logger.info("Beacon report ready", extra={"path": str(out)})


@app.command()
def graph(
    input: Annotated[
        Path | None, typer.Option("--input", "-i", help="Flow file with Src/Dst identity columns.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Where to write the host-graph report.")
    ] = None,
    demo: Annotated[
        bool, typer.Option(help="Analyze a synthetic capture with a planted scan and pivot chain.")
    ] = False,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Rank scan fan-out + lateral-movement chains (topology the per-flow model can't see)."""
    from netsentry.intel.graph import run_graph_report

    settings = _load(config, override)
    if input is None and not demo:
        logger.error("Provide a flow file with --input, or use --demo.")
        raise typer.Exit(code=2)
    out = run_graph_report(settings, input_path=input, output_path=output, demo=demo)
    logger.info("Host-graph report ready", extra={"path": str(out)})


@app.command()
def stix(
    input: Annotated[Path, typer.Option("--input", "-i", help="Flow file (CSV/Parquet).")],
    output: Annotated[
        Path, typer.Option("--output", help="Where to write the STIX 2.1 bundle.")
    ] = Path("stix_bundle.json"),
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
) -> None:
    """Score a flow file and export the detections as a STIX 2.1 threat-intel bundle."""
    from netsentry.intel.stix import build_stix_bundle
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    stats = build_stix_bundle(settings, input, output, profile=profile)
    logger.info("STIX bundle ready", extra={**stats, "output": str(output)})


@app.command()
def pcap(
    input: Annotated[
        Path | None, typer.Option("--input", "-i", help="Packet capture (pcap or pcapng).")
    ] = None,
    output: Annotated[Path, typer.Option("--output", help="Where to write scored flows.")] = Path(
        "pcap_scored.csv"
    ),
    flows_out: Annotated[
        Path | None, typer.Option(help="Also write the extracted CIC feature rows (CSV/Parquet).")
    ] = None,
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
    demo: Annotated[
        bool, typer.Option(help="Generate and score the synthetic demo capture.")
    ] = False,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Score a raw packet capture: parse packets, assemble CIC flows, run the model."""
    from netsentry.capture.score import score_capture
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle

    settings = _load(config, override)
    if demo:
        from netsentry.capture.demo import write_demo_pcap

        input = write_demo_pcap(input or Path("examples/demo_capture.pcap"), seed=settings.seed)
    if input is None:
        logger.error("Provide a capture with --input, or use --demo.")
        raise typer.Exit(code=2)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    summary = score_capture(settings, input, output, flows_out=flows_out, profile=profile)
    logger.info("Capture scored", extra={**summary, "output": str(output)})


@app.command()
def watch(
    spool: Annotated[
        Path, typer.Option("--spool", "-s", help="Directory to watch for new flow files.")
    ],
    alerts_out: Annotated[
        Path, typer.Option("--alerts", help="ECS JSON-lines alert file to append to.")
    ] = Path("alerts.ndjson"),
    state: Annotated[Path, typer.Option(help="State file tracking processed spool files.")] = Path(
        ".netsentry_watch_state.json"
    ),
    profile: Annotated[str | None, typer.Option(help="Threshold profile.")] = None,
    once: Annotated[
        bool, typer.Option(help="Drain the current backlog once and exit (else poll).")
    ] = False,
    interval: Annotated[float, typer.Option(help="Poll interval in seconds.")] = 5.0,
    emit_all: Annotated[
        bool, typer.Option(help="Emit every flow, not just the ones flagged as attacks.")
    ] = False,
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Watch a spool directory; score new flow files into ECS JSON-lines alerts."""
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.bundle import build_serving_bundle
    from netsentry.serving.watch import run_watch

    settings = _load(config, override)
    if settings.serving.artifact_path is None and latest_bundle(settings) is None:
        logger.info("No model bundle found; building a serving bundle (requires `prep`).")
        build_serving_bundle(settings)
    totals = run_watch(
        settings,
        spool=spool,
        alerts_out=alerts_out,
        state_path=state,
        profile=profile,
        once=once,
        interval=interval,
        emit_all=emit_all,
    )
    logger.info("Watch pass complete", extra={**totals, "alerts_out": str(alerts_out)})


@app.command()
def benchmark(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    url: Annotated[str, typer.Option(help="Base URL of the service.")] = DEFAULT_BENCH_URL,
    requests: Annotated[int, typer.Option(help="Requests to send.")] = 500,
    explain: Annotated[
        bool, typer.Option(help="Request SHAP explanations (--no-explain drives the fast path).")
    ] = True,
) -> None:
    """Drive the API and report p50/p95/p99 latency and throughput."""
    from netsentry.serving.benchmark import run_benchmark

    settings = _load(config, override)
    run_benchmark(settings, base_url=url, n_requests=requests, explain=explain)


@app.command()
def demo(
    port: Annotated[int, typer.Option(help="Port for the Streamlit dashboard.")] = 8501,
) -> None:
    """Launch the Streamlit demo dashboard (needs the demo extra: pip install '.[demo]')."""
    import subprocess
    import sys

    dashboard = Path(__file__).resolve().parent / "demo" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard), "--server.port", str(port)]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


@app.command("onnx")
def onnx_export(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Export the model to ONNX, quantize, and benchmark against the sklearn path."""
    from netsentry.serving.onnx_export import run_onnx_export

    settings = _load(config, override)
    out = run_onnx_export(settings)
    logger.info("ONNX report ready", extra={"path": str(out)})


if __name__ == "__main__":
    app()
