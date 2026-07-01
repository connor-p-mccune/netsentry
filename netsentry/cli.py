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
def conformal(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Build conformal prediction sets with a coverage guarantee and selective alerting."""
    from netsentry.evaluation.conformal import run_conformal_report

    settings = _load(config, override)
    out = run_conformal_report(settings)
    logger.info("Conformal report ready", extra={"path": str(out)})


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
def benchmark(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    url: Annotated[str, typer.Option(help="Base URL of the service.")] = DEFAULT_BENCH_URL,
    requests: Annotated[int, typer.Option(help="Requests to send.")] = 500,
) -> None:
    """Drive the API and report p50/p95/p99 latency and throughput."""
    from netsentry.serving.benchmark import run_benchmark

    settings = _load(config, override)
    run_benchmark(settings, base_url=url, n_requests=requests)


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
