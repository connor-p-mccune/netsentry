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


@train_app.command("supervised")
def train_supervised_cmd(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
) -> None:
    """Train the LightGBM/HistGB classifier on the temporal split; log to MLflow."""
    from netsentry.training.train_supervised import train_supervised

    settings = _load(config, override)
    train_supervised(settings)


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
def serve(
    config: ConfigOpt = None,
    override: OverrideOpt = None,
    host: Annotated[str | None, typer.Option(help="Bind host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port (overrides config).")] = None,
) -> None:
    """Run the FastAPI inference service."""
    import uvicorn

    from netsentry.serving.app import create_app

    settings = _load(config, override)
    app_obj = create_app(settings)
    uvicorn.run(
        app_obj,
        host=host or settings.serving.host,
        port=port or settings.serving.port,
    )


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


if __name__ == "__main__":
    app()
