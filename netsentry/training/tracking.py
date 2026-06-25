"""Experiment tracking.

Wraps MLflow when it is installed and enabled; otherwise degrades to a local
JSON run log, so every run is recorded (params, metrics, artifacts, environment,
seed) and remains reproducible regardless of the environment.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from netsentry.log import get_logger
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "torch", "shap", "mlflow")


class Tracker(Protocol):
    """Minimal tracking interface used by training entrypoints."""

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_dict(self, payload: dict[str, Any], name: str) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in _TRACKED_PACKAGES:
        try:
            env[f"lib.{pkg}"] = version(pkg)
        except PackageNotFoundError:
            continue
    return env


class _MlflowTracker:
    def __init__(self, mlflow_mod: Any) -> None:
        self._mlflow = mlflow_mod

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_dict(self, payload: dict[str, Any], name: str) -> None:
        self._mlflow.log_dict(payload, name)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))


class _LocalTracker:
    """Filesystem fallback that mirrors the MLflow tracker's interface."""

    def __init__(self, settings: Settings, run_name: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.run_dir = settings.paths.mlruns_dir / "local" / f"{run_name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._metrics.update(metrics)

    def log_dict(self, payload: dict[str, Any], name: str) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def log_artifact(self, path: Path) -> None:
        if path.exists():
            shutil.copy2(path, self.run_dir / path.name)

    def flush(self) -> None:
        self.log_dict(self._params, "params.json")
        self.log_dict(self._metrics, "metrics.json")
        logger.info("Logged run locally", extra={"run_dir": str(self.run_dir)})


def _try_setup_mlflow(settings: Settings) -> Any | None:
    """Configure MLflow, returning the module or None to fall back to local."""
    if not (settings.mlflow.enabled and is_available("mlflow")):
        return None
    try:
        import mlflow

        # Recent MLflow gates the file store behind this opt-out; keep the simple
        # ./mlruns layout rather than requiring a database backend.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri())
        mlflow.set_experiment(settings.mlflow.experiment_name)
    except Exception as exc:  # tracking must never break a training run
        logger.warning("MLflow setup failed (%s); using local tracking", exc)
        return None
    return mlflow


@contextmanager
def track_run(
    settings: Settings, run_name: str, tags: dict[str, str] | None = None
) -> Iterator[Tracker]:
    """Open a tracking run; yields a tracker that logs params/metrics/artifacts."""
    mlflow = _try_setup_mlflow(settings)
    if mlflow is not None:
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            tracker = _MlflowTracker(mlflow)
            tracker.log_params(_environment())
            yield tracker
        return

    local = _LocalTracker(settings, run_name)
    local.log_params(_environment())
    if tags:
        local.log_params(tags)
    try:
        yield local
    finally:
        local.flush()
