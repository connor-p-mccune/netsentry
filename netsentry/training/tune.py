"""Hyperparameter optimization for the supervised classifier — validation only.

Wires the previously-unused ``supervised.tune`` / ``tune_trials`` config. The
search is **leakage-safe by construction**: the pipeline is fit on train, every
trial trains on train with early stopping on validation and is scored by
validation PR-AUC, and the **test split is never touched**. Optuna (TPE) is used
when installed; otherwise it degrades to a seeded random search over the same
space, so tuning runs anywhere and is reproducible from the seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.train_supervised import quick_metrics
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

# Tunable fields and their ranges. (low, high, log) for floats; ("int", low, high) ints.
_FLOAT_SPACE: dict[str, tuple[float, float, bool]] = {
    "learning_rate": (0.01, 0.3, True),
    "subsample": (0.5, 1.0, False),
    "colsample_bytree": (0.5, 1.0, False),
    "reg_lambda": (1e-3, 10.0, True),
}
_INT_SPACE: dict[str, tuple[int, int]] = {
    "num_leaves": (15, 255),
    "max_depth": (3, 12),
    "min_child_samples": (5, 200),
}


@dataclass
class TuneResult:
    """The outcome of a hyperparameter search."""

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    method: str  # "optuna" or "random"


def suggest_random(rng: np.random.Generator) -> dict[str, Any]:
    """Sample one hyperparameter configuration from the search space."""
    params: dict[str, Any] = {}
    for name, (low, high, log) in _FLOAT_SPACE.items():
        params[name] = (
            float(np.exp(rng.uniform(np.log(low), np.log(high))))
            if log
            else float(rng.uniform(low, high))
        )
    for name, (low, high) in _INT_SPACE.items():
        params[name] = int(rng.integers(low, high + 1))
    return params


def apply_params(settings: Settings, params: dict[str, Any]) -> Settings:
    """Return a settings copy with the supervised hyperparameters overridden."""
    variant = settings.model_copy(deep=True)
    for key, value in params.items():
        if hasattr(variant.supervised, key):
            setattr(variant.supervised, key, value)
    return variant


def _target(task: str) -> str:
    return BINARY_TARGET if task == "binary" else MULTICLASS_TARGET


def _make_objective(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[dict[str, Any]], float]:
    """Build the val-PR-AUC objective: train on train, score on validation."""
    task = settings.supervised.task

    def objective(params: dict[str, Any]) -> float:
        variant = apply_params(settings, params)
        model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
        return quick_metrics(y_val, model.predict_proba(x_val), model.classes_, task)["pr_auc"]

    return objective


def tune_supervised(settings: Settings) -> TuneResult:
    """Search supervised hyperparameters on validation PR-AUC (test untouched)."""
    seed_everything(settings.seed)
    strategy = settings.split.strategy
    task = settings.supervised.task
    target = _target(task)

    train = load_split(settings, strategy, "train")
    val = load_split(settings, strategy, "val")
    pipeline = build_pipeline(settings)
    x_train = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_val = pipeline.transform(val)
    y_train, y_val = train[target].to_numpy(), val[target].to_numpy()

    objective = _make_objective(settings, x_train, y_train, x_val, y_val)
    n_trials = settings.supervised.tune_trials

    if is_available("optuna"):
        best_params, best_value = _optuna_search(objective, n_trials, settings.seed)
        method = "optuna"
    else:
        logger.info("Optuna not installed; using a seeded random search fallback.")
        best_params, best_value = _random_search(objective, n_trials, settings.seed)
        method = "random"

    logger.info(
        "Tuning complete",
        extra={"method": method, "best_val_pr_auc": round(best_value, 4), "trials": n_trials},
    )
    return TuneResult(
        best_params=best_params, best_value=best_value, n_trials=n_trials, method=method
    )


def write_tuned_config(result: TuneResult, path: Path) -> Path:
    """Write the best hyperparameters as a YAML override the CLI can consume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "supervised": {**result.best_params, "tune": False},
        # Provenance, not consumed by Settings (extra keys are ignored on load).
        "_tuning": {
            "best_val_pr_auc": round(result.best_value, 6),
            "n_trials": result.n_trials,
            "method": result.method,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.info("Wrote tuned config", extra={"path": str(path)})
    return path


def _random_search(
    objective: Callable[[dict[str, Any]], float], n_trials: int, seed: int
) -> tuple[dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    best_params: dict[str, Any] = {}
    best_value = -np.inf
    for _ in range(n_trials):
        params = suggest_random(rng)
        value = objective(params)
        if value > best_value:
            best_value, best_params = value, params
    return best_params, float(best_value)


def _optuna_search(
    objective: Callable[[dict[str, Any]], float], n_trials: int, seed: int
) -> tuple[dict[str, Any], float]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optuna_objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        for name, (low, high, log) in _FLOAT_SPACE.items():
            params[name] = trial.suggest_float(name, low, high, log=log)
        for name, (low, high) in _INT_SPACE.items():
            params[name] = trial.suggest_int(name, low, high)
        return objective(params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)
