"""Detection and guarding of optional heavy dependencies.

NetSentry's core install is intentionally light; LightGBM, SHAP, MLflow, and
PyTorch are extras. Modules that can use them import lazily and call
:func:`require` (to fail loudly with install instructions) or check
:func:`is_available` (to fall back gracefully).
"""

from __future__ import annotations

import importlib.util

# Maps an importable module name to the pip extra that provides it.
_EXTRA_FOR_MODULE = {
    "lightgbm": "train",
    "shap": "train",
    "mlflow": "train",
    "imblearn": "train",
    "optuna": "train",
    "seaborn": "train",
    "torch": "ae",
    "fastapi": "serve",
    "uvicorn": "serve",
    "prometheus_client": "serve",
}


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but not installed."""


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported without importing it."""
    return importlib.util.find_spec(module) is not None


def require(module: str, *, purpose: str) -> None:
    """Raise a clear, actionable error if ``module`` is not installed.

    Args:
        module: Importable module name (e.g. ``"lightgbm"``).
        purpose: Human-readable description of what needs it, used in the message.
    """
    if is_available(module):
        return
    extra = _EXTRA_FOR_MODULE.get(module, "all")
    raise OptionalDependencyError(
        f"{purpose} requires the optional dependency '{module}', which is not "
        f"installed. Install it with:  pip install 'netsentry[{extra}]'"
    )
