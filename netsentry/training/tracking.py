"""Experiment tracking.

Wraps MLflow when available and degrades to a local JSON run log otherwise, so a
run is always recorded (params, metrics, artifacts, env, seed). Implemented in
Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def start_run(settings: Settings, run_name: str) -> object:
    """Open a tracking run context."""
    raise NotImplementedError("Implemented in Phase 4")
