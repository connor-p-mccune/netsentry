"""Train the supervised classifier on the temporal split and log to MLflow.

Saves the deployable pipeline+model bundle. Implemented in Phase 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def train_supervised(settings: Settings) -> object:
    """Run supervised training end-to-end; return a result summary."""
    raise NotImplementedError("Implemented in Phase 4")
