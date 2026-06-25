"""Train benign-only anomaly detectors and evaluate leave-one-attack-out.

Implemented in Phase 6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def train_anomaly(settings: Settings) -> object:
    """Train benign-only detectors and report held-out attack detection."""
    raise NotImplementedError("Implemented in Phase 6")
