"""Honest train/val/test splitting.

Provides the temporal/by-day headline split, a stratified reference split, and a
leave-one-attack-out helper for the anomaly detector. Validation is always carved
from TRAIN only, and splits are persisted with a content hash for
reproducibility. Implemented in Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def make_splits(settings: Settings) -> object:
    """Produce persisted train/val/test splits per the configured strategy."""
    raise NotImplementedError("Implemented in Phase 3")
