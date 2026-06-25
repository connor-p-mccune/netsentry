"""Inference engine: load the bundle once and score flows.

Returns class + probability + anomaly score + SHAP top features + threshold
profile. Implemented in Phase 8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


class InferenceEngine:
    """Holds the loaded pipeline+model bundle and serves predictions."""

    def __init__(self, settings: Settings) -> None:
        raise NotImplementedError("Implemented in Phase 8")
