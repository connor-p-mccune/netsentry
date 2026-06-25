"""Unsupervised novel-attack detection.

Isolation Forest (benign-fit) and a benign-only PyTorch autoencoder whose
reconstruction error is thresholded at a target FPR on a benign validation set.
Implemented in Phase 6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def build_anomaly_detector(settings: Settings, kind: str) -> object:
    """Construct an anomaly detector (``iforest`` or ``autoencoder``)."""
    raise NotImplementedError("Implemented in Phase 6")
