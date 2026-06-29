"""Rolling-window drift monitor for the serving layer.

Holds a compact reference summary (per-feature bin edges + reference
proportions). Incoming flows are buffered; once a full window accumulates, PSI
per feature is computed against the reference and returned for export as
Prometheus gauges, then the window resets (tumbling). All of this is best-effort:
drift monitoring must never break — or meaningfully slow — a prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from netsentry.monitoring.drift import (
    DriftReport,
    bin_proportions,
    psi_from_proportions,
    quantile_bin_edges,
)


def reference_summary(reference: pd.DataFrame, columns: list[str], *, bins: int) -> dict[str, Any]:
    """Compact, picklable drift reference: bin edges + proportions per column.

    Stored in the serving bundle so the deployed model carries its own drift
    reference and serving needs no access to the processed dataset.
    """
    edges: dict[str, np.ndarray] = {}
    props: dict[str, np.ndarray] = {}
    for col in columns:
        if col in reference.columns:
            col_edges = quantile_bin_edges(reference[col].to_numpy(), bins)
            edges[col] = col_edges
            props[col] = bin_proportions(reference[col].to_numpy(), col_edges)
    return {"bins": bins, "edges": edges, "props": props}


class DriftMonitor:
    """Accumulate served flows and compute windowed feature PSI vs a reference."""

    def __init__(
        self,
        edges: dict[str, np.ndarray],
        ref_props: dict[str, np.ndarray],
        *,
        window: int,
        moderate: float,
        major: float,
    ) -> None:
        self._edges = edges
        self._ref_props = ref_props
        self.window = max(1, window)
        self.moderate = moderate
        self.major = major
        self._buffer: dict[str, list[float]] = {col: [] for col in edges}
        self._count = 0

    @classmethod
    def from_summary(
        cls, summary: dict[str, Any], *, window: int, moderate: float, major: float
    ) -> DriftMonitor:
        """Reconstruct a monitor from a bundle's stored reference summary."""
        return cls(
            summary["edges"], summary["props"], window=window, moderate=moderate, major=major
        )

    def observe(self, frame: pd.DataFrame) -> DriftReport | None:
        """Buffer a batch of raw flows; return a report once a window completes."""
        for col, buf in self._buffer.items():
            if col in frame.columns:
                buf.extend(frame[col].to_numpy(dtype=float).tolist())
        self._count += len(frame)
        if self._count < self.window:
            return None
        report = self._report()
        self._buffer = {col: [] for col in self._edges}
        self._count = 0
        return report

    def _report(self) -> DriftReport:
        feature_psi: dict[str, float] = {}
        for col, edges in self._edges.items():
            values = np.asarray(self._buffer[col], dtype=float)
            if values.size:
                feature_psi[col] = psi_from_proportions(
                    self._ref_props[col], bin_proportions(values, edges)
                )
        return DriftReport(feature_psi=feature_psi, moderate=self.moderate, major=self.major)
