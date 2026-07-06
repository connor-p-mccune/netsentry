"""Behavioral canaries — does the deployed artifact still predict what it predicted?

``netsentry verify`` proves the artifact's *bytes* are intact; it cannot prove the
runtime *behaves*. Environment skew — a numpy/scikit-learn/LightGBM version bump, a
different BLAS, a subtly incompatible unpickle — can move scores without touching a
byte of the artifact, and it is invisible until detection quietly degrades in
production. The canary closes that gap: at build time the bundle embeds a handful
of raw validation flows together with the exact calibrated scores it produced for
them; at load time (and on demand via ``netsentry canary``) the serving path
re-scores those flows through the full pipeline and compares.

This is deliberately the smallest end-to-end contract that could work: raw flows in,
calibrated attack probabilities out, through the identical code path predictions
take. A mismatch beyond tolerance means this runtime does not reproduce the model
that was validated — the behavioral analogue of a checksum failure, and grounds to
refuse traffic (``serving.canary_strict``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

CANARY_KEY = "canary"


@dataclass
class CanaryResult:
    """Outcome of replaying the bundle's canary flows in the current runtime."""

    present: bool
    ok: bool
    n: int
    max_delta: float
    tolerance: float
    message: str


def _json_safe_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, float | None]]:
    """Raw flow rows as plain dicts (NaN -> None) so the payload survives any dump."""
    rows: list[dict[str, float | None]] = []
    for _, row in frame[columns].iterrows():
        rows.append({col: (None if pd.isna(row[col]) else float(row[col])) for col in columns})
    return rows


def embed_canary(bundle: ModelBundle, frame: pd.DataFrame, settings: Settings) -> None:
    """Score ``frame`` through the bundle and embed rows + expected scores.

    Called by the bundle builder with a small, class-mixed sample of validation
    flows. The expected scores are whatever the *build-time* runtime produces —
    the canary asserts reproduction, not correctness.
    """
    stored = bundle.metadata.get("input_columns")
    columns = [c for c in (stored if isinstance(stored, list) else frame.columns) if c in frame]
    sample = frame.head(max(settings.serving.canary_rows, 1))
    expected = bundle.attack_scores(sample)
    bundle.metadata[CANARY_KEY] = {
        "columns": columns,
        "rows": _json_safe_rows(sample, columns),
        "expected_scores": [float(s) for s in expected],
        "tolerance": settings.serving.canary_tolerance,
        "created_at": datetime.now(UTC).isoformat(),
    }
    logger.info("Embedded canary", extra={"rows": len(sample)})


def run_canary(bundle: ModelBundle) -> CanaryResult:
    """Re-score the embedded canary flows and compare against the build-time scores."""
    payload = bundle.metadata.get(CANARY_KEY)
    if not isinstance(payload, dict) or not payload.get("rows"):
        return CanaryResult(
            present=False,
            ok=True,
            n=0,
            max_delta=0.0,
            tolerance=0.0,
            message="bundle carries no canary (built before canaries existed)",
        )
    columns = [str(c) for c in payload.get("columns", [])]
    raw_rows = payload["rows"]
    frame = pd.DataFrame(
        [
            {col: (np.nan if row.get(col) is None else row[col]) for col in columns}
            for row in raw_rows
        ],
        columns=columns,
    )
    expected = np.asarray(payload.get("expected_scores", []), dtype=float)
    tolerance = float(payload.get("tolerance", 1e-6))
    try:
        scores = np.asarray(bundle.attack_scores(frame), dtype=float)
    except Exception as exc:
        return CanaryResult(
            present=True,
            ok=False,
            n=len(raw_rows),
            max_delta=float("inf"),
            tolerance=tolerance,
            message=f"canary scoring raised: {exc}",
        )
    if len(scores) != len(expected):
        return CanaryResult(
            present=True,
            ok=False,
            n=len(raw_rows),
            max_delta=float("inf"),
            tolerance=tolerance,
            message=f"scored {len(scores)} canaries but expected {len(expected)}",
        )
    max_delta = float(np.max(np.abs(scores - expected))) if len(expected) else 0.0
    ok = bool(np.isfinite(max_delta) and max_delta <= tolerance)
    message = (
        f"{len(expected)} canaries reproduced within {tolerance:g} (max delta {max_delta:.2e})"
        if ok
        else (
            f"canary mismatch: max |score delta| {max_delta:.2e} exceeds {tolerance:g} - "
            "this runtime does not reproduce the model that was validated"
        )
    )
    return CanaryResult(
        present=True,
        ok=ok,
        n=len(expected),
        max_delta=max_delta,
        tolerance=tolerance,
        message=message,
    )
