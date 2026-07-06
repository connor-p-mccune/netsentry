"""Behavioral canaries: embed at build, replay at load, fail on runtime skew."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.config import Settings
from netsentry.serving.canary import CANARY_KEY, embed_canary, run_canary


class _StubBundle:
    """The minimal bundle surface the canary machinery touches."""

    def __init__(self, scores: list[float], metadata: dict[str, object] | None = None) -> None:
        self.metadata: dict[str, object] = metadata if metadata is not None else {}
        self._scores = np.asarray(scores, dtype=float)

    def attack_scores(self, frame: pd.DataFrame) -> np.ndarray:
        return self._scores[: len(frame)]


def _frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {"Flow Duration": np.linspace(10, 40, n), "Flow Bytes/s": [1.0, np.nan, 3.0, 4.0][:n]}
    )


def test_embed_then_replay_reproduces_exactly(settings: Settings) -> None:
    bundle = _StubBundle([0.1, 0.9, 0.5, 0.2])
    embed_canary(bundle, _frame(), settings)  # type: ignore[arg-type]
    payload = bundle.metadata[CANARY_KEY]
    assert isinstance(payload, dict) and len(payload["rows"]) == 4
    assert payload["rows"][1]["Flow Bytes/s"] is None  # NaN survives the round-trip as None

    result = run_canary(bundle)  # type: ignore[arg-type]
    assert result.present and result.ok
    assert result.max_delta == 0.0


def test_runtime_skew_beyond_tolerance_fails(settings: Settings) -> None:
    bundle = _StubBundle([0.1, 0.9, 0.5, 0.2])
    embed_canary(bundle, _frame(), settings)  # type: ignore[arg-type]
    bundle._scores = bundle._scores + 5e-3  # a runtime that scores differently

    result = run_canary(bundle)  # type: ignore[arg-type]
    assert result.present and not result.ok
    assert np.isclose(result.max_delta, 5e-3)
    assert "does not reproduce" in result.message


def test_skew_within_tolerance_passes(settings: Settings) -> None:
    settings.serving.canary_tolerance = 1e-2
    bundle = _StubBundle([0.1, 0.9, 0.5, 0.2])
    embed_canary(bundle, _frame(), settings)  # type: ignore[arg-type]
    bundle._scores = bundle._scores + 5e-3  # inside the configured band

    assert run_canary(bundle).ok  # type: ignore[arg-type]


def test_bundle_without_canary_is_reported_not_failed() -> None:
    result = run_canary(_StubBundle([0.5]))  # type: ignore[arg-type]
    assert not result.present and result.ok and result.n == 0


def test_row_count_mismatch_fails_loudly(settings: Settings) -> None:
    bundle = _StubBundle([0.1, 0.9, 0.5, 0.2])
    embed_canary(bundle, _frame(), settings)  # type: ignore[arg-type]
    bundle._scores = bundle._scores[:2]  # a runtime that silently drops rows

    result = run_canary(bundle)  # type: ignore[arg-type]
    assert not result.ok and "expected" in result.message
