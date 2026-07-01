"""Auto-generated model card: renders the artifact's facts from bundle metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from netsentry.evaluation.model_card import render_model_card


@dataclass
class _FakeBundle:
    """Minimal stand-in exposing the metadata/thresholds render_model_card reads."""

    metadata: dict[str, Any]
    thresholds: dict[str, float]
    anomaly_detector: object | None = None
    _classes: np.ndarray = field(default_factory=lambda: np.array(["BENIGN", "DoS Hulk"]))

    @property
    def classes(self) -> np.ndarray:
        return self._classes


def _bundle(**over: Any) -> _FakeBundle:
    meta = {
        "version": "0.1.0",
        "task": "multiclass",
        "split_strategy": "stratified",
        "backend": "lightgbm",
        "n_features": 77,
        "n_train": 40000,
        "created_at": "2026-01-01T00:00:00+00:00",
        "calibration": {"enabled": True, "method": "isotonic"},
        "conformal": {"alpha": 0.1, "tau_benign": 0.5, "tau_attack": 0.5},
        "drift_reference": {"cols": {}},
    }
    meta.update(over.pop("metadata", {}))
    return _FakeBundle(metadata=meta, thresholds={"fpr_1pct": 0.77, "cost_optimal": 0.58}, **over)


def test_card_includes_artifact_facts() -> None:
    card = render_model_card(_bundle())  # type: ignore[arg-type]
    assert "Model Card (auto-generated)" in card
    assert "lightgbm" in card
    assert "isotonic" in card
    assert "fpr_1pct" in card and "cost_optimal" in card
    assert "DoS Hulk" in card


def test_card_reports_component_presence() -> None:
    with_components = render_model_card(_bundle(anomaly_detector=object()))  # type: ignore[arg-type]
    assert "| benign-only anomaly detector | yes |" in with_components
    assert "| conformal prediction set | yes |" in with_components

    bare = render_model_card(
        _bundle(metadata={"conformal": None, "drift_reference": None})  # type: ignore[arg-type]
    )
    assert "| benign-only anomaly detector | no |" in bare
    assert "| conformal prediction set | no |" in bare


def test_card_handles_uncalibrated() -> None:
    card = render_model_card(_bundle(metadata={"calibration": {"enabled": False}}))  # type: ignore[arg-type]
    assert "Probability calibration: **none**" in card
