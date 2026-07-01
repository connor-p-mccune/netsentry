"""Counterfactual recourse: greedy minimal-change search (fast) + end-to-end (slow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.explain.counterfactual import Change, recourse_for_row


class _StubModel:
    """A linear-ish scorer: attack prob rises with feature 0. Class 1 == attack."""

    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p_attack = np.clip(0.5 + 0.5 * np.asarray(x)[:, 0], 0.0, 1.0)
        return np.column_stack([1 - p_attack, p_attack])


class _StubBundle:
    def __init__(self) -> None:
        self.model = _StubModel()
        self.metadata = {"benign_label": "BENIGN"}
        self.calibrator = None

    @property
    def classes(self) -> np.ndarray:
        return self.model.classes_


def test_change_direction() -> None:
    assert Change("f", -1.0).direction == "decrease"
    assert Change("f", 2.0).direction == "increase"


def test_recourse_flips_by_moving_the_driving_feature() -> None:
    bundle = _StubBundle()
    x = np.array([[1.0, 0.0]])  # feature 0 high -> attack prob 1.0
    centroid = np.array([0.0, 0.0])  # benign has feature 0 == 0 -> prob 0.5
    rec = recourse_for_row(
        bundle, x, centroid, np.array([0]), threshold=0.6, feature_names=["f0", "f1"], max_steps=3
    )
    assert rec.flipped
    assert rec.final_score < 0.6 <= rec.original_score
    assert [c.feature for c in rec.changes] == ["f0"]
    assert rec.changes[0].direction == "decrease"


def test_recourse_gives_up_when_no_controllable_feature_helps() -> None:
    bundle = _StubBundle()
    x = np.array([[1.0, 0.0]])
    centroid = np.array([0.0, 0.0])
    # Only feature 1 is controllable, but it does not affect the score -> no recourse.
    rec = recourse_for_row(
        bundle, x, centroid, np.array([1]), threshold=0.6, feature_names=["f0", "f1"], max_steps=3
    )
    assert not rec.flipped
    assert rec.changes == []


def test_recourse_respects_step_budget() -> None:
    bundle = _StubBundle()
    x = np.array([[1.0, 0.0]])
    centroid = np.array([0.0, 0.0])
    rec = recourse_for_row(
        bundle, x, centroid, np.array([0]), threshold=-1.0, feature_names=["f0", "f1"], max_steps=1
    )
    # Threshold impossible to meet -> stops at the budget of 1 change.
    assert len(rec.changes) <= 1


@pytest.mark.slow
def test_recourse_report_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.explain.counterfactual import run_recourse_report
    from netsentry.serving.bundle import build_serving_bundle

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)
    build_serving_bundle(settings)

    out = run_recourse_report(settings, n_examples=3)
    assert out.exists()
    assert "Counterfactual Recourse" in out.read_text(encoding="utf-8")
