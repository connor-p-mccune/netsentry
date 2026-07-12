"""Partial dependence + ICE: the grid, the sweep math, direction/effect, and e2e."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.explain.partial_dependence import (
    PartialDependence,
    _grid_for,
    partial_dependence_1d,
)


def test_grid_trims_tails_and_is_monotone() -> None:
    values = np.concatenate([np.arange(0, 100, dtype=float), np.array([1e9])])  # one outlier
    grid = _grid_for(values, points=10, trim_quantile=0.05)
    assert grid.size == 10
    assert grid[0] < grid[-1]
    assert grid[-1] < 1e6  # the outlier does not stretch the grid
    assert np.all(np.diff(grid) > 0)


def test_grid_collapses_for_constant_feature() -> None:
    grid = _grid_for(np.full(50, 7.0), points=10, trim_quantile=0.05)
    assert grid.size == 1


def test_partial_dependence_recovers_a_linear_response() -> None:
    # A score function that depends only on feature "x": monotone increasing in it.
    base = pd.DataFrame({"x": np.random.default_rng(0).normal(size=200), "y": 1.0})

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-frame["x"].to_numpy()))  # sigmoid(x)

    grid = np.linspace(-3, 3, 15)
    average, ice = partial_dependence_1d(score_fn, base, "x", grid, n_ice=10)
    assert average.shape == (15,)
    assert ice.shape == (10, 15)
    # Every row shares the same x sweep, so the PDP is the exact response and monotone.
    assert np.all(np.diff(average) > 0)
    np.testing.assert_allclose(average, 1.0 / (1.0 + np.exp(-grid)), atol=1e-9)


def test_ice_captures_heterogeneity_from_an_interaction() -> None:
    # score = sigmoid(x * sign) — the response direction flips with a second column.
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.normal(size=100), "sign": rng.choice([-1.0, 1.0], size=100)})

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(frame["x"] * frame["sign"]).to_numpy()))

    grid = np.linspace(-3, 3, 11)
    average, ice = partial_dependence_1d(score_fn, base, "x", grid, n_ice=40)
    # The averaged PDP is roughly flat (the two directions cancel) while individual
    # ICE curves swing hard — exactly the heterogeneity the report warns the mean hides.
    assert average.max() - average.min() < 0.15
    ice_swings = ice.max(axis=1) - ice.min(axis=1)
    assert ice_swings.max() > 0.6


def test_direction_and_effect_properties() -> None:
    grid = np.linspace(0, 1, 5)
    up = PartialDependence("f", grid, np.array([0.1, 0.2, 0.3, 0.4, 0.9]), np.empty((0, 5)), 1.0)
    assert up.direction == "increasing"
    assert up.effect == pytest.approx(0.8)

    hump = PartialDependence("f", grid, np.array([0.1, 0.5, 0.9, 0.5, 0.1]), np.empty((0, 5)), 1.0)
    assert hump.direction == "non-monotone"

    down = PartialDependence("f", grid, np.array([0.9, 0.7, 0.5, 0.3, 0.1]), np.empty((0, 5)), 1.0)
    assert down.direction == "decreasing"


@pytest.mark.slow
def test_partial_dependence_report_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.explain.partial_dependence import run_partial_dependence_report

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.reports_dir = tmp_path / "reports"
    settings.paths.figures_dir = tmp_path / "figures"
    settings.paths.mlruns_dir = tmp_path / "mlruns"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 40
    settings.partial_dependence.top_k = 3
    settings.partial_dependence.grid_points = 8
    settings.partial_dependence.sample_rows = 150
    settings.partial_dependence.ice_samples = 10

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    out = run_partial_dependence_report(settings)
    text = out.read_text(encoding="utf-8").lower()
    assert out.exists() and "partial dependence" in text and "ice" in text
    assert "marginal response" in text
    assert (settings.paths.figures_dir / "partial_dependence.png").exists()
