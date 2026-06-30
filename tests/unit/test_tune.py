"""Hyperparameter tuning: search space, param application, and the search loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from netsentry.config import Settings
from netsentry.training.tune import (
    TuneResult,
    _random_search,
    apply_params,
    suggest_random,
    write_tuned_config,
)


def test_suggest_random_respects_bounds() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = suggest_random(rng)
        assert 0.01 <= p["learning_rate"] <= 0.3
        assert 15 <= p["num_leaves"] <= 255
        assert 3 <= p["max_depth"] <= 12
        assert 5 <= p["min_child_samples"] <= 200
        assert 0.5 <= p["subsample"] <= 1.0
        assert 1e-3 <= p["reg_lambda"] <= 10.0


def test_apply_params_overrides_supervised_copy(settings: Settings) -> None:
    original_lr = settings.supervised.learning_rate
    variant = apply_params(settings, {"learning_rate": 0.123, "num_leaves": 31})
    assert variant.supervised.learning_rate == 0.123
    assert variant.supervised.num_leaves == 31
    # The original is untouched (pure function, deep copy).
    assert settings.supervised.learning_rate == original_lr


def test_random_search_finds_the_argmax() -> None:
    # Objective peaks at learning_rate near 0.3; the search should climb toward it.
    def objective(params: dict[str, float]) -> float:
        return -abs(params["learning_rate"] - 0.3)

    best_params, best_value = _random_search(objective, n_trials=40, seed=1)
    assert best_value <= 0.0
    assert abs(best_params["learning_rate"] - 0.3) < 0.1


def test_random_search_is_seed_reproducible() -> None:
    def objective(params: dict[str, float]) -> float:
        return params["subsample"]

    a = _random_search(objective, n_trials=10, seed=42)
    b = _random_search(objective, n_trials=10, seed=42)
    assert a == b


def test_write_tuned_config_roundtrips(tmp_path: Path) -> None:
    result = TuneResult(
        best_params={"learning_rate": 0.05, "num_leaves": 63},
        best_value=0.7,
        n_trials=5,
        method="random",
    )
    out = write_tuned_config(result, tmp_path / "tuned.yaml")
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded["supervised"]["learning_rate"] == 0.05
    assert loaded["supervised"]["tune"] is False  # avoid re-tuning when reused
    assert loaded["_tuning"]["method"] == "random"


@pytest.mark.slow
def test_tune_supervised_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.training.tune import tune_supervised

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.split.strategy = "stratified"
    settings.supervised.task = "binary"
    settings.supervised.n_estimators = 40
    settings.supervised.tune_trials = 3

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    result = tune_supervised(settings)
    assert result.n_trials == 3
    assert 0.0 <= result.best_value <= 1.0
    assert "learning_rate" in result.best_params
