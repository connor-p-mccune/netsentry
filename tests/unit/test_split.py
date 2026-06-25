"""Split-integrity tests: disjoint parts, temporal ordering, reproducibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from netsentry.config import Settings, load_settings
from netsentry.data import schema
from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import (
    content_hash,
    leave_one_attack_out,
    make_splits,
    stratified_split,
    temporal_split,
)


def test_temporal_respects_day_boundaries(clean_synth: pd.DataFrame, settings: Settings) -> None:
    result = temporal_split(clean_synth, settings)
    train_days = set(settings.split.train_days)
    test_days = set(settings.split.test_days)
    assert set(result.train[schema.DAY_COLUMN]).issubset(train_days)
    assert set(result.val[schema.DAY_COLUMN]).issubset(train_days)  # val carved from train
    assert set(result.test[schema.DAY_COLUMN]).issubset(test_days)


def test_temporal_parts_are_disjoint(clean_synth: pd.DataFrame, settings: Settings) -> None:
    result = temporal_split(clean_synth, settings)
    train, val, test = set(result.train.index), set(result.val.index), set(result.test.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_stratified_parts_are_disjoint(clean_synth: pd.DataFrame, settings: Settings) -> None:
    result = stratified_split(clean_synth, settings)
    train, val, test = set(result.train.index), set(result.val.index), set(result.test.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # All rows are accounted for exactly once.
    assert len(train) + len(val) + len(test) == len(clean_synth)


def test_split_is_reproducible_from_seed(clean_synth: pd.DataFrame, settings: Settings) -> None:
    a = temporal_split(clean_synth, settings)
    b = temporal_split(clean_synth, settings)
    for part in ("train", "val", "test"):
        assert content_hash(getattr(a, part)) == content_hash(getattr(b, part))


def test_leave_one_attack_out_holds_out_class(
    clean_synth: pd.DataFrame, settings: Settings
) -> None:
    result = leave_one_attack_out(clean_synth, "DoS Hulk", settings)
    # The detector trains on benign only...
    assert set(result.train[MULTICLASS_TARGET]) == {schema.BENIGN_LABEL}
    assert set(result.val[MULTICLASS_TARGET]) == {schema.BENIGN_LABEL}
    # ...and the held-out attack appears only in test.
    assert "DoS Hulk" in set(result.test[MULTICLASS_TARGET])


def test_make_splits_persists_both_strategies(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)

    manifest = make_splits(settings)

    splits = settings.paths.data_processed / "splits"
    for strategy in ("temporal", "stratified"):
        for part in ("train", "val", "test"):
            assert (splits / strategy / f"{part}.parquet").exists()
    assert (splits / "manifest.json").exists()
    strategies = manifest["strategies"]
    assert isinstance(strategies, dict)
    assert "temporal" in strategies and "stratified" in strategies
