"""Learning-curve subsampling (fast) and an end-to-end curve (slow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation.learning_curve import subsample


def _labelled(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    return pd.DataFrame({"feat": rng.normal(size=n), BINARY_TARGET: y})


def test_subsample_reduces_size_and_keeps_both_classes() -> None:
    df = _labelled()
    sub = subsample(df, 0.25, seed=1)
    assert 0.2 * len(df) <= len(sub) <= 0.3 * len(df)
    assert set(sub[BINARY_TARGET].unique()) == {0, 1}  # stratified: both classes survive


def test_subsample_full_fraction_is_identity() -> None:
    df = _labelled()
    assert len(subsample(df, 1.0, seed=1)) == len(df)


def test_subsample_is_seed_reproducible() -> None:
    df = _labelled()
    a = subsample(df, 0.5, seed=7)
    b = subsample(df, 0.5, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.slow
def test_learning_curve_end_to_end(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.config import load_settings
    from netsentry.data.split import make_splits
    from netsentry.evaluation.learning_curve import compute_learning_curve

    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.supervised.n_estimators = 40

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    points = compute_learning_curve(settings, "stratified", [0.25, 1.0])
    assert len(points) == 2
    ns = [n for n, _ in points]
    assert ns[0] < ns[1]  # more data at the second point
    assert all(0.0 <= pr <= 1.0 for _, pr in points)
