"""Synthetic generation and the idempotent, verifying downloader (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from netsentry.config import Settings, load_settings
from netsentry.data import schema
from netsentry.data.download import download_dataset
from netsentry.data.synthetic import generate_synthetic


def _tmp_settings(repo_root: Path, tmp_path: Path) -> Settings:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_raw = tmp_path / "raw"
    settings.data.synthetic_rows = 2000
    settings.data.allow_synthetic = True
    settings.data.source_url = None
    return settings


def test_synthetic_has_full_schema_and_quirks(repo_root: Path, tmp_path: Path) -> None:
    settings = _tmp_settings(repo_root, tmp_path)
    frame = generate_synthetic(settings, rows=3000, seed=0)

    for column in schema.FEATURE_COLUMNS:
        assert column in frame.columns
    for leak in ("Flow ID", "Source IP", "Destination IP", "Timestamp"):
        assert leak in frame.columns
    assert schema.LABEL_COLUMN in frame.columns
    assert schema.DAY_COLUMN in frame.columns

    # The dataset's defects are reproduced for the cleaning step to handle.
    assert bool(np.isinf(frame["Flow Bytes/s"]).any())
    assert bool((frame["Init_Win_bytes_forward"] == -1).any())
    assert bool(frame.isna().any().any())

    # Benign dominates (imbalance), and attacks land on their real day.
    assert (frame[schema.LABEL_COLUMN] == schema.BENIGN_LABEL).mean() > 0.5
    portscan = frame[frame[schema.LABEL_COLUMN] == "PortScan"]
    if len(portscan):
        assert (portscan[schema.DAY_COLUMN] == "Friday").all()


def test_synthetic_is_deterministic(repo_root: Path, tmp_path: Path) -> None:
    settings = _tmp_settings(repo_root, tmp_path)
    a = generate_synthetic(settings, rows=1000, seed=7)
    b = generate_synthetic(settings, rows=1000, seed=7)
    assert a[schema.LABEL_COLUMN].tolist() == b[schema.LABEL_COLUMN].tolist()


def test_download_writes_and_is_idempotent(repo_root: Path, tmp_path: Path) -> None:
    settings = _tmp_settings(repo_root, tmp_path)
    paths = download_dataset(settings)
    assert paths
    assert all(p.suffix == ".csv" for p in paths)

    # A second call must skip and return the same verified files.
    again = download_dataset(settings)
    assert sorted(again) == sorted(paths)


def test_download_raises_with_clear_message_when_unavailable(
    repo_root: Path, tmp_path: Path
) -> None:
    settings = _tmp_settings(repo_root, tmp_path)
    settings.data.allow_synthetic = False
    settings.data.source_url = None
    with pytest.raises(FileNotFoundError, match="CIC-IDS2017"):
        download_dataset(settings)
