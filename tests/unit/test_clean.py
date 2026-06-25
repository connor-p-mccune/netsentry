"""Cleaning correctness: the high-value tests that the pipeline is honest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from netsentry.config import Settings, load_settings
from netsentry.data import schema
from netsentry.data.clean import (
    BINARY_TARGET,
    MULTICLASS_TARGET,
    clean_dataframe,
    clean_raw,
    normalize_label,
)
from netsentry.data.download import download_dataset


def test_headers_are_stripped(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    assert all(col == col.strip() for col in cleaned.columns)
    assert "Flow Bytes/s" in cleaned.columns
    assert "Destination Port" in cleaned.columns


def test_identifier_columns_are_dropped(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    for leak in ("Flow ID", "Source IP"):
        assert leak not in cleaned.columns


def test_inf_is_replaced_with_nan(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    assert not np.isinf(cleaned["Flow Bytes/s"].to_numpy(dtype="float64")).any()
    assert cleaned["Flow Bytes/s"].isna().any()


def test_exact_duplicates_are_dropped(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    # Rows 0 and 1 are identical -> 5 in, 4 out.
    cleaned = clean_dataframe(quirky_raw, settings)
    assert len(cleaned) == 4


def test_labels_are_normalized_and_consolidated(
    quirky_raw: pd.DataFrame, settings: Settings
) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    labels = set(cleaned[MULTICLASS_TARGET].tolist())
    assert "Web Attack" in labels  # the en-dash XSS variant was consolidated
    assert "Web Attack \x96 XSS" not in labels


def test_binary_target_is_correct(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    benign_mask = cleaned[schema.LABEL_COLUMN] == schema.BENIGN_LABEL
    assert (cleaned.loc[benign_mask, BINARY_TARGET] == 0).all()
    assert (cleaned.loc[~benign_mask, BINARY_TARGET] == 1).all()


def test_negative_sentinel_kept_by_default(quirky_raw: pd.DataFrame, settings: Settings) -> None:
    cleaned = clean_dataframe(quirky_raw, settings)
    assert (cleaned["Init_Win_bytes_forward"] == -1).any()


def test_negative_sentinel_to_nan_when_configured(
    quirky_raw: pd.DataFrame, settings: Settings
) -> None:
    settings.data.negative_sentinel_strategy = "nan"
    cleaned = clean_dataframe(quirky_raw, settings)
    assert not (cleaned["Init_Win_bytes_forward"] == -1).any()
    assert cleaned["Init_Win_bytes_forward"].isna().any()


def test_normalize_label_keeps_hyphenated_names() -> None:
    assert normalize_label("FTP-Patator") == "FTP-Patator"
    assert normalize_label("Web Attack \x96 Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("  BENIGN ") == "BENIGN"


def test_clean_raw_writes_parquet_without_leaks(repo_root: Path, tmp_path: Path) -> None:
    settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_raw = tmp_path / "raw"
    settings.paths.data_processed = tmp_path / "processed"
    settings.data.synthetic_rows = 3000

    download_dataset(settings)
    out = clean_raw(settings)

    assert out.exists()
    frame = pd.read_parquet(out)
    assert BINARY_TARGET in frame.columns
    assert MULTICLASS_TARGET in frame.columns
    assert schema.DAY_COLUMN in frame.columns
    # No identifier/leaky column survives into the processed dataset.
    for leak in schema.identifier_columns():
        assert leak not in frame.columns
    # Inf was eliminated.
    numeric = frame.select_dtypes("number").to_numpy(dtype="float64", na_value=np.nan)
    assert not np.isinf(numeric).any()
