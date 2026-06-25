"""Clean raw CIC-IDS2017 CSVs into a documented, modelling-ready dataset.

Implements the cleaning checklist, logging a before/after count for every
transformation:

1. strip whitespace from column headers;
2. normalise + consolidate labels (the en-dash ``Web Attack`` variants);
3. drop identifier/leaky columns;
4. coerce features to numeric and replace ±Inf with NaN (imputation is deferred
   to the feature pipeline, where it is fit on the training split only);
5. handle negative "not set" sentinels per config;
6. build binary and multiclass targets;
7. drop exact duplicate rows.

Note what cleaning deliberately does NOT do: it never imputes, scales, or
computes any cross-row statistic, because those must be fit on train only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

BINARY_TARGET = "label_binary"
MULTICLASS_TARGET = "label_multiclass"
CLEAN_FILENAME = "clean.parquet"

# Dash variants in the raw 'Web Attack' labels: en dash, em dash,
# non-breaking hyphen, and the cp1252 0x96 byte (decoded via latin-1).
_DASHES = tuple(chr(cp) for cp in (0x2013, 0x2014, 0x2011, 0x0096))


def normalize_label(value: object) -> str:
    """Normalise a raw label: unify dashes and collapse whitespace.

    Turns e.g. ``'Web Attack \\x96 Brute Force'`` into ``'Web Attack - Brute
    Force'`` while leaving hyphenated names like ``'FTP-Patator'`` untouched.
    """
    text = str(value)
    for dash in _DASHES:
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text).strip()


def clean_dataframe(
    df: pd.DataFrame, settings: Settings, *, source: str | None = None
) -> pd.DataFrame:
    """Apply the full cleaning checklist to a single raw flow DataFrame."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]

    out = _normalize_and_consolidate_labels(out, settings)
    out = _drop_identifier_columns(out)
    out = _coerce_features(out)
    out = _handle_negative_sentinels(out, settings)
    out = _build_targets(out, settings)

    before = len(out)
    if settings.data.drop_duplicates:
        out = out.drop_duplicates(ignore_index=True)
    removed = before - len(out)

    logger.info(
        "Cleaned frame",
        extra={
            "source": source or "<frame>",
            "rows_in": len(df),
            "rows_out": len(out),
            "duplicates_dropped": removed,
        },
    )
    return out


def _normalize_and_consolidate_labels(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if schema.LABEL_COLUMN not in df.columns:
        return df
    normalized = df[schema.LABEL_COLUMN].map(normalize_label)
    consolidation = settings.labels.consolidation
    df[schema.LABEL_COLUMN] = normalized.map(lambda lbl: consolidation.get(lbl, lbl))
    return df


def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    leaks = set(schema.identifier_columns())
    present = [col for col in df.columns if col in leaks]
    if present:
        logger.info("Dropping identifier/leaky columns", extra={"columns": present})
    return df.drop(columns=present)


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [col for col in schema.FEATURE_COLUMNS if col in df.columns]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(df[features].to_numpy(dtype="float64", na_value=np.nan)).sum())
    if inf_count:
        logger.info("Replacing Inf with NaN", extra={"inf_values": inf_count})
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df


def _handle_negative_sentinels(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if settings.data.negative_sentinel_strategy != "nan":
        return df  # keep -1 as an informative sentinel
    for col in settings.data.negative_sentinel_columns:
        if col in df.columns:
            mask = df[col] == -1
            count = int(mask.sum())
            if count:
                df.loc[mask, col] = np.nan
                logger.info("Sentinel -1 -> NaN", extra={"column": col, "count": count})
    return df


def _build_targets(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if schema.LABEL_COLUMN not in df.columns:
        return df
    benign = settings.labels.benign_label
    df[BINARY_TARGET] = (df[schema.LABEL_COLUMN] != benign).astype("int64")
    df[MULTICLASS_TARGET] = df[schema.LABEL_COLUMN].astype("string")
    return df


def clean_raw(settings: Settings) -> Path:
    """Clean every raw CSV into a single processed parquet; return its path."""
    raw_dir = settings.paths.data_raw
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No raw CSVs in {raw_dir}. Run `netsentry download` first.")

    frames: list[pd.DataFrame] = []
    rows_in = 0
    for csv in csvs:
        # latin-1 tolerates the dataset's cp1252 bytes (e.g. the 0x96 dash).
        raw = pd.read_csv(csv, encoding="latin-1", low_memory=False)
        rows_in += len(raw)
        raw.columns = [str(col).strip() for col in raw.columns]
        day = schema.day_from_filename(csv.name)
        if day and schema.DAY_COLUMN not in raw.columns:
            raw[schema.DAY_COLUMN] = day
        frames.append(clean_dataframe(raw, settings, source=csv.name))

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    if settings.data.drop_duplicates:
        combined = combined.drop_duplicates(ignore_index=True)
    cross_file_dupes = before - len(combined)

    out_path = settings.paths.data_processed / CLEAN_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    attack_rate = (
        float(combined[BINARY_TARGET].mean()) if BINARY_TARGET in combined else float("nan")
    )
    logger.info(
        "Wrote clean dataset",
        extra={
            "path": str(out_path),
            "rows_in": rows_in,
            "rows_out": len(combined),
            "cross_file_duplicates_dropped": cross_file_dupes,
            "classes": (
                int(combined[MULTICLASS_TARGET].nunique()) if MULTICLASS_TARGET in combined else 0
            ),
            "attack_rate": round(attack_rate, 4),
        },
    )
    return out_path
