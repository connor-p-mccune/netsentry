"""Honest train/val/test splitting — the methodological core.

Three strategies, in order of authority:

1. **temporal/by-day** (HEADLINE): train on earlier days, test on later days. On
   CIC-IDS2017 this is harder and far more honest than a shuffled split, because
   near-duplicate flows from one attack burst no longer straddle train/test.
2. **stratified** random (REFERENCE): optimistic; reported only to expose the gap.
3. **leave-one-attack-out** (for the anomaly detector): train on benign only,
   test detection of an attack class held out entirely.

Validation is always carved from the **training** split only (for threshold
selection and early stopping); the test set is touched once. Splits are persisted
with a content hash so the same rows never drift between runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from netsentry.data.clean import BINARY_TARGET, CLEAN_FILENAME, MULTICLASS_TARGET
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

SPLITS_DIRNAME = "splits"
MANIFEST_NAME = "manifest.json"


@dataclass
class SplitResult:
    """A train/val/test partition produced by one strategy (indices preserved)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def content_hash(df: pd.DataFrame) -> str:
    """Stable short hash of a frame's contents (incl. index) for reproducibility."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    """Return labels usable for stratification, or None if a class is too rare."""
    counts = labels.value_counts()
    if len(counts) >= 2 and int(counts.min()) >= 2:
        return labels
    return None


def _choose_stratify(df: pd.DataFrame) -> pd.Series | None:
    """Prefer multiclass stratification; fall back to binary, else None."""
    multiclass = _safe_stratify(df[MULTICLASS_TARGET])
    if multiclass is not None:
        return multiclass
    return _safe_stratify(df[BINARY_TARGET])


def _carve_validation(
    train_full: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set from the TRAIN split only, stratified by attack/benign."""
    train, val = train_test_split(
        train_full,
        test_size=settings.split.val_size,
        random_state=settings.seed,
        stratify=_safe_stratify(train_full[BINARY_TARGET]),
    )
    return train, val


def temporal_split(df: pd.DataFrame, settings: Settings) -> SplitResult:
    """HEADLINE split: train on configured earlier days, test on later days."""
    day = settings.split.day_column
    if day not in df.columns:
        raise ValueError(f"Temporal split needs a '{day}' column; none found.")
    train_full = df[df[day].isin(settings.split.train_days)]
    test = df[df[day].isin(settings.split.test_days)]
    if train_full.empty or test.empty:
        raise ValueError("Temporal split produced an empty side; check day configuration.")
    train, val = _carve_validation(train_full, settings)
    return SplitResult(train, val, test, "temporal")


def stratified_split(df: pd.DataFrame, settings: Settings) -> SplitResult:
    """REFERENCE split: optimistic stratified random partition."""
    train_full, test = train_test_split(
        df,
        test_size=settings.split.stratified_test_size,
        random_state=settings.seed,
        stratify=_choose_stratify(df),
    )
    train, val = _carve_validation(train_full, settings)
    return SplitResult(train, val, test, "stratified")


def leave_one_attack_out(df: pd.DataFrame, holdout_attack: str, settings: Settings) -> SplitResult:
    """Anomaly split: benign-only train/val; test = held-out benign + one attack.

    The detector never sees the held-out attack (nor any attack) during training;
    the test set measures whether it flags that novel class.
    """
    benign = df[df[MULTICLASS_TARGET] == settings.labels.benign_label]
    attack = df[df[MULTICLASS_TARGET] == holdout_attack]
    if attack.empty:
        raise ValueError(f"No rows for held-out attack '{holdout_attack}'.")
    benign_train, benign_test = train_test_split(
        benign, test_size=settings.split.stratified_test_size, random_state=settings.seed
    )
    train, val = train_test_split(
        benign_train, test_size=settings.split.val_size, random_state=settings.seed
    )
    test = pd.concat([benign_test, attack], ignore_index=False)
    return SplitResult(train, val, test, f"loao_{holdout_attack}")


def build_splits(df: pd.DataFrame, settings: Settings, strategy: str) -> SplitResult:
    """Dispatch to a split strategy by name."""
    if strategy == "temporal":
        return temporal_split(df, settings)
    if strategy == "stratified":
        return stratified_split(df, settings)
    raise ValueError(f"Unknown split strategy: {strategy!r}")


def make_splits(settings: Settings) -> dict[str, object]:
    """Build and persist train/val/test for both split strategies; write a manifest."""
    clean_path = settings.paths.data_processed / CLEAN_FILENAME
    if not clean_path.exists():
        raise FileNotFoundError(f"{clean_path} not found. Run the clean stage first.")
    df = pd.read_parquet(clean_path)

    splits_root = settings.paths.data_processed / SPLITS_DIRNAME
    manifest: dict[str, object] = {
        "seed": settings.seed,
        "val_size": settings.split.val_size,
        "rows": len(df),
        "strategies": {},
    }
    strategies = manifest["strategies"]
    assert isinstance(strategies, dict)

    for strategy in ("temporal", "stratified"):
        result = build_splits(df, settings, strategy)
        _assert_disjoint(result)
        entry: dict[str, object] = {}
        for part, frame in result.as_dict().items():
            entry[part] = {"rows": len(frame), "hash": content_hash(frame)}
            if settings.split.persist:
                out = splits_root / strategy / f"{part}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(out, index=True)
        strategies[strategy] = entry
        logger.info(
            "Built split",
            extra={
                "strategy": strategy,
                "train": len(result.train),
                "val": len(result.val),
                "test": len(result.test),
            },
        )

    if settings.split.persist:
        splits_root.mkdir(parents=True, exist_ok=True)
        (splits_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _assert_disjoint(result: SplitResult) -> None:
    """Guard: the three parts must not share any row index."""
    train_idx = set(result.train.index)
    val_idx = set(result.val.index)
    test_idx = set(result.test.index)
    if train_idx & val_idx or train_idx & test_idx or val_idx & test_idx:
        raise AssertionError(f"Split '{result.strategy}' has overlapping rows across parts.")


def load_split(settings: Settings, strategy: str, part: str) -> pd.DataFrame:
    """Load a persisted split partition."""
    path = settings.paths.data_processed / SPLITS_DIRNAME / strategy / f"{part}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `netsentry prep` first.")
    return pd.read_parquet(path)
