"""Acquire the CIC-IDS2017 CSVs into ``data/raw`` with integrity checks.

Idempotent: if verified CSVs are already present it skips re-downloading. When no
``source_url`` is configured it either generates a clearly-labelled synthetic
stand-in (if ``data.allow_synthetic``) or prints precise manual-download
instructions — the real dataset requires registration with the CIC.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.data import schema
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

_MANUAL_INSTRUCTIONS = """\
CIC-IDS2017 was not found and could not be downloaded automatically.

The dataset is distributed by the Canadian Institute for Cybersecurity and
requires accepting their terms. To provide it, do ONE of:

  1. Download the CSVs and place them in '{raw_dir}', or
  2. Set data.source_url in your config to a .zip mirror you are licensed to use
     (optionally set data.archive_sha256 to verify it), or
  3. Set data.allow_synthetic: true to generate a schema-faithful synthetic
     dataset for development/CI (clearly labelled as synthetic; not a real result).

Source: https://www.unb.ca/cic/datasets/ids-2017.html
"""


def download_dataset(settings: Settings, *, force: bool = False) -> list[Path]:
    """Fetch/locate the raw CSVs and verify them; return the local paths."""
    raw_dir = settings.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob("*.csv"))
    if existing and not force:
        logger.info("Raw CSVs already present; skipping download", extra={"files": len(existing)})
        _verify_csvs(existing, settings)
        return existing

    if settings.data.source_url:
        paths = _download_and_extract(settings)
    elif settings.data.allow_synthetic:
        logger.warning(
            "No source_url configured - generating SYNTHETIC data (not real CIC-IDS2017)"
        )
        from netsentry.data.synthetic import write_synthetic_raw

        paths = write_synthetic_raw(settings)
    else:
        raise FileNotFoundError(_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir))

    _verify_csvs(paths, settings)
    return paths


def _download_and_extract(settings: Settings) -> list[Path]:
    """Download the configured archive, verify its checksum, and extract CSVs."""
    raw_dir = settings.paths.data_raw
    url = settings.data.source_url
    assert url is not None  # guarded by the caller
    archive = raw_dir / settings.data.archive_name

    logger.info("Downloading dataset archive", extra={"url": url, "dest": str(archive)})
    with urllib.request.urlopen(url) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)

    size = archive.stat().st_size
    if size == 0:
        raise OSError(f"Downloaded archive is empty: {archive}")

    expected = settings.data.archive_sha256
    if expected:
        digest = _sha256(archive)
        if digest.lower() != expected.lower():
            raise ValueError(f"Checksum mismatch for {archive}: expected {expected}, got {digest}")
        logger.info("Archive checksum verified", extra={"sha256": digest})

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for member in members:
            target = raw_dir / Path(member).name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    logger.info("Extracted CSVs", extra={"count": len(members)})
    return sorted(raw_dir.glob("*.csv"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_csvs(paths: list[Path], settings: Settings) -> None:
    """Check each CSV is non-empty and carries a recognisable header with a label."""
    if not paths:
        raise FileNotFoundError("No CSV files found after acquisition.")

    for path in paths:
        if path.stat().st_size == 0:
            raise OSError(f"Raw CSV is empty: {path}")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        columns = {col.strip() for col in header.split(",")}
        if schema.LABEL_COLUMN not in columns:
            raise ValueError(f"{path.name} has no '{schema.LABEL_COLUMN}' column in its header.")

    expected = settings.data.expected_csv_count
    if not settings.data.allow_synthetic and len(paths) != expected:
        logger.warning(
            "Unexpected number of CSVs", extra={"found": len(paths), "expected": expected}
        )
    logger.info("Verified raw CSVs", extra={"files": len(paths)})
