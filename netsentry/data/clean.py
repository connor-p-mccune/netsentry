"""Clean raw CIC-IDS2017 CSVs into a documented, modelling-ready dataset.

Implements the cleaning checklist (whitespace headers, Inf->NaN, duplicate
removal, negative sentinels, label consolidation, binary + multiclass targets),
logging a before/after count for every transformation. Implemented in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings


def clean_dataframe(df: "pd.DataFrame", settings: "Settings") -> "pd.DataFrame":
    """Apply the full cleaning checklist to a raw flow DataFrame."""
    raise NotImplementedError("Implemented in Phase 2")


def clean_raw(settings: "Settings") -> Path:
    """Clean all raw CSVs and write a single processed parquet; return its path."""
    raise NotImplementedError("Implemented in Phase 2")
