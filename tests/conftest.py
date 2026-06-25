"""Shared pytest fixtures, including a tiny DataFrame that reproduces the
dataset's defects (whitespace headers, an Inf, a duplicate row, an identifier
column, a dash-variant label, a -1 sentinel)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path(repo_root: Path) -> Path:
    """Path to the committed default configuration."""
    return repo_root / "configs" / "default.yaml"


@pytest.fixture
def settings(default_config_path: Path) -> Settings:
    """A fresh Settings loaded from the default config (mutable per test)."""
    return load_settings(default_config_path)


@pytest.fixture
def quirky_raw() -> pd.DataFrame:
    """A minimal raw frame carrying every CIC-IDS2017 quirk we must handle.

    Rows 0 and 1 are exact duplicates; row 2 has an Inf rate; row 3 uses the
    cp1252 en-dash in a Web Attack label; row 4 has a -1 'not set' sentinel.
    """
    return pd.DataFrame(
        {
            " Flow ID": ["a", "a", "b", "c", "d"],
            "Source IP": ["1.1.1.1", "1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"],
            " Destination Port": [80, 80, 80, 80, 53],
            "Flow Duration": [100, 100, 5, 50, 2],
            " Flow Bytes/s": [1000.0, 1000.0, np.inf, 500.0, 200.0],
            "Init_Win_bytes_forward": [8192, 8192, 256, 256, -1],
            " Label": ["BENIGN", "BENIGN", "DoS Hulk", "Web Attack \x96 XSS", "PortScan"],
        }
    )
