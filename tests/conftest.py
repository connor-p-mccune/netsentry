"""Shared pytest fixtures.

Phase 0 only needs the repo-root locator; the quirky synthetic dataset fixture
that reproduces CIC-IDS2017's defects is added in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config_path(repo_root: Path) -> Path:
    """Path to the committed default configuration."""
    return repo_root / "configs" / "default.yaml"
