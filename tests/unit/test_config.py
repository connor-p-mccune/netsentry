"""Configuration loading, override merging, and env-var precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from netsentry.config import Settings, load_settings


def test_default_config_loads(default_config_path: Path) -> None:
    settings = load_settings(default_config_path)
    assert isinstance(settings, Settings)
    assert settings.seed == 42
    assert settings.split.strategy == "temporal"
    assert settings.thresholds.fpr_targets == [0.001, 0.01]
    # The headline model must not use Destination Port by default (leakage care).
    assert settings.features.encode_destination_port is False


def test_overrides_are_merged(repo_root: Path) -> None:
    settings = load_settings(
        repo_root / "configs" / "default.yaml",
        overrides=[repo_root / "configs" / "cicids2017.yaml"],
    )
    # The CIC override turns off synthetic generation for the headline run...
    assert settings.data.allow_synthetic is False
    # ...while leaving unrelated defaults intact.
    assert settings.seed == 42


def test_env_var_overrides_yaml(default_config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NETSENTRY_SEED", "7")
    monkeypatch.setenv("NETSENTRY_SUPERVISED__LEARNING_RATE", "0.123")
    settings = load_settings(default_config_path)
    assert settings.seed == 7
    assert settings.supervised.learning_rate == 0.123


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")
