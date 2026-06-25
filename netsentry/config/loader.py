"""Load :class:`~netsentry.config.settings.Settings` from YAML files.

Resolution order (lowest to highest precedence): model defaults -> base YAML ->
override YAML(s) -> environment variables. This lets ``configs/default.yaml``
hold the full knob surface and ``configs/cicids2017.yaml`` override only the
dataset-specific bits.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from netsentry.config.settings import Settings, _yaml_overrides

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict (empty dict for an empty file)."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level, got {type(loaded)}")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_settings(
    config_path: Path | None = None,
    *,
    overrides: list[Path] | None = None,
) -> Settings:
    """Build a :class:`Settings` from a base config plus optional overrides.

    Args:
        config_path: Base YAML. Defaults to ``configs/default.yaml`` if it exists,
            otherwise pure model defaults are used.
        overrides: Additional YAML files merged on top, in order.

    Returns:
        A fully validated settings object (env vars still take final precedence).
    """
    data: dict[str, Any] = {}

    base = config_path if config_path is not None else DEFAULT_CONFIG
    if base.exists():
        data = load_yaml(base)
    elif config_path is not None:
        # An explicitly requested file that does not exist is an error.
        raise FileNotFoundError(f"Config file not found: {config_path}")

    for override in overrides or []:
        data = deep_merge(data, load_yaml(override))

    # Inject the merged YAML as a low-priority source so env vars can override it.
    token = _yaml_overrides.set(data)
    try:
        return Settings()
    finally:
        _yaml_overrides.reset(token)
