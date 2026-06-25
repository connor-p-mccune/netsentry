"""Named feature groups and the with/without ``Destination Port`` variants.

The headline model uses ``full_no_port``; ``full_with_port`` exists so the
port-leakage gap can be measured (see DATA_CARD.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from netsentry.data import schema

if TYPE_CHECKING:
    from netsentry.config import Settings

FEATURE_SETS: dict[str, list[str]] = {
    "full_no_port": schema.feature_columns(include_destination_port=False),
    "full_with_port": schema.feature_columns(include_destination_port=True),
}


def get_feature_set(name: str, *, include_destination_port: bool = False) -> list[str]:
    """Return the ordered feature columns for a named feature set."""
    if name == "full":
        return schema.feature_columns(include_destination_port=include_destination_port)
    if name in FEATURE_SETS:
        return list(FEATURE_SETS[name])
    raise KeyError(f"Unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)} or 'full'.")


def numeric_features() -> list[str]:
    """The numeric feature columns (never includes the borderline port)."""
    return schema.feature_columns(include_destination_port=False)


def categorical_features(settings: Settings) -> list[str]:
    """Categorical features: ``Destination Port`` only when explicitly enabled."""
    return [schema.DESTINATION_PORT] if settings.features.encode_destination_port else []


def model_features(settings: Settings) -> list[str]:
    """All columns the model consumes, given the configured port handling."""
    return numeric_features() + categorical_features(settings)
