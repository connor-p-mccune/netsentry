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

# Behavioural families the CICFlowMeter statistics fall into, defined by keyword so
# the partition is legible and covers new columns automatically. Ordered because the
# first matching family wins (e.g. "Flow Packets/s" is a rate, not a volume count).
# Used by the feature-group ablation study to measure each family's marginal value.
_FEATURE_GROUP_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timing/IAT", ("IAT", "Active", "Idle", "Duration")),
    ("flow rates", ("/s",)),
    ("packet size", ("Packet Length", "Segment Size", "Packet Size")),
    ("TCP flags", ("Flag", "PSH", "URG", "ECE", "CWE")),
    ("volume/counts", ("Total", "Subflow", "act_data_pkt", "Down/Up")),
    ("header/window/bulk", ("Header", "Init_Win", "min_seg", "Bulk")),
)


def feature_group(feature: str) -> str:
    """Behavioural family a feature belongs to (first keyword match, else 'other')."""
    for group, keywords in _FEATURE_GROUP_KEYWORDS:
        if any(keyword in feature for keyword in keywords):
            return group
    return "other"


def feature_groups(*, include_destination_port: bool = False) -> dict[str, list[str]]:
    """Partition the feature columns into behavioural families (non-empty groups)."""
    groups: dict[str, list[str]] = {}
    for feature in schema.feature_columns(include_destination_port=include_destination_port):
        groups.setdefault(feature_group(feature), []).append(feature)
    return groups


def get_feature_set(name: str, *, include_destination_port: bool = False) -> list[str]:
    """Return the ordered feature columns for a named feature set."""
    if name == "full":
        return schema.feature_columns(include_destination_port=include_destination_port)
    if name in FEATURE_SETS:
        return list(FEATURE_SETS[name])
    raise KeyError(f"Unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)} or 'full'.")


def display_feature_name(name: str) -> str:
    """Strip a ColumnTransformer branch prefix for human-facing output.

    The fitted pipeline names its outputs ``numeric__Flow Duration``; every
    surface an analyst reads (API ``top_features``, distilled rules, evasion
    tables) should say ``Flow Duration``. One helper so they all agree.
    """
    return name.split("__", 1)[1] if "__" in name else name


def numeric_features() -> list[str]:
    """The numeric feature columns (never includes the borderline port)."""
    return schema.feature_columns(include_destination_port=False)


def categorical_features(settings: Settings) -> list[str]:
    """Categorical features: ``Destination Port`` only when explicitly enabled."""
    return [schema.DESTINATION_PORT] if settings.features.encode_destination_port else []


def model_features(settings: Settings) -> list[str]:
    """All columns the model consumes, given the configured port handling."""
    return numeric_features() + categorical_features(settings)
