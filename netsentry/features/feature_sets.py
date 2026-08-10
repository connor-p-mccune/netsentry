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


# --- When each statistic exists ----------------------------------------------------
# A second, orthogonal partition of the same columns: not *what* a feature measures but
# *when its value is knowable*. CICFlowMeter emits one record per finished flow, so the
# deployed model is structurally a post-mortem detector; the earliness study uses this
# partition to price that. Three nested tiers, in the order a detector could act on them:
#
#   handshake  — fixed by the connection setup and never revised afterwards.
#   in-flight  — *intensive* statistics (mean/min/max/std/rate/ratio). Their value over a
#                prefix of the flow is a noisy estimate of the same quantity, so a detector
#                may act on them mid-flow.
#   complete   — *extensive* statistics (totals, cumulative sums, counts) and teardown or
#                whole-timeline quantities (durations, flag counts, active/idle periods).
#                A prefix value of an accumulating feature is not a noisy version of the
#                final value, it is a systematically smaller different number, so acting on
#                it early is not an approximation — it is a category error.
#
# Keyword-driven, first match wins, so new columns are classified automatically. Note that
# `SYN Flag Count` lands in `complete` although a SYN arrives first: the *count* keeps
# accruing (retransmits) and the exporter only reports it at flow end. The conservative
# call, stated in the report's scope.
_HANDSHAKE_FEATURES: frozenset[str] = frozenset(
    {
        schema.DESTINATION_PORT,  # in the first packet; excluded from the headline anyway
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "min_seg_size_forward",
    }
)
_COMPLETE_KEYWORDS: tuple[str, ...] = (
    "Total",
    "Subflow",
    "Duration",
    "Idle",
    "Active",
    "Flag",
    "Flags",
    "Header Length",
    "Bulk",
    "act_data_pkt",
)

AVAILABILITY_TIERS: tuple[str, ...] = ("handshake", "in_flight", "complete")


def availability_tier(feature: str) -> str:
    """Earliest tier at which ``feature`` has the value the model was trained on."""
    if feature in _HANDSHAKE_FEATURES:
        return "handshake"
    if any(keyword in feature for keyword in _COMPLETE_KEYWORDS):
        return "complete"
    return "in_flight"


def availability_sets(*, include_destination_port: bool = False) -> dict[str, list[str]]:
    """The **nested** feature sets a detector may use at each decision time.

    Nested rather than disjoint because availability accumulates: an in-flight detector
    still knows the handshake fields, and the deployed complete-flow model knows
    everything. ``availability_sets()["complete"]`` is therefore the full feature list.
    """
    columns = schema.feature_columns(include_destination_port=include_destination_port)
    tiers = {c: availability_tier(c) for c in columns}
    out: dict[str, list[str]] = {}
    allowed: set[str] = set()
    for tier in AVAILABILITY_TIERS:
        allowed.add(tier)
        out[tier] = [c for c in columns if tiers[c] in allowed]
    return out


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
