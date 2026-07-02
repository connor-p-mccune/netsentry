"""Feature-group tests: the family partition covers the schema and is keyword-correct,
and the ablation column-grouping maps prefixed transform names back to families."""

from __future__ import annotations

from netsentry.data import schema
from netsentry.evaluation.ablation import _column_groups
from netsentry.features.feature_sets import feature_group, feature_groups


def test_every_feature_lands_in_exactly_one_group() -> None:
    groups = feature_groups()
    flat = [f for members in groups.values() for f in members]
    assert sorted(flat) == sorted(schema.feature_columns(include_destination_port=False))
    assert len(flat) == len(set(flat))  # partition: no feature in two groups


def test_keyword_assignment_is_behaviourally_sensible() -> None:
    assert feature_group("Flow Packets/s") == "flow rates"  # rate wins over "Packets"
    assert feature_group("Total Fwd Packets") == "volume/counts"
    assert feature_group("Flow IAT Mean") == "timing/IAT"
    assert feature_group("Fwd Packet Length Max") == "packet size"
    assert feature_group("SYN Flag Count") == "TCP flags"
    assert feature_group("Init_Win_bytes_forward") == "header/window/bulk"


def test_unknown_feature_is_other() -> None:
    assert feature_group("Some Novel Statistic") == "other"


def test_column_groups_strip_transform_prefixes() -> None:
    names = ["numeric__Flow IAT Mean", "numeric__Total Fwd Packets", "numeric__Flow Bytes/s"]
    groups = _column_groups(names)
    assert groups["timing/IAT"] == {0}
    assert groups["volume/counts"] == {1}
    assert groups["flow rates"] == {2}


def test_groups_are_non_empty() -> None:
    assert all(members for members in feature_groups().values())
