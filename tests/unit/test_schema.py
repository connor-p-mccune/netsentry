"""The schema is the single source of truth for columns, leaks, and labels."""

from __future__ import annotations

from netsentry.data import schema


def test_feature_columns_exclude_destination_port_by_default() -> None:
    feats = schema.feature_columns()
    assert schema.DESTINATION_PORT not in feats
    assert schema.feature_columns(include_destination_port=True)[0] == schema.DESTINATION_PORT


def test_identifier_columns_cover_the_known_leaks() -> None:
    ids = set(schema.identifier_columns())
    for leak in ("Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"):
        assert leak in ids


def test_identifiers_are_disjoint_from_features() -> None:
    feats = set(schema.feature_columns(include_destination_port=True))
    assert feats.isdisjoint(set(schema.identifier_columns()))


def test_label_vocabulary() -> None:
    labels = schema.label_values()
    assert schema.BENIGN_LABEL in labels
    assert len(schema.attack_labels()) == 14
    assert schema.BENIGN_LABEL not in schema.attack_labels()


def test_label_days_cover_all_attacks() -> None:
    for attack in schema.attack_labels():
        assert attack in schema.LABEL_DAYS
        assert schema.LABEL_DAYS[attack] in schema.DAY_ORDER


def test_day_from_filename() -> None:
    assert schema.day_from_filename("Monday-WorkingHours.pcap_ISCX.csv") == "Monday"
    assert schema.day_from_filename("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") == "Friday"
    assert schema.day_from_filename("totally-unrelated.csv") is None
