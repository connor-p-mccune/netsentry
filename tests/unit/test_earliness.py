"""Decision latency: the tiers must be nested and honest, and the wait must be computed.

The claim the earliness report makes is that a flow which never showed a teardown cannot be
scored until the exporter's idle timer fires. That is an arithmetic claim about the latency
model and a structural claim about which features exist when, so both are pinned here rather
than left to the report's prose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.data import schema
from netsentry.evaluation.earliness import (
    MICROSECONDS,
    decision_latency_us,
    detected_within,
    emit_delay_us,
    observed_teardown,
)
from netsentry.features.feature_sets import (
    AVAILABILITY_TIERS,
    availability_sets,
    availability_tier,
)


# --------------------------------------------------------------------------------------
# The feature partition
# --------------------------------------------------------------------------------------
def test_tiers_are_nested_and_the_last_one_is_everything() -> None:
    sets = availability_sets()
    handshake, in_flight, complete = (set(sets[t]) for t in AVAILABILITY_TIERS)
    assert handshake < in_flight < complete
    assert complete == set(schema.feature_columns())


def test_every_feature_column_lands_in_exactly_one_tier() -> None:
    tiers = {c: availability_tier(c) for c in schema.feature_columns()}
    assert set(tiers.values()) <= set(AVAILABILITY_TIERS)
    assert all(t in AVAILABILITY_TIERS for t in tiers.values())


def test_accumulating_statistics_are_complete_only() -> None:
    """A prefix of a total is a different number, not a noisy version of the final one."""
    for feature in (
        "Total Fwd Packets",
        "Total Length of Bwd Packets",
        "Subflow Fwd Bytes",
        "Fwd IAT Total",
        "Flow Duration",
        "FIN Flag Count",
        "Idle Mean",
        "Active Max",
        "Fwd Header Length",
        "act_data_pkt_fwd",
    ):
        assert availability_tier(feature) == "complete", feature


def test_intensive_statistics_are_available_in_flight() -> None:
    """Means, extremes, spreads, rates and ratios are estimable from a prefix."""
    for feature in (
        "Flow Bytes/s",
        "Flow IAT Mean",
        "Fwd Packet Length Max",
        "Packet Length Std",
        "Down/Up Ratio",
        "Average Packet Size",
    ):
        assert availability_tier(feature) == "in_flight", feature


def test_setup_fields_are_available_at_the_handshake() -> None:
    for feature in ("Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward"):
        assert availability_tier(feature) == "handshake", feature


def test_no_identifier_column_can_reach_any_tier() -> None:
    """The leakage contract survives the new partition (rules/ml.md section 1)."""
    every = set(availability_sets(include_destination_port=True)["complete"])
    assert every.isdisjoint(schema.IDENTIFIER_COLUMNS)


def test_the_port_only_appears_when_it_is_explicitly_enabled() -> None:
    assert schema.DESTINATION_PORT not in availability_sets()["complete"]
    assert schema.DESTINATION_PORT in availability_sets(include_destination_port=True)["handshake"]


# --------------------------------------------------------------------------------------
# The wait
# --------------------------------------------------------------------------------------
def test_a_flow_that_showed_a_teardown_is_exported_at_its_own_duration() -> None:
    delay = emit_delay_us(np.array([1000.0, 250.0]), np.array([True, True]), 120_000_000)
    assert np.allclose(delay, [1000.0, 250.0])


def test_a_flow_that_merely_stopped_waits_out_the_idle_timer() -> None:
    """The report's central number: no FIN, no RST, so the exporter cannot know it is over."""
    delay = emit_delay_us(np.array([1000.0]), np.array([False]), 120_000_000)
    assert delay[0] == pytest.approx(120_001_000.0)


def test_the_timeout_dominates_a_short_unclosed_flow() -> None:
    """A single unanswered scan probe lasts microseconds and is invisible for two minutes."""
    delay = emit_delay_us(np.array([5.0]), np.array([False]), 120_000_000)
    assert delay[0] / MICROSECONDS > 119.0


def test_teardown_is_read_from_either_flag_and_is_false_when_neither_fired() -> None:
    frame = pd.DataFrame(
        {"FIN Flag Count": [1, 0, 0, 0], "RST Flag Count": [0, 1, 0, 0], "other": [9, 9, 9, 9]}
    )
    assert list(observed_teardown(frame)) == [True, True, False, False]


def test_teardown_defaults_to_false_when_the_flag_columns_are_absent() -> None:
    """Missing columns must not silently claim every flow closed itself."""
    assert not observed_teardown(pd.DataFrame({"Flow Duration": [1.0, 2.0]})).any()


def test_handshake_decides_at_once_and_in_flight_is_capped_by_the_flow_itself() -> None:
    frame = pd.DataFrame(
        {"Flow Duration": [200.0, 5_000_000.0], "FIN Flag Count": [0, 1], "RST Flag Count": [0, 0]}
    )
    assert np.allclose(decision_latency_us(frame, "handshake", 120_000_000, 1_000_000), 0.0)
    in_flight = decision_latency_us(frame, "in_flight", 120_000_000, 1_000_000)
    assert np.allclose(in_flight, [200.0, 1_000_000.0])  # short flow ends before the horizon


def test_complete_latency_uses_the_timeout_only_for_unclosed_flows() -> None:
    frame = pd.DataFrame(
        {"Flow Duration": [200.0, 300.0], "FIN Flag Count": [0, 1], "RST Flag Count": [0, 0]}
    )
    latency = decision_latency_us(frame, "complete", 120_000_000, 1_000_000)
    assert np.allclose(latency, [120_000_200.0, 300.0])


# --------------------------------------------------------------------------------------
# The frontier
# --------------------------------------------------------------------------------------
def test_detected_within_divides_by_every_hostile_flow_not_the_detected_ones() -> None:
    """Dividing by the detected subset is the survivorship bias this curve exists to avoid."""
    latency = np.array([0.5, 0.5, 0.5, 0.5]) * MICROSECONDS
    detected = np.array([True, False, False, False])
    assert detected_within(latency, detected, np.array([1.0]))[0] == pytest.approx(0.25)


def test_detected_within_is_non_decreasing_in_the_horizon() -> None:
    rng = np.random.default_rng(0)
    latency = rng.uniform(0, 200, size=500) * MICROSECONDS
    detected = rng.random(500) < 0.4
    curve = detected_within(latency, detected, np.array([0.1, 1.0, 10.0, 60.0, 300.0]))
    assert np.all(np.diff(curve) >= -1e-12)


def test_a_flow_detected_after_the_horizon_does_not_count_towards_it() -> None:
    latency = np.array([10.0, 200.0]) * MICROSECONDS
    detected = np.array([True, True])
    assert detected_within(latency, detected, np.array([60.0]))[0] == pytest.approx(0.5)
