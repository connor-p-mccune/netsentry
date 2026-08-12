"""The as-of join: what each flow may see, and the ways a naive implementation lets in the future.

Point-in-time correctness is a property that is easy to claim and easy to get subtly wrong — an
inclusive window boundary, a tie at one-second resolution, an event that arrives out of order.
Each of those is a separate test here, on frames small enough that the right answer can be
counted by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.features.store import (
    FEATURE_NAMES,
    as_of_features,
    leakage_gap,
    leaky_features,
)

KWARGS = {
    "entity_column": "src",
    "time_column": "ts",
    "dest_host_column": "dst",
    "dest_port_column": "port",
}


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["src", "ts", "dst", "port"])


def test_the_first_flow_of_a_host_has_no_context() -> None:
    frame = _frame([("a", "2017-07-03 09:00:00", "x", 80)])
    out = as_of_features(frame, lookback_seconds=60, **KWARGS)
    assert list(out.iloc[0]) == [0.0, 0.0, 0.0, 0.0]


def test_context_counts_only_that_hosts_own_earlier_flows() -> None:
    # Three flows from `a` and two from `b`, interleaved. `a`'s third flow must see exactly the
    # two earlier `a` flows -- never `b`'s, however close in time.
    frame = _frame(
        [
            ("a", "2017-07-03 09:00:00", "x", 80),
            ("b", "2017-07-03 09:00:01", "y", 443),
            ("a", "2017-07-03 09:00:02", "y", 443),
            ("b", "2017-07-03 09:00:03", "z", 22),
            ("a", "2017-07-03 09:00:04", "z", 22),
        ]
    )
    out = as_of_features(frame, lookback_seconds=60, **KWARGS)
    assert out["ctx_flows_in_window"].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert out["ctx_distinct_dest_hosts"].iloc[4] == 2.0
    assert out["ctx_distinct_dest_ports"].iloc[4] == 2.0


def test_a_flow_never_sees_its_own_instant() -> None:
    # Two flows from one host at the *same* second. Neither may count the other: at one-second
    # resolution, "simultaneous" is the most common way a label-bearing row leaks into itself.
    frame = _frame(
        [
            ("a", "2017-07-03 09:00:00", "x", 80),
            ("a", "2017-07-03 09:00:00", "y", 443),
        ]
    )
    out = as_of_features(frame, lookback_seconds=60, **KWARGS)
    assert out["ctx_flows_in_window"].tolist() == [0.0, 0.0]


def test_events_outside_the_lookback_window_are_excluded() -> None:
    frame = _frame(
        [
            ("a", "2017-07-03 09:00:00", "x", 80),
            ("a", "2017-07-03 09:00:30", "y", 443),
            ("a", "2017-07-03 09:02:00", "z", 22),  # 90s after the second, 120s after the first
        ]
    )
    out = as_of_features(frame, lookback_seconds=60, **KWARGS)
    assert out["ctx_flows_in_window"].tolist() == [0.0, 1.0, 0.0]


def test_the_window_is_closed_on_the_left_and_open_on_the_right() -> None:
    # `[t - lookback, t)`: an event exactly `lookback` seconds old is the oldest one still
    # visible, and one second older than that is gone. Pinned because an off-by-one here is
    # invisible in aggregate and changes every boundary row.
    frame = _frame(
        [
            ("a", "2017-07-03 09:00:00", "x", 80),
            ("a", "2017-07-03 09:01:00", "y", 443),
        ]
    )
    assert as_of_features(frame, lookback_seconds=60, **KWARGS)["ctx_flows_in_window"].iloc[1] == 1
    assert as_of_features(frame, lookback_seconds=59, **KWARGS)["ctx_flows_in_window"].iloc[1] == 0


def test_context_is_independent_of_row_order_in_the_frame() -> None:
    # The sweep sorts by time internally, so a frame handed to it out of order must produce the
    # same per-row context -- otherwise the store's answer would depend on file layout.
    rows = [
        ("a", "2017-07-03 09:00:00", "x", 80),
        ("a", "2017-07-03 09:00:10", "y", 443),
        ("a", "2017-07-03 09:00:20", "z", 22),
    ]
    ordered = as_of_features(_frame(rows), lookback_seconds=60, **KWARGS)
    shuffled_frame = _frame([rows[2], rows[0], rows[1]])
    shuffled = as_of_features(shuffled_frame, lookback_seconds=60, **KWARGS)
    assert shuffled["ctx_flows_in_window"].tolist() == [2.0, 0.0, 1.0]
    assert sorted(ordered["ctx_flows_in_window"]) == sorted(shuffled["ctx_flows_in_window"])


def test_as_of_features_rejects_a_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        as_of_features(
            _frame([("a", "2017-07-03 09:00:00", "x", 80)]), lookback_seconds=0, **KWARGS
        )


def test_the_leaky_join_gives_every_flow_the_hosts_whole_capture_totals() -> None:
    # The bug this exists to demonstrate: the *first* flow of a host is told how many flows that
    # host will make in total, including ones hours in its future.
    frame = _frame(
        [
            ("a", "2017-07-03 09:00:00", "x", 80),
            ("a", "2017-07-03 17:00:00", "y", 443),
            ("a", "2017-07-03 17:00:01", "z", 22),
        ]
    )
    leaky = leaky_features(
        frame, entity_column="src", dest_host_column="dst", dest_port_column="port"
    )
    assert leaky["ctx_flows_in_window"].tolist() == [3.0, 3.0, 3.0]
    as_of = as_of_features(frame, lookback_seconds=60, **KWARGS)
    assert as_of["ctx_flows_in_window"].iloc[0] == 0.0  # the honest answer for the first flow


def test_leakage_gap_reports_the_ratio_between_the_two_joins() -> None:
    as_of = pd.DataFrame({name: [1.0, 3.0] for name in FEATURE_NAMES})
    leaky = pd.DataFrame({name: [4.0, 4.0] for name in FEATURE_NAMES})
    gaps = leakage_gap(as_of, leaky)
    assert all(np.isclose(v, 2.0) for v in gaps.values())  # mean 4 vs mean 2


def test_leakage_gap_is_unbounded_when_the_honest_context_is_empty() -> None:
    as_of = pd.DataFrame({name: [0.0, 0.0] for name in FEATURE_NAMES})
    leaky = pd.DataFrame({name: [5.0, 5.0] for name in FEATURE_NAMES})
    assert all(np.isinf(v) for v in leakage_gap(as_of, leaky).values())
