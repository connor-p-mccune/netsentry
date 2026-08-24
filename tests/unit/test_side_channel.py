"""Reading a verdict off the shape of a reply: the estimators, and the fix ladder.

The study's claim is that a *length* recovers a decision, so the two functions that turn a
length into a claim get tested directly: an AUC that reports leakage in either direction, and a
best-cut search whose optimum has to be the genuine one rather than a grid artefact. The fix
ladder gets tested against bodies built by hand, because the whole point is which change closes
the channel -- and a ladder that reported "closed" for the wrong rung would be worse than none.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config.settings import SideChannelConfig
from netsentry.serving.side_channel import (
    ChannelRow,
    SideChannelStudy,
    _fields,
    _fix_ladder,
    separation,
    serialised_size,
    threshold_accuracy,
)

# --------------------------------------------------------------------------------------
# The estimators.
# --------------------------------------------------------------------------------------


def test_a_perfectly_separating_signal_scores_one() -> None:
    verdicts = np.array([0, 0, 1, 1])
    assert separation(np.array([10.0, 11.0, 20.0, 21.0]), verdicts) == pytest.approx(1.0)


def test_leakage_is_reported_in_either_direction() -> None:
    """A body that is reliably *shorter* on an alert leaks exactly as much as a longer one."""
    verdicts = np.array([0, 0, 1, 1])
    assert separation(np.array([20.0, 21.0, 10.0, 11.0]), verdicts) == pytest.approx(1.0)


def test_a_constant_signal_carries_nothing() -> None:
    assert separation(np.full(8, 512.0), np.array([0, 1] * 4)) == pytest.approx(0.5)


def test_separation_is_defined_when_every_verdict_is_the_same() -> None:
    assert separation(np.arange(5.0), np.zeros(5)) == pytest.approx(0.5)


def test_the_best_cut_is_the_genuine_optimum() -> None:
    """Found by a prefix scan, so it cannot be an artefact of a grid's resolution."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    verdicts = np.array([0, 0, 0, 1, 1, 1])
    cut, accuracy = threshold_accuracy(signal, verdicts)
    assert accuracy == pytest.approx(1.0)
    assert 3.0 <= cut <= 4.0


def test_the_best_cut_beats_the_majority_class_when_there_is_signal() -> None:
    signal = np.array([1.0, 1.0, 1.0, 9.0, 9.0])
    verdicts = np.array([0, 0, 0, 1, 1])
    assert threshold_accuracy(signal, verdicts)[1] == pytest.approx(1.0)


def test_a_signal_with_no_information_falls_back_to_the_base_rate() -> None:
    signal = np.array([5.0, 5.0, 5.0, 5.0])
    verdicts = np.array([0, 0, 0, 1])
    assert threshold_accuracy(signal, verdicts)[1] == pytest.approx(0.75)


def test_a_single_class_returns_its_own_share() -> None:
    assert threshold_accuracy(np.arange(4.0), np.ones(4))[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Response bodies.
# --------------------------------------------------------------------------------------


def _bodies() -> list[dict[str, object]]:
    """Two clear replies and two alerts, differing only in the fields the contract varies."""
    clear = {
        "predicted_class": "BENIGN",
        "is_attack": False,
        "attack_probability": 0.01,
        "mitre": None,
        "recommended_action": "auto_clear",
        "top_features": [1, 2, 3],
    }
    alert = {
        "predicted_class": "DDoS",
        "is_attack": True,
        "attack_probability": 0.99,
        "mitre": {"tactic": "Impact", "technique_id": "T1499"},
        "recommended_action": "auto_alert",
        "top_features": [1, 2, 3],
    }
    return [dict(clear), dict(clear), dict(alert), dict(alert)]


def test_serialised_size_is_stable_and_compact() -> None:
    body = {"a": 1, "b": "xy"}
    assert serialised_size(body) == len('{"a":1,"b":"xy"}')


def test_a_field_present_only_on_alerts_is_labelled_as_such() -> None:
    rows = {row.field: row for row in _fields(_bodies())}
    assert rows["mitre"].present_on == "**alerts only**"
    assert rows["top_features"].present_on == "always"


def test_null_fields_are_not_counted_as_present() -> None:
    """`mitre: null` on a clear reply is an absence, and treating it as presence hides it."""
    rows = {row.field: row for row in _fields(_bodies())}
    assert rows["mitre"].mean_bytes == pytest.approx(
        serialised_size({"tactic": "Impact", "technique_id": "T1499"}) - 2, abs=3
    )


# --------------------------------------------------------------------------------------
# The fix ladder.
# --------------------------------------------------------------------------------------


def _config(**kwargs: object) -> SideChannelConfig:
    return SideChannelConfig(**kwargs)  # type: ignore[arg-type]


def test_the_shipped_contract_leaks_completely() -> None:
    ladder = _fix_ladder(_bodies(), _config())
    assert ladder[0].size_auc == pytest.approx(1.0)
    assert ladder[0].added_bytes == pytest.approx(0.0)


def test_dropping_the_derivable_field_is_measured_rather_than_assumed() -> None:
    """The rung the report leans on: removing `mitre` has to be *checked*, not asserted."""
    ladder = _fix_ladder(_bodies(), _config())
    without_mitre = ladder[1]
    assert without_mitre.added_bytes < 0.0  # the reply gets smaller
    # It does not by itself close the channel here, because other fields still vary in length.
    assert without_mitre.size_auc > 0.5


def test_fixing_the_obvious_fields_is_not_enough() -> None:
    """The rung that catches everyone: `true` is four bytes and `false` is five.

    Enumerating the variable-length fields feels like the fix and is not one, because a boolean
    is a variable-length field. This test exists so a future contract change cannot make the
    ladder silently claim the channel is closed when it is not.
    """
    ladder = _fix_ladder(_bodies(), _config())
    assert ladder[2].size_auc > 0.5


def test_normalising_the_boolean_verdict_too_closes_the_channel() -> None:
    ladder = _fix_ladder(_bodies(), _config())
    assert ladder[3].size_auc == pytest.approx(0.5)


def test_padding_closes_the_channel_by_construction_and_costs_bytes() -> None:
    ladder = _fix_ladder(_bodies(), _config())
    assert ladder[-1].size_auc == pytest.approx(0.5)
    assert ladder[-1].added_bytes > 0.0


# --------------------------------------------------------------------------------------
# The records the report reads from.
# --------------------------------------------------------------------------------------


def _row(endpoint: str, size_auc: float) -> ChannelRow:
    return ChannelRow(
        endpoint=endpoint,
        size_auc=size_auc,
        size_accuracy=0.9,
        latency_auc=0.5,
        latency_accuracy=0.5,
        mean_size_benign=500.0,
        mean_size_alert=620.0,
        mean_ms_benign=10.0,
        mean_ms_alert=11.0,
    )


def test_the_worst_channel_is_the_one_that_leaks_most() -> None:
    study = SideChannelStudy(
        channels=[_row("a", 0.7), _row("b", 0.95)],
        fields=[],
        mitigations=[],
        n_flows=10,
        alert_share=0.2,
    )
    worst = study.worst()
    assert worst is not None and worst.endpoint == "b"
    assert study.channel("a") is not None
    assert study.channel("missing") is None


def test_the_size_gap_is_the_difference_a_watcher_sees() -> None:
    assert _row("a", 0.9).size_gap == pytest.approx(120.0)
