"""Wald's SPRT: boundaries, evidence accounting, stopping, and the repeated-trials arithmetic.

Every claim the report makes about error control and speed is a property of the sequential
test itself, checkable in closed form against Wald's 1945 construction. These pin the
boundaries, the log-likelihood increments, the stopping rule (including the undecided
outcome a fixed-window test hides), and the closed form for why a per-flow budget gives no
host-level guarantee.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netsentry.intel.sequential import (
    CLEAN,
    COMPROMISED,
    UNDECIDED,
    expected_sample_number,
    hypothesis_rates,
    llr_increments,
    naive_host_false_alarm,
    simulate_host_stream,
    sprt_decide,
    wald_boundaries,
)


def test_wald_boundaries_match_the_closed_form() -> None:
    upper, lower = wald_boundaries(0.01, 0.10)
    assert upper == pytest.approx(math.log(0.90 / 0.01))
    assert lower == pytest.approx(math.log(0.10 / 0.99))
    assert upper > 0 > lower  # evidence must be able to fall either way


def test_tighter_error_rates_widen_the_boundaries() -> None:
    loose_up, loose_low = wald_boundaries(0.10, 0.10)
    tight_up, tight_low = wald_boundaries(0.001, 0.001)
    # Demanding fewer errors means demanding more evidence in both directions.
    assert tight_up > loose_up and tight_low < loose_low


def test_invalid_error_rates_are_rejected() -> None:
    for bad in ((0.0, 0.1), (0.1, 1.0), (-0.1, 0.1)):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            wald_boundaries(*bad)


def test_compromised_hypothesis_is_a_mixture_not_the_bare_tpr() -> None:
    # A 10%-compromised host runs 90% ordinary traffic, which alerts at the FPR. Modelling
    # H1 as the bare TPR would describe a host whose every flow is an attack.
    p1, p0 = hypothesis_rates(tpr=0.5, fpr=0.01, compromise_mix=0.10)
    assert p1 == pytest.approx(0.10 * 0.5 + 0.90 * 0.01)
    assert p0 == pytest.approx(0.01)
    assert p1 < 0.5


def test_a_fully_compromised_host_alerts_at_the_tpr() -> None:
    p1, _ = hypothesis_rates(tpr=0.5, fpr=0.01, compromise_mix=1.0)
    assert p1 == pytest.approx(0.5)


def test_an_uncompromised_mixture_collapses_to_the_clean_hypothesis() -> None:
    p1, p0 = hypothesis_rates(tpr=0.5, fpr=0.01, compromise_mix=0.0)
    assert p1 == pytest.approx(p0)


def test_an_alerting_flow_carries_positive_evidence_and_a_quiet_one_negative() -> None:
    inc = llr_increments(np.array([True, False]), p1=0.5, p0=0.001)
    assert inc[0] == pytest.approx(math.log(0.5 / 0.001))
    assert inc[1] == pytest.approx(math.log(0.5 / 0.999))
    assert inc[0] > 0 > inc[1]


def test_evidence_per_alert_grows_as_the_hypotheses_separate() -> None:
    weak = llr_increments(np.array([True]), p1=0.2, p0=0.1)[0]
    strong = llr_increments(np.array([True]), p1=0.9, p0=0.001)[0]
    assert strong > weak


def test_indistinguishable_hypotheses_carry_no_evidence_either_way() -> None:
    inc = llr_increments(np.array([True, False]), p1=0.3, p0=0.3)
    assert np.allclose(inc, 0.0)


def test_the_mixture_hypothesis_makes_quiet_flows_nearly_uninformative() -> None:
    # The bug this replaced: with H1 = bare TPR, a quiet flow carried enough evidence to
    # acquit a compromised host in a few dozen flows. Under the mixture it carries almost
    # none, so the test waits for alerts instead of acquitting on silence.
    bare = abs(llr_increments(np.array([False]), p1=0.09, p0=0.001)[0])
    mixed_p1, p0 = hypothesis_rates(tpr=0.09, fpr=0.001, compromise_mix=0.10)
    mixed = abs(llr_increments(np.array([False]), p1=mixed_p1, p0=p0)[0])
    assert mixed < bare / 10


def test_sprt_stops_at_the_first_upper_crossing() -> None:
    upper, lower = 2.0, -2.0
    # Cumulative: 1.0, 2.0 -> crosses on the second observation and stops there.
    verdict, n = sprt_decide(np.array([1.0, 1.0, 1.0, 1.0]), upper, lower)
    assert verdict == COMPROMISED and n == 2


def test_sprt_stops_at_the_first_lower_crossing() -> None:
    verdict, n = sprt_decide(np.array([-1.0, -1.5, 5.0]), 2.0, -2.0)
    assert verdict == CLEAN and n == 2


def test_sprt_reports_undecided_rather_than_forcing_a_call() -> None:
    # The outcome a fixed-window rule silently converts into a false negative.
    verdict, n = sprt_decide(np.array([0.1, -0.1, 0.1]), 2.0, -2.0)
    assert verdict == UNDECIDED and n == 3


def test_sprt_on_an_empty_stream_is_undecided() -> None:
    assert sprt_decide(np.zeros(0), 1.0, -1.0) == (UNDECIDED, 0)


def test_expected_sample_number_falls_as_the_detector_improves() -> None:
    weak = expected_sample_number(0.01, 0.1, p1=0.15, p0=0.01, under="H1")
    strong = expected_sample_number(0.01, 0.1, p1=0.90, p0=0.001, under="H1")
    assert strong < weak and strong > 0


def test_expected_sample_number_is_positive_under_both_hypotheses() -> None:
    for under in ("H0", "H1"):
        assert expected_sample_number(0.01, 0.1, 0.5, 0.01, under=under) > 0


def test_expected_sample_number_is_infinite_for_a_useless_detector() -> None:
    # No evidence per flow means the walk never drifts to a boundary.
    assert expected_sample_number(0.01, 0.1, 0.3, 0.3, under="H1") == float("inf")


def test_naive_host_false_alarm_grows_with_host_chattiness() -> None:
    # The report's arithmetic: a 0.1% per-flow budget over 1000 flows is a ~63% host-level
    # false alarm. The rate is fixed; the number of trials is not.
    assert naive_host_false_alarm(0.001, 1000) == pytest.approx(0.6323, abs=1e-3)
    assert naive_host_false_alarm(0.001, 10) < naive_host_false_alarm(0.001, 100)


def test_naive_host_false_alarm_is_zero_for_a_silent_host() -> None:
    assert naive_host_false_alarm(0.5, 0) == 0.0


def test_a_clean_host_stream_contains_only_benign_scores() -> None:
    benign, attack = np.zeros(50), np.ones(50)
    stream = simulate_host_stream(benign, attack, 0.0, 200, np.random.default_rng(0))
    assert len(stream) == 200 and (stream == 0.0).all()


def test_a_compromised_host_stream_mixes_in_roughly_the_requested_share() -> None:
    benign, attack = np.zeros(50), np.ones(50)
    stream = simulate_host_stream(benign, attack, 0.25, 4000, np.random.default_rng(1))
    assert len(stream) == 4000
    assert stream.mean() == pytest.approx(0.25, abs=0.03)
