"""Survival analysis: censored subjects must count, and a missing median must stay missing.

Kaplan-Meier is easy to implement as "one minus the empirical CDF of the detected ones",
which is exactly the bias the report exists to remove. The first tests pin it against
hand-computed values where a censored subject changes the answer, so that shortcut would
fail them.
"""

from __future__ import annotations

import math

import numpy as np

from netsentry.evaluation.survival import (
    Episode,
    build_episodes,
    kaplan_meier,
    logrank_test,
    median_survival,
    restricted_mean,
)


def test_with_no_censoring_survival_is_one_minus_the_empirical_cdf() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0])
    curve = kaplan_meier(times, np.ones(4, dtype=int))
    assert np.allclose(curve.survival, [0.75, 0.5, 0.25, 0.0])


def test_a_censored_subject_stays_in_the_denominator_until_it_leaves() -> None:
    """Hand-computed: 5 subjects, detections at 2 and 4, one censored at 3.

    At t=2: 5 at risk, 1 detected -> S = 4/5 = 0.8.
    The censored subject leaves at t=3 having contributed to that first denominator.
    At t=4: 3 at risk (the 4.0 and 5.0 subjects plus the detected-at-4 one), 1 detected
    -> S = 0.8 * 2/3 = 0.5333...
    """
    times = np.array([2.0, 3.0, 4.0, 5.0, 5.0])
    events = np.array([1, 0, 1, 0, 0])
    curve = kaplan_meier(times, events)
    assert np.allclose(curve.times, [2.0, 4.0])
    assert np.allclose(curve.survival, [0.8, 0.8 * 2 / 3])
    assert curve.n_censored == 3


def test_dropping_the_censored_subjects_would_give_a_different_answer() -> None:
    """The bias this module removes, demonstrated: the naive curve is optimistic."""
    times = np.array([2.0, 3.0, 4.0, 5.0, 5.0])
    events = np.array([1, 0, 1, 0, 0])
    honest = kaplan_meier(times, events)
    naive = kaplan_meier(times[events == 1], events[events == 1])
    # The naive curve, seeing only detections, claims everything is detected by t=4.
    assert naive.survival[-1] == 0.0
    assert honest.survival[-1] > 0.5


def test_ties_at_one_time_are_handled_as_a_single_step() -> None:
    curve = kaplan_meier(np.array([3.0, 3.0, 5.0]), np.array([1, 1, 1]))
    assert np.allclose(curve.times, [3.0, 5.0])
    assert np.allclose(curve.survival, [1 / 3, 0.0])


def test_survival_is_monotone_non_increasing() -> None:
    rng = np.random.default_rng(0)
    times = rng.integers(1, 40, size=300).astype(float)
    events = rng.integers(0, 2, size=300)
    curve = kaplan_meier(times, events)
    assert np.all(np.diff(curve.survival) <= 1e-12)


def test_confidence_band_never_escapes_zero_to_one() -> None:
    rng = np.random.default_rng(1)
    curve = kaplan_meier(rng.integers(1, 20, size=60).astype(float), rng.integers(0, 2, size=60))
    assert np.all(curve.lower >= 0.0) and np.all(curve.upper <= 1.0)
    assert np.all(curve.lower <= curve.survival) and np.all(curve.upper >= curve.survival)


def test_median_is_the_first_time_survival_reaches_one_half() -> None:
    curve = kaplan_meier(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, dtype=int))
    assert median_survival(curve) == 2.0


def test_median_is_infinite_when_most_subjects_are_never_detected() -> None:
    """The honest non-answer: a median that does not exist must not be invented."""
    times = np.array([5.0, 10.0, 10.0, 10.0, 10.0])
    events = np.array([1, 0, 0, 0, 0])
    assert math.isinf(median_survival(kaplan_meier(times, events)))


def test_restricted_mean_of_a_never_detected_cohort_is_the_whole_horizon() -> None:
    curve = kaplan_meier(np.array([10.0, 10.0, 10.0]), np.zeros(3, dtype=int))
    assert restricted_mean(curve, 10.0) == 10.0


def test_restricted_mean_is_the_area_under_the_step_curve() -> None:
    # Two subjects, detected at t=1 and t=2. S = 0.5 on [1,2), 0 after.
    curve = kaplan_meier(np.array([1.0, 2.0]), np.ones(2, dtype=int))
    # Area up to 3 = 1*1.0 (on [0,1)) + 1*0.5 (on [1,2)) + 1*0 = 1.5
    assert math.isclose(restricted_mean(curve, 3.0), 1.5)


def test_restricted_mean_falls_as_detection_gets_faster() -> None:
    slow = kaplan_meier(np.array([8.0, 9.0, 10.0]), np.ones(3, dtype=int))
    fast = kaplan_meier(np.array([1.0, 2.0, 3.0]), np.ones(3, dtype=int))
    assert restricted_mean(fast, 10.0) < restricted_mean(slow, 10.0)


def test_logrank_finds_nothing_between_two_identical_groups() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 1, 0, 1, 0])
    chi2, p = logrank_test(times, events, times.copy(), events.copy())
    assert chi2 < 1e-9 and p > 0.99


def test_logrank_separates_two_clearly_different_groups() -> None:
    fast = np.arange(1.0, 31.0)
    slow = np.arange(60.0, 90.0)
    chi2, p = logrank_test(fast, np.ones(30, dtype=int), slow, np.ones(30, dtype=int))
    assert chi2 > 20.0 and p < 0.001


def test_logrank_p_value_matches_the_chi_square_tail() -> None:
    """One degree of freedom: P(X > x) = erfc(sqrt(x/2)). Checked against known quantiles."""
    fast = np.array([1.0, 2.0, 3.0])
    slow = np.array([10.0, 11.0, 12.0])
    chi2, p = logrank_test(fast, np.ones(3, dtype=int), slow, np.ones(3, dtype=int))
    assert math.isclose(p, math.erfc(math.sqrt(chi2 / 2.0)), rel_tol=1e-12)


def test_logrank_returns_a_null_result_when_nobody_is_detected() -> None:
    zeros = np.zeros(4, dtype=int)
    chi2, p = logrank_test(np.arange(4.0), zeros, np.arange(4.0), zeros)
    assert chi2 == 0.0 and p == 1.0


def test_build_episodes_times_the_first_alert_within_each_burst() -> None:
    labels = np.array(["DDoS"] * 6)
    days = np.array(["Friday"] * 6)
    # Burst 1: third flow alerts. Burst 2: nothing alerts.
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.1, 0.2])
    episodes = build_episodes(labels, days, scores, 0.5, "BENIGN", episode_flows=3)
    assert [(e.time, e.detected) for e in episodes] == [(3, True), (3, False)]


def test_build_episodes_ignores_benign_flows_entirely() -> None:
    labels = np.array(["BENIGN"] * 4)
    episodes = build_episodes(
        labels, np.array(["Monday"] * 4), np.ones(4), 0.5, "BENIGN", episode_flows=2
    )
    assert episodes == []


def test_build_episodes_keeps_days_and_classes_apart() -> None:
    labels = np.array(["DDoS", "DDoS", "Bot", "Bot"])
    days = np.array(["Thursday", "Friday", "Thursday", "Friday"])
    episodes = build_episodes(labels, days, np.ones(4), 0.5, "BENIGN", episode_flows=10)
    # Each (day, class) pair has a single flow, which is below the two-flow minimum.
    assert episodes == []


def test_an_episode_is_a_plain_record() -> None:
    e = Episode("DDoS", "Friday", 7, True)
    assert (e.attack_class, e.day, e.time, e.detected) == ("DDoS", "Friday", 7, True)
