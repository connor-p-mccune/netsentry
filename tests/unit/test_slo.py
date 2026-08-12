"""SLO arithmetic: budgets, burn rates, detection times, and the generated Prometheus rules.

Every quantity in the report is closed form, so every quantity has an answer that can be
derived by hand. The values pinned here are the ones from the SRE Workbook's worked example
(a 30-day period, a 14.4x burn exhausting the budget in two days), plus the rolling-window
logic the replay depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.monitoring.slo import (
    AlertPolicy,
    SLIDefinition,
    budget_burn_time,
    budget_consumed_at_detection,
    burn_rate,
    default_policies,
    error_budget,
    first_firing,
    render_prometheus_rules,
    rolling_rate,
    time_to_detect,
)

PERIOD_HOURS = 30 * 24.0


def test_error_budget_is_the_complement_of_the_objective() -> None:
    assert error_budget(0.999) == pytest.approx(0.001)
    assert error_budget(0.99) == pytest.approx(0.01)


def test_error_budget_rejects_a_degenerate_objective() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        error_budget(1.0)


def test_burn_rate_of_one_means_exactly_on_plan() -> None:
    assert burn_rate(0.001, 0.001) == pytest.approx(1.0)
    assert burn_rate(0.0144, 0.001) == pytest.approx(14.4)


def test_burn_rate_rejects_a_zero_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        burn_rate(0.01, 0.0)


def test_a_14_4x_burn_exhausts_a_30_day_budget_in_two_days() -> None:
    # The canonical SRE Workbook figure: 30 days / 14.4 = 50 hours ~ 2 days.
    hours = budget_burn_time(0.001, 0.0144, PERIOD_HOURS)
    assert hours == pytest.approx(50.0, rel=1e-6)


def test_budget_burn_time_is_infinite_when_nothing_is_going_wrong() -> None:
    assert budget_burn_time(0.001, 0.0, PERIOD_HOURS) == float("inf")


def test_time_to_detect_is_the_window_fraction_the_average_needs_to_climb() -> None:
    # A rate 4x the alerting threshold pulls a 1-hour moving average up to the threshold in a
    # quarter of the window.
    budget, threshold = 0.001, 14.4
    actual = 4 * threshold * budget
    assert time_to_detect(1.0, threshold, budget, actual) == pytest.approx(0.25)


def test_time_to_detect_is_infinite_when_the_window_can_never_average_that_high() -> None:
    # A bad rate below the alerting threshold never lifts the window average past it.
    assert time_to_detect(1.0, 14.4, 0.001, 0.001) == float("inf")
    assert time_to_detect(1.0, 14.4, 0.001, 0.0) == float("inf")


def test_budget_consumed_at_detection_matches_the_workbook_percentage() -> None:
    # The 1h/14.4x row is designed to page after 2% of a 30-day budget is spent, when the
    # incident is burning at exactly the threshold rate.
    budget = 0.001
    consumed = budget_consumed_at_detection(1.0, 14.4, budget, 14.4 * budget, PERIOD_HOURS)
    assert consumed == pytest.approx(0.02, rel=1e-6)


def test_the_six_hour_row_pages_after_five_percent_of_the_budget() -> None:
    budget = 0.001
    consumed = budget_consumed_at_detection(6.0, 6.0, budget, 6.0 * budget, PERIOD_HOURS)
    assert consumed == pytest.approx(0.05, rel=1e-6)


def test_rolling_rate_is_a_trailing_mean_that_warms_up_from_the_first_event() -> None:
    events = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rolling_rate(events, 2), [1.0, 0.5, 0.0, 0.0])


def test_rolling_rate_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rolling_rate(np.zeros(3), 0)


def test_a_policy_needs_both_windows_to_exceed_the_threshold() -> None:
    policy = AlertPolicy("page", 1.0, 5.0 / 60.0, 14.4)
    budget = 0.001
    over, under = 14.4 * budget * 2, 0.0
    assert policy.fires(over, over, budget)
    assert not policy.fires(over, under, budget)  # the short window is the reset condition
    assert not policy.fires(under, over, budget)


def test_first_firing_finds_a_sustained_burst_and_ignores_a_clean_stream() -> None:
    budget = 0.01
    policy = AlertPolicy("page", 1.0, 0.5, 2.0)
    flows_per_hour = 100.0
    clean = np.zeros(500)
    assert first_firing(clean, policy, budget, flows_per_hour) is None
    bursty = clean.copy()
    bursty[200:300] = 1.0  # a solid block of alerts, far above 2% of either window
    fired = first_firing(bursty, policy, budget, flows_per_hour)
    assert fired is not None and 200 <= fired <= 210


def test_generated_rules_pair_every_policy_row_with_both_of_its_windows() -> None:
    sli = SLIDefinition("alert ratio", "alerts / scored flows", True, 0.98, 0.01)
    text = render_prometheus_rules(sli, default_policies())
    assert "netsentry:alert_ratio:rate1h" in text
    assert "netsentry:alert_ratio:rate5m" in text
    for policy in default_policies():
        assert f"long_window: {policy.name.split('/')[0]}" in text
        assert f"short_window: {policy.name.split('/')[1]}" in text
    # Every threshold is the burn multiple of the budget, not a hand-picked number.
    assert f"> {14.4 * sli.budget:.6g}" in text


def test_generated_rules_are_valid_yaml_prometheus_can_load() -> None:
    import yaml

    sli = SLIDefinition("alert ratio", "alerts / scored flows", True, 0.98, 0.01)
    parsed = yaml.safe_load(render_prometheus_rules(sli, default_policies()))
    names = {group["name"] for group in parsed["groups"]}
    assert names == {"netsentry-slo-recording", "netsentry-slo-burn"}
    burn = next(g for g in parsed["groups"] if g["name"] == "netsentry-slo-burn")
    assert len(burn["rules"]) == len(default_policies())
    assert {r["labels"]["severity"] for r in burn["rules"]} == {"page", "ticket"}


def test_first_firing_waits_for_the_window_to_fill_before_it_can_page() -> None:
    # A single alert in the first two events reads as a 50% rate on a half-filled window.
    # Scoring during warm-up would manufacture a page no running deployment would see.
    events = np.zeros(400)
    events[0] = 1.0
    policy = AlertPolicy("page", 1.0, 0.5, 2.0)
    assert first_firing(events, policy, 0.01, flows_per_hour=100.0) is None


def test_first_firing_declines_to_judge_a_window_longer_than_the_stream() -> None:
    policy = AlertPolicy("ticket", 72.0, 6.0, 1.0)
    assert first_firing(np.ones(50), policy, 0.01, flows_per_hour=100.0) is None


def test_generated_rule_thresholds_scale_with_the_objective() -> None:
    tight = SLIDefinition("alert ratio", "d", True, 0.999, 0.0)
    loose = SLIDefinition("alert ratio", "d", True, 0.9, 0.0)
    assert f"> {14.4 * tight.budget:.6g}" in render_prometheus_rules(tight, default_policies())
    assert f"> {14.4 * loose.budget:.6g}" in render_prometheus_rules(loose, default_policies())
