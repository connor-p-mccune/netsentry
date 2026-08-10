"""Group DRO: the adversary's update and the reweighting that makes it DRO and not bagging.

The distinction that matters is subtle and easy to get wrong: dividing each group's weight
by its size is what turns "weight some rows more" into "optimise the worst per-group
average". These pin that, plus the multiplicative update's direction and stability.
"""

from __future__ import annotations

import numpy as np

from netsentry.training.dro import (
    exponentiated_gradient_step,
    group_losses,
    sample_weights_from_groups,
    worst_group,
)

GROUPS = np.array(["http", "http", "http", "dns", "ssh"])
NAMES = ["http", "dns", "ssh"]


def test_group_losses_are_per_group_means_not_pooled() -> None:
    y = np.array([1, 1, 1, 1, 0])
    # http is confidently right, dns is confidently wrong: the pooled mean would hide it.
    scores = np.array([0.99, 0.99, 0.99, 0.01, 0.01])
    losses = group_losses(y, scores, GROUPS, NAMES)
    assert losses[0] < 0.02  # http
    assert losses[1] > 4.0  # dns
    assert losses[2] < 0.02  # ssh


def test_group_losses_report_zero_for_a_group_with_no_rows() -> None:
    losses = group_losses(np.array([1]), np.array([0.5]), np.array(["http"]), ["http", "absent"])
    assert losses[1] == 0.0


def test_exponentiated_gradient_moves_weight_toward_the_worst_group() -> None:
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    updated = exponentiated_gradient_step(w, np.array([0.1, 2.0, 0.1]), step_size=1.0)
    assert updated[1] > updated[0] and updated[1] > updated[2]
    assert np.isclose(updated.sum(), 1.0)


def test_exponentiated_gradient_leaves_equal_losses_untouched() -> None:
    w = np.array([0.5, 0.3, 0.2])
    updated = exponentiated_gradient_step(w, np.array([1.0, 1.0, 1.0]), step_size=3.0)
    assert np.allclose(updated, w)


def test_exponentiated_gradient_survives_a_loss_large_enough_to_overflow() -> None:
    """Shifting by the max before exponentiating is a no-op after renormalising."""
    w = np.array([0.5, 0.5])
    updated = exponentiated_gradient_step(w, np.array([0.0, 5_000.0]), step_size=10.0)
    assert np.all(np.isfinite(updated)) and np.isclose(updated.sum(), 1.0)
    assert updated[1] > updated[0]


def test_a_bigger_step_concentrates_weight_faster() -> None:
    w = np.array([0.5, 0.5])
    losses = np.array([0.1, 1.0])
    gentle = exponentiated_gradient_step(w, losses, step_size=0.5)
    sharp = exponentiated_gradient_step(w, losses, step_size=5.0)
    assert sharp[1] > gentle[1]


def test_sample_weights_divide_each_groups_weight_across_its_rows() -> None:
    """The property that makes this DRO: influence per group, not per row."""
    weights = sample_weights_from_groups(GROUPS, NAMES, np.array([1 / 3, 1 / 3, 1 / 3]))
    http_total = weights[GROUPS == "http"].sum()
    dns_total = weights[GROUPS == "dns"].sum()
    # http has three rows and dns one, yet the two groups carry the same total influence.
    assert np.isclose(http_total, dns_total)


def test_a_row_in_a_small_group_outweighs_a_row_in_a_large_one() -> None:
    weights = sample_weights_from_groups(GROUPS, NAMES, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert weights[GROUPS == "dns"][0] > weights[GROUPS == "http"][0]


def test_sample_weights_track_the_adversarys_emphasis() -> None:
    heavy_dns = sample_weights_from_groups(GROUPS, NAMES, np.array([0.1, 0.8, 0.1]))
    even = sample_weights_from_groups(GROUPS, NAMES, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert heavy_dns[GROUPS == "dns"].sum() > even[GROUPS == "dns"].sum()
    assert heavy_dns[GROUPS == "http"].sum() < even[GROUPS == "http"].sum()


def test_sample_weights_average_to_one_so_the_fit_sees_the_same_total_mass() -> None:
    weights = sample_weights_from_groups(GROUPS, NAMES, np.array([0.2, 0.5, 0.3]))
    assert np.isclose(weights.mean(), 1.0)


def test_worst_group_picks_the_lowest_value() -> None:
    assert worst_group({"http": 0.9, "dns": 0.2, "ssh": 0.5}) == ("dns", 0.2)


def test_worst_group_of_nothing_is_not_a_crash() -> None:
    name, value = worst_group({})
    assert name == "" and np.isnan(value)
