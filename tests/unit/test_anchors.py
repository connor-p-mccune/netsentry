"""Anchor search: the greedy rule finds a separating predicate and respects its guarantees.

These pin the behaviours the report leans on — a perfectly separating feature yields a
one-predicate, precision-1 anchor; a signal-free background never reaches the precision
target; a predicate without enough support is refused; and the lower confidence bound is a
sound floor on precision.
"""

from __future__ import annotations

import numpy as np

from netsentry.explain.anchors import _precision_lcb, greedy_anchor


def test_perfect_separator_is_a_one_predicate_anchor() -> None:
    rng = np.random.default_rng(0)
    n = 400
    f0 = rng.integers(0, 3, n)  # the separating feature
    f1 = rng.integers(0, 3, n)  # noise
    background_bins = np.column_stack([f0, f1])
    x_bins = np.array([1, 2])
    background_class = (f0 == 1).astype(int)  # class 1 exactly when f0 is in the flow's bin
    anchor = greedy_anchor(
        x_bins,
        background_bins,
        background_class,
        x_class=1,
        tau=0.9,
        max_predicates=3,
        min_match=5,
        z=1.64,
    )
    assert anchor.features[0] == 0  # picked the separating feature first
    assert anchor.precision > 0.99
    assert len(anchor.features) == 1  # cleared the threshold immediately, no padding


def test_no_signal_never_reaches_the_precision_target() -> None:
    rng = np.random.default_rng(1)
    n = 1200
    background_bins = rng.integers(0, 3, (n, 3))
    background_class = (rng.random(n) < 0.5).astype(int)  # independent of every bin
    x_bins = np.array([0, 1, 2])
    anchor = greedy_anchor(
        x_bins,
        background_bins,
        background_class,
        x_class=1,
        tau=0.95,
        max_predicates=3,
        min_match=40,
        z=1.64,
    )
    assert anchor.precision_lcb < 0.95  # the guarantee never clears the target
    assert len(anchor.features) == 3  # so it exhausts the predicate budget


def test_predicate_without_support_is_refused() -> None:
    n = 300
    f0 = np.zeros(n, dtype=int)
    f0[:3] = 1  # the flow's bin for f0 has only 3 rows
    f1 = np.arange(n) % 3
    background_bins = np.column_stack([f0, f1])
    background_class = (f0 == 1).astype(int)  # f0 would be perfect, but lacks support
    x_bins = np.array([1, 0])
    anchor = greedy_anchor(
        x_bins,
        background_bins,
        background_class,
        x_class=1,
        tau=0.9,
        max_predicates=2,
        min_match=20,
        z=1.64,
    )
    assert 0 not in anchor.features  # too few matching rows to trust it


def test_precision_lcb_is_a_sound_floor() -> None:
    assert _precision_lcb(100, 100, 1.64) == 1.0  # unanimous -> no downward slack
    assert _precision_lcb(0, 0, 1.64) == 0.0  # no data -> zero
    lcb = _precision_lcb(50, 100, 1.64)
    assert 0.0 < lcb < 0.5  # strictly below the point estimate of 0.5
    assert np.isclose(lcb, 0.5 - 1.64 * np.sqrt(0.25 / 100))
