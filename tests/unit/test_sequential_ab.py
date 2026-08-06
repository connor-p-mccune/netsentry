"""Anytime-valid A/B testing: the mixture boundary, peeking inflation, and stopping.

The report's central claim — that repeatedly checking a fixed-n test inflates its error rate
while a confidence sequence survives the same behaviour — is simulated rather than asserted,
so it is worth pinning here on pure streams where the truth is known by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netsentry.evaluation.sequential_ab import (
    brier_loss,
    confidence_sequence,
    first_conclusive,
    fixed_n_required,
    fixed_n_significant,
    mixture_boundary,
    peeking_error_rate,
    sequence_error_rate,
)


def test_brier_loss_is_zero_for_a_perfect_prediction() -> None:
    assert brier_loss(np.array([1.0, 0.0]), np.array([1, 0])).tolist() == [0.0, 0.0]


def test_brier_loss_is_bounded_by_one() -> None:
    # Boundedness is what makes the variance proxy honest; log-loss would be unbounded.
    loss = brier_loss(np.array([0.0, 1.0]), np.array([1, 0]))
    assert loss.max() == 1.0


def test_mixture_boundary_matches_the_closed_form() -> None:
    v, rho, alpha = 100.0, 1.0, 0.05
    assert mixture_boundary(v, rho, alpha) == pytest.approx(
        math.sqrt((v + rho) * math.log((v + rho) / (rho * alpha**2)))
    )


def test_mixture_boundary_grows_with_accumulated_variance() -> None:
    assert mixture_boundary(1000.0, 1.0, 0.05) > mixture_boundary(10.0, 1.0, 0.05)


def test_a_tighter_alpha_widens_the_boundary() -> None:
    assert mixture_boundary(100.0, 1.0, 0.01) > mixture_boundary(100.0, 1.0, 0.10)


def test_mixture_boundary_rejects_invalid_parameters() -> None:
    for rho, alpha in ((0.0, 0.05), (-1.0, 0.05), (1.0, 0.0), (1.0, 1.0)):
        with pytest.raises(ValueError, match="rho must be positive"):
            mixture_boundary(10.0, rho, alpha)


def test_the_confidence_interval_narrows_as_evidence_accumulates() -> None:
    rng = np.random.default_rng(0)
    _, lower, upper = confidence_sequence(
        rng.normal(0.5, 1.0, size=5000), rho=1.0, alpha=0.05, sigma=1.0
    )
    widths = upper - lower
    assert widths[-1] < widths[100] < widths[10]


def test_the_interval_contains_the_true_mean_throughout() -> None:
    rng = np.random.default_rng(1)
    _, lower, upper = confidence_sequence(
        rng.normal(2.0, 1.0, size=3000), rho=1.0, alpha=0.05, sigma=1.0
    )
    # Anytime validity: coverage holds at *every* index, not just the last one.
    assert np.all((lower[50:] <= 2.0) & (upper[50:] >= 2.0))


def test_the_interval_eventually_excludes_zero_for_a_real_effect() -> None:
    rng = np.random.default_rng(2)
    _, lower, upper = confidence_sequence(
        rng.normal(1.0, 1.0, size=2000), rho=1.0, alpha=0.05, sigma=1.0
    )
    stop = first_conclusive(lower, upper)
    assert 0 < stop < 2000


def test_no_effect_means_no_conclusion() -> None:
    assert first_conclusive(np.array([-1.0, -0.5]), np.array([1.0, 0.5])) == 0


def test_first_conclusive_returns_the_earliest_crossing() -> None:
    lower = np.array([-1.0, -1.0, 0.2, 0.3])
    upper = np.array([1.0, 1.0, 1.2, 1.3])
    assert first_conclusive(lower, upper) == 3  # 1-based


def test_a_running_variance_plug_in_would_break_coverage() -> None:
    # Why sigma is fixed in advance: the sample variance of the first observation is zero,
    # so a plug-in interval starts infinitely narrow and fires immediately under the null.
    # Guarding the contract directly -- sigma must be a scale, not an estimate that shrinks.
    with pytest.raises(ValueError, match="sigma must be positive"):
        confidence_sequence(np.ones(10), rho=1.0, alpha=0.05, sigma=0.0)


def test_a_smaller_scale_proxy_gives_a_tighter_interval() -> None:
    stream = np.random.default_rng(9).normal(0.0, 0.1, size=1000)
    _, tight_lo, tight_hi = confidence_sequence(stream, rho=1.0, alpha=0.05, sigma=0.1)
    _, loose_lo, loose_hi = confidence_sequence(stream, rho=1.0, alpha=0.05, sigma=1.0)
    assert (tight_hi - tight_lo)[-1] < (loose_hi - loose_lo)[-1]


def test_confidence_sequence_on_an_empty_stream_is_empty() -> None:
    means, lower, upper = confidence_sequence(np.zeros(0), rho=1.0, alpha=0.05, sigma=1.0)
    assert len(means) == len(lower) == len(upper) == 0


def test_fixed_n_grows_as_the_effect_shrinks() -> None:
    big = fixed_n_required(0.5, 1.0, alpha=0.05, power=0.8)
    small = fixed_n_required(0.05, 1.0, alpha=0.05, power=0.8)
    assert small > big * 50  # n scales with 1/delta^2


def test_fixed_n_matches_the_textbook_formula() -> None:
    # (1.96 + 0.8416)^2 = 7.849 for a unit effect at 5% / 80%.
    assert fixed_n_required(1.0, 1.0, alpha=0.05, power=0.8) == 8


def test_fixed_n_of_a_zero_effect_is_undefined_and_returns_zero() -> None:
    assert fixed_n_required(0.0, 1.0, alpha=0.05, power=0.8) == 0


def test_fixed_n_test_detects_a_clear_effect_and_not_a_null() -> None:
    rng = np.random.default_rng(3)
    assert fixed_n_significant(rng.normal(1.0, 1.0, size=500), 0.05)
    assert not fixed_n_significant(rng.normal(0.0, 1.0, size=20), 0.05)


def test_peeking_inflates_the_fixed_n_error_rate_beyond_its_nominal_level() -> None:
    # The report's headline claim, on a stream that is null by construction.
    rng = np.random.default_rng(4)
    inflated = peeking_error_rate(200, 1000, 20, 0.05, rng)
    assert inflated > 0.05


def test_the_confidence_sequence_survives_the_same_peeking() -> None:
    rng = np.random.default_rng(5)
    assert sequence_error_rate(200, 1000, 0.05, 1.0, rng) <= 0.05


def test_error_rate_simulations_handle_degenerate_sizes() -> None:
    rng = np.random.default_rng(6)
    assert peeking_error_rate(0, 100, 5, 0.05, rng) == 0.0
    assert sequence_error_rate(10, 0, 0.05, 1.0, rng) == 0.0
