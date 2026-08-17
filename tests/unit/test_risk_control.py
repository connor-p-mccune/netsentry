"""Distribution-free risk control, tested against its guarantees.

A selector that returns a plausible threshold while its p-values are invalid is the failure
mode here: everything downstream still runs, the table still fills, and the certificate is
worthless. So the tests check the guarantees directly -- the p-value's validity under the null
by simulation, the monotone selector's boundary behaviour, and the empirical exceedance rate of
each selector against the level it claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netsentry.evaluation.risk_control import (
    crc_threshold,
    hoeffding_bentkus_pvalue,
    log_binomial_cdf,
    ltt_valid_set,
    multi_risk_valid_set,
)

# --------------------------------------------------------------------------------------
# The binomial tail the Bentkus bound needs.
# --------------------------------------------------------------------------------------


def test_log_binomial_cdf_matches_a_direct_sum() -> None:
    n, p = 30, 0.2
    for k in (0, 3, 10, 29):
        direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert math.exp(log_binomial_cdf(k, n, p)) == pytest.approx(direct, rel=1e-9)


def test_log_binomial_cdf_survives_a_large_n() -> None:
    # The whole reason for log space: at n = 5,000 the individual terms underflow float64.
    value = log_binomial_cdf(10, 5000, 0.01)
    assert -60.0 < value < 0.0


def test_log_binomial_cdf_edges() -> None:
    assert log_binomial_cdf(-1, 10, 0.5) == -math.inf
    assert log_binomial_cdf(10, 10, 0.5) == 0.0


# --------------------------------------------------------------------------------------
# p-value validity. The property that makes the certificate mean anything.
# --------------------------------------------------------------------------------------


def test_the_pvalue_is_one_when_the_empirical_risk_already_exceeds_the_target() -> None:
    assert hoeffding_bentkus_pvalue(0.30, 500, 0.10) == 1.0
    assert hoeffding_bentkus_pvalue(0.10, 500, 0.10) == 1.0


def test_the_pvalue_shrinks_with_evidence() -> None:
    tight = hoeffding_bentkus_pvalue(0.02, 2000, 0.10)
    loose = hoeffding_bentkus_pvalue(0.02, 50, 0.10)
    assert tight < loose < 1.0
    assert hoeffding_bentkus_pvalue(0.02, 500, 0.10) < hoeffding_bentkus_pvalue(0.08, 500, 0.10)


def test_the_pvalue_is_valid_under_the_null_by_simulation() -> None:
    # Validity means P(p <= delta) <= delta when the null is true (risk exactly at the
    # boundary). Simulate 2,000 calibration sets whose true risk *is* alpha and check the
    # rejection rate against the level. A p-value that fails this makes every certificate in
    # the report a lie, so it is worth 2,000 draws.
    rng = np.random.default_rng(0)
    alpha, n, delta = 0.10, 400, 0.10
    losses = rng.random((2000, n)) < alpha
    rejections = [hoeffding_bentkus_pvalue(float(row.mean()), n, alpha) <= delta for row in losses]
    assert float(np.mean(rejections)) <= delta


# --------------------------------------------------------------------------------------
# The selectors.
# --------------------------------------------------------------------------------------


def _grid_and_risks() -> tuple[np.ndarray, np.ndarray]:
    """A monotone risk curve: the threshold rises, the miss rate rises with it."""
    grid = np.linspace(0.0, 1.0, 11)
    risks = np.linspace(0.0, 1.0, 11)
    return grid, risks


def test_crc_picks_the_largest_threshold_inside_the_inflated_bound() -> None:
    grid, risks = _grid_and_risks()
    n = 99
    chosen = crc_threshold(grid, risks, n, alpha=0.31)
    # (n * R + 1) / (n + 1) <= 0.31 allows R <= 0.30, i.e. grid point 0.3.
    assert chosen == pytest.approx(0.3)


def test_crc_inflation_is_what_makes_it_conservative() -> None:
    # With a tiny calibration set the +B term dominates and the selector refuses everything
    # but the most permissive threshold. That is the guarantee working, not a bug.
    grid, risks = _grid_and_risks()
    assert crc_threshold(grid, risks, n=1, alpha=0.10) == pytest.approx(grid[0])


def test_ltt_returns_a_prefix_of_the_grid_under_fixed_sequence() -> None:
    grid, risks = _grid_and_risks()
    valid = ltt_valid_set(grid, risks, n=2000, alpha=0.30, delta=0.1)
    assert len(valid) > 0
    assert np.array_equal(valid, grid[: len(valid)])  # a prefix: it stops at the first failure
    assert valid[-1] < 0.30  # and never certifies a threshold whose risk is already at target


def test_ltt_is_more_conservative_than_crc_at_the_same_level() -> None:
    # The high-probability bound must cost something relative to the expectation bound, or one
    # of the two is implemented wrongly.
    grid, risks = _grid_and_risks()
    n = 500
    valid = ltt_valid_set(grid, risks, n, alpha=0.30, delta=0.1)
    assert float(valid[-1]) <= crc_threshold(grid, risks, n, alpha=0.30)


def test_ltt_bonferroni_is_no_more_permissive_than_fixed_sequence() -> None:
    grid, risks = _grid_and_risks()
    sequence = ltt_valid_set(grid, risks, 500, 0.30, 0.1, method="fixed_sequence")
    bonferroni = ltt_valid_set(grid, risks, 500, 0.30, 0.1, method="bonferroni")
    assert float(bonferroni.max()) <= float(sequence.max())


def test_ltt_returns_nothing_when_nothing_is_certifiable() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    risks = np.full(5, 0.9)  # every threshold misses 90% of attacks
    assert len(ltt_valid_set(grid, risks, 500, alpha=0.1, delta=0.1)) == 0


# --------------------------------------------------------------------------------------
# Two risks at once.
# --------------------------------------------------------------------------------------


def test_multi_risk_control_finds_the_band_where_both_hold() -> None:
    grid = np.linspace(0.0, 1.0, 21)
    miss = grid.copy()  # rises with the threshold
    volume = 1.0 - grid  # falls with it
    feasible = multi_risk_valid_set(
        grid, miss, volume, 4000, 4000, alpha_miss=0.6, alpha_volume=0.6, delta=0.1
    )
    assert len(feasible) > 0
    assert float(feasible.min()) > 0.2 and float(feasible.max()) < 0.8


def test_multi_risk_control_returns_an_empty_certificate_when_the_contract_is_impossible() -> None:
    # The two clauses cannot both hold: to miss under 10% you must alert on nearly everything.
    grid = np.linspace(0.0, 1.0, 21)
    feasible = multi_risk_valid_set(
        grid, grid.copy(), 1.0 - grid, 4000, 4000, alpha_miss=0.1, alpha_volume=0.1, delta=0.1
    )
    assert len(feasible) == 0


def test_multi_risk_control_is_stricter_than_either_constraint_alone() -> None:
    grid = np.linspace(0.0, 1.0, 21)
    miss, volume = grid.copy(), 1.0 - grid
    both = multi_risk_valid_set(grid, miss, volume, 4000, 4000, 0.6, 0.6, 0.1)
    miss_only = ltt_valid_set(grid, miss, 4000, 0.6, 0.1, method="bonferroni")
    assert set(both.tolist()) <= set(miss_only.tolist())


# --------------------------------------------------------------------------------------
# End-to-end: does each selector deliver the level it claims?
# --------------------------------------------------------------------------------------


def test_the_two_selectors_deliver_the_two_different_guarantees() -> None:
    """CRC controls the mean; LTT controls the tail. Both are checked by simulation.

    This is the report's headline as a test: an expectation bound is routinely exceeded by
    individual deployments, and a high-probability bound is not.
    """
    rng = np.random.default_rng(1)
    alpha, delta, n = 0.2, 0.1, 300
    grid = np.linspace(0.0, 1.0, 101)
    crc_realised, ltt_realised = [], []
    for _ in range(300):
        # Scores of "attacks": uniform, so the miss rate at threshold t is exactly t.
        cal = rng.random(n)
        evaluation = rng.random(n)
        risks = np.searchsorted(np.sort(cal), grid, side="left") / n
        crc = crc_threshold(grid, risks, n, alpha)
        valid = ltt_valid_set(grid, risks, n, alpha, delta)
        ltt = float(valid[-1]) if len(valid) else float(grid[0])
        crc_realised.append(float(np.mean(evaluation < crc)))
        ltt_realised.append(float(np.mean(evaluation < ltt)))

    assert float(np.mean(crc_realised)) <= alpha  # the expectation bound holds
    assert float(np.mean(np.array(crc_realised) > alpha)) > 2 * delta  # ... and is often exceeded
    assert float(np.mean(np.array(ltt_realised) > alpha)) <= delta  # the tail bound holds
