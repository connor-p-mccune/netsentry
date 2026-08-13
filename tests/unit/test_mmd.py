"""The kernel two-sample test, and the blindness it exists to remove.

The high-value test here is the last one: a change that leaves every marginal exactly intact.
It is what separates a joint test from the per-feature monitors already deployed, and the
assertion is on *bit-identical* KS statistics rather than on "similar" ones, because the
blindness is an algebraic property of the fault and not a matter of sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.monitoring.detectors import ks_feature_tests
from netsentry.monitoring.mmd import (
    dependence_fault,
    linear_time_mmd,
    marginal_shift,
    median_bandwidth,
    mmd2_from_kernel,
    mmd2_unbiased,
    mmd_permutation_test,
    rbf_kernel,
    squared_distances,
)


def _correlated(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Two standard-normal features with a fixed correlation, plus one independent."""
    a = rng.standard_normal(n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    c = rng.standard_normal(n)
    return np.column_stack([a, b, c])


def test_squared_distances_are_symmetric_and_zero_on_the_diagonal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 3))
    d2 = squared_distances(x, x)
    assert np.allclose(d2, d2.T)
    assert np.allclose(np.diag(d2), 0.0)
    assert (d2 >= 0).all()


def test_rbf_kernel_is_one_on_the_diagonal_and_decays() -> None:
    x = np.array([[0.0], [3.0]])
    kernel = rbf_kernel(x, x, gamma=0.5)
    assert np.allclose(np.diag(kernel), 1.0)
    assert kernel[0, 1] < 0.05


def test_median_bandwidth_tracks_the_data_scale() -> None:
    # gamma = 1 / median squared distance, so scaling the data by 10 must shrink gamma 100-fold.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 4))
    small = median_bandwidth(x, x, rng=np.random.default_rng(2))
    large = median_bandwidth(10 * x, 10 * x, rng=np.random.default_rng(2))
    assert small > 0 and large > 0
    assert np.isclose(small / large, 100.0, rtol=0.05)


def test_the_unbiased_estimator_is_centred_on_zero_under_the_null() -> None:
    # Unbiasedness is the whole point of dropping the diagonal: the biased V-statistic is
    # strictly positive even when the two samples come from the same distribution.
    rng = np.random.default_rng(3)
    values = [
        mmd2_unbiased(rng.standard_normal((80, 3)), rng.standard_normal((80, 3)), gamma=0.5)
        for _ in range(40)
    ]
    assert abs(float(np.mean(values))) < 0.01
    assert min(values) < 0  # an unbiased estimator of zero must go negative sometimes


def test_batched_permutations_match_one_split_at_a_time() -> None:
    # The permutation null is computed as a single matrix product; it must agree exactly with
    # evaluating each split on its own, or the p-value is measuring something else.
    rng = np.random.default_rng(4)
    pooled = rng.standard_normal((30, 2))
    kernel = rbf_kernel(pooled, pooled, gamma=0.7)
    masks = np.column_stack([rng.permutation(np.r_[np.ones(14), np.zeros(16)]) for _ in range(5)])
    batched = mmd2_from_kernel(kernel, masks)
    one_at_a_time = [float(mmd2_from_kernel(kernel, masks[:, i])) for i in range(masks.shape[1])]
    assert np.allclose(batched, one_at_a_time)


def test_permutation_p_value_is_never_zero() -> None:
    # (1 + #{null >= observed}) / (1 + B): reporting p = 0 from a finite permutation set is a
    # claim the test cannot make.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((60, 2))
    y = x + 50.0  # about as separated as two samples get
    test = mmd_permutation_test(x, y, rng=rng, permutations=20)
    assert test.p_value == 1.0 / 21.0


def test_the_test_holds_its_level_on_stationary_data() -> None:
    rng = np.random.default_rng(6)
    fired = 0
    trials = 40
    for _ in range(trials):
        x = rng.standard_normal((60, 3))
        y = rng.standard_normal((60, 3))
        fired += int(mmd_permutation_test(x, y, rng=rng, permutations=99).rejects(0.05))
    assert fired <= 4  # 5% of 40 is 2; allow the binomial slack


def test_the_test_detects_a_mean_shift() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((150, 3))
    y = rng.standard_normal((150, 3)) + 0.8
    assert mmd_permutation_test(x, y, rng=rng, permutations=199).rejects(0.01)


def test_linear_estimator_agrees_on_the_direction_and_is_cheap() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal((400, 3))
    same = linear_time_mmd(x, rng.standard_normal((400, 3)), gamma=0.5, rng=rng)
    shifted = linear_time_mmd(x, rng.standard_normal((400, 3)) + 1.0, gamma=0.5, rng=rng)
    assert same.p_value > 0.05
    assert shifted.p_value < 0.01
    assert shifted.statistic > same.statistic


def test_marginal_shift_moves_only_the_named_columns() -> None:
    rng = np.random.default_rng(9)
    window = rng.standard_normal((100, 4))
    out = marginal_shift(window, np.array([1, 3]), sigmas=2.0, rng=rng)
    assert np.allclose(out[:, [0, 2]], window[:, [0, 2]])
    assert out[:, 1].mean() > window[:, 1].mean() + 1.0


def test_dependence_fault_preserves_every_marginal_exactly() -> None:
    rng = np.random.default_rng(10)
    window = _correlated(200, 0.9, rng)
    out = dependence_fault(window, np.array([1]), rng=rng)
    # Same multiset of values in every column -- the fault only changes which row holds which.
    for j in range(window.shape[1]):
        assert np.allclose(np.sort(out[:, j]), np.sort(window[:, j]))
    assert abs(np.corrcoef(out[:, 0], out[:, 1])[0, 1]) < 0.3  # the correlation is gone


def test_only_the_joint_test_sees_a_dependence_only_change() -> None:
    """The report's central claim, asserted rather than described.

    Reference and current are drawn from the same correlated distribution; the current window is
    then scrambled so that one feature's values are re-paired with other rows. Every per-feature
    statistic is *mathematically* unchanged -- the KS statistics come back bit-identical -- while
    the joint distribution has been destroyed, and only the kernel test can say so.
    """
    rng = np.random.default_rng(11)
    reference = _correlated(400, 0.95, rng)
    current = _correlated(400, 0.95, rng)
    scrambled = dependence_fault(current, np.array([1]), rng=rng)
    names = ["a", "b", "c"]

    def ks_statistics(frame: np.ndarray) -> list[float]:
        tests = ks_feature_tests(
            pd.DataFrame(reference, columns=names), pd.DataFrame(frame, columns=names), names
        )
        return sorted(t.statistic for t in tests)

    assert ks_statistics(scrambled) == ks_statistics(current)  # blind, exactly
    assert mmd_permutation_test(reference, current, rng=rng, permutations=199).p_value > 0.05
    assert mmd_permutation_test(reference, scrambled, rng=rng, permutations=199).p_value < 0.05


def test_the_test_is_deterministic_given_a_seed() -> None:
    x = np.random.default_rng(12).standard_normal((80, 3))
    y = np.random.default_rng(13).standard_normal((80, 3)) + 0.4
    first = mmd_permutation_test(x, y, rng=np.random.default_rng(14), permutations=49)
    second = mmd_permutation_test(x, y, rng=np.random.default_rng(14), permutations=49)
    assert (first.statistic, first.p_value) == (second.statistic, second.p_value)
