"""Covariate shift: the density-ratio cross-fit, the effective sample size, and localization.

The report's estimator is a domain classifier turned into a density ratio, and its diagnostics
are the classifier-two-sample-test AUC and the Kish effective sample size. The C2ST recovering
a *known* shift (and finding none where there is none), the ESS arithmetic, and the per-group
localization are all provable on constructed data — pinned here.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from netsentry.monitoring.covariate_shift import (
    crossfit_domain_ratio,
    effective_sample_size,
    per_group_mean_weight,
)


def _logistic_builder() -> LogisticRegression:
    return LogisticRegression(max_iter=500)


def test_c2st_auc_is_near_half_when_train_and_test_share_a_distribution() -> None:
    # No covariate shift: train and test drawn from the same Gaussian -> indistinguishable.
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(800, 4))
    x_test = rng.normal(size=(800, 4))
    _, auc = crossfit_domain_ratio(x_train, x_test, _logistic_builder, seed=0, n_folds=4, clip=20)
    assert 0.44 <= auc <= 0.56  # a coin flip, within sampling noise


def test_c2st_auc_detects_a_clear_covariate_shift() -> None:
    # Test flows shifted +3 in every feature: trivially separable -> high AUC, big weights.
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(800, 4))
    x_test = rng.normal(loc=3.0, size=(800, 4))
    weights, auc = crossfit_domain_ratio(
        x_train, x_test, _logistic_builder, seed=0, n_folds=4, clip=50
    )
    assert auc > 0.9  # the domain classifier easily tells them apart
    # Train rows nearest the test cloud (high values) must carry the largest weights.
    high_side = x_train.mean(axis=1) > 1.0
    assert weights[high_side].mean() > weights[~high_side].mean()


def test_density_ratio_weights_are_nonnegative_and_clipped() -> None:
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(500, 3))
    x_test = rng.normal(loc=2.0, size=(500, 3))
    weights, _ = crossfit_domain_ratio(
        x_train, x_test, _logistic_builder, seed=0, n_folds=3, clip=5
    )
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 5.0)
    assert len(weights) == len(x_train)


def test_effective_sample_size_is_full_for_uniform_weights() -> None:
    # All-equal weights: ESS = n (no variance penalty).
    assert np.isclose(effective_sample_size(np.ones(100)), 100.0)


def test_effective_sample_size_collapses_for_a_single_spike() -> None:
    # One giant weight, the rest near zero: ESS -> ~1, the concentration the report warns about.
    w = np.concatenate([[1000.0], np.full(999, 1e-6)])
    assert effective_sample_size(w) < 1.01


def test_effective_sample_size_matches_the_kish_formula() -> None:
    w = np.array([1.0, 2.0, 3.0, 4.0])
    expected = w.sum() ** 2 / np.sum(w**2)  # 100 / 30
    assert np.isclose(effective_sample_size(w), expected)


def test_per_group_mean_weight_averages_within_group_and_keeps_order() -> None:
    weights = np.array([1.0, 3.0, 10.0, 20.0])
    groups = np.array(["Mon", "Mon", "Wed", "Wed"])
    result = per_group_mean_weight(weights, groups)
    assert result == [("Mon", 2.0), ("Wed", 15.0)]
