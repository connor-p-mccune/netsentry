"""Density detectors, and the decomposition that says what an anomaly score is measuring.

Each detector is checked against a property it must have to be the thing it claims to be -- a
Mahalanobis distance has to grow with distance from the mean, a reconstruction error has to be
small on the subspace it was fitted to -- because a detector that returns plausible numbers and
ranks nothing is the failure mode that survives every smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.models.density import (
    GaussianMixtureDetector,
    KernelDensityDetector,
    MahalanobisDetector,
    NormDetector,
    PCAReconstructionDetector,
    complexity_proxy,
    rank_residual,
)


@pytest.fixture
def benign() -> np.ndarray:
    """Benign traffic as a correlated Gaussian blob: the distribution every arm must learn."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(600, 3))
    return np.column_stack([base[:, 0], base[:, 0] * 0.8 + base[:, 1] * 0.2, base[:, 2]])


# --------------------------------------------------------------------------------------
# Mahalanobis.
# --------------------------------------------------------------------------------------


def test_mahalanobis_scores_grow_with_distance_from_the_benign_mean(benign: np.ndarray) -> None:
    detector = MahalanobisDetector().fit(benign)
    near = detector.score(np.zeros((1, 3)))
    far = detector.score(np.full((1, 3), 8.0))
    assert far[0] > near[0]


def test_mahalanobis_is_not_fooled_by_the_correlated_direction(benign: np.ndarray) -> None:
    """The point of a Mahalanobis distance: equal Euclidean distance, unequal unlikelihood.

    Feature 1 is nearly a copy of feature 0 in this fixture, so moving *along* that ridge is
    ordinary and moving across it is not. A detector that cannot tell them apart is measuring
    a norm.
    """
    detector = MahalanobisDetector(ridge=1e-6).fit(benign)
    along = detector.score(np.array([[3.0, 2.4, 0.0]]))  # follows the correlation
    across = detector.score(np.array([[3.0, -2.4, 0.0]]))  # violates it
    assert across[0] > along[0]


def test_a_norm_detector_cannot_tell_those_two_apart(benign: np.ndarray) -> None:
    # The same two points, judged by the control: identical Euclidean length, identical score.
    detector = NormDetector().fit(benign)
    along = detector.score(np.array([[3.0, 2.4, 0.0]]))
    across = detector.score(np.array([[3.0, -2.4, 0.0]]))
    assert along[0] == pytest.approx(across[0])


def test_the_ridge_keeps_a_singular_covariance_invertible() -> None:
    # An exactly duplicated column makes the empirical covariance singular, which is the normal
    # case for flow statistics rather than a corner case.
    rng = np.random.default_rng(1)
    column = rng.normal(size=(200, 1))
    x = np.hstack([column, column, rng.normal(size=(200, 1))])
    scores = MahalanobisDetector().fit(x).score(x)
    assert np.all(np.isfinite(scores))


# --------------------------------------------------------------------------------------
# Reconstruction, and its linear form.
# --------------------------------------------------------------------------------------


def test_pca_reconstruction_error_is_small_on_the_subspace_it_learned() -> None:
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(400, 1))
    on_plane = np.hstack([latent, 2 * latent, -latent])  # a one-dimensional subspace
    detector = PCAReconstructionDetector(n_components=1, seed=0).fit(on_plane)
    off_plane = np.array([[0.0, 0.0, 5.0]])
    assert detector.score(on_plane[:5]).max() < detector.score(off_plane)[0]


def test_pca_reconstruction_error_is_never_negative() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 4))
    scores = PCAReconstructionDetector(n_components=2, seed=0).fit(x).score(x)
    assert np.all(scores >= 0.0)


# --------------------------------------------------------------------------------------
# Densities.
# --------------------------------------------------------------------------------------


def test_the_mixture_finds_two_modes_a_single_gaussian_would_average(benign: np.ndarray) -> None:
    """The mixture's reason to exist: a benign distribution with more than one centre.

    A single Gaussian fitted to two clusters puts its mean in the empty space between them and
    calls that region *typical*, which is exactly backwards.
    """
    rng = np.random.default_rng(4)
    left = rng.normal(loc=-5.0, scale=0.3, size=(300, 2))
    right = rng.normal(loc=5.0, scale=0.3, size=(300, 2))
    bimodal = np.vstack([left, right])
    middle = np.zeros((1, 2))

    mixture = GaussianMixtureDetector(n_components=2, seed=0).fit(bimodal)
    gaussian = MahalanobisDetector().fit(bimodal)

    assert mixture.score(middle)[0] > mixture.score(left[:5]).max()
    assert gaussian.score(middle)[0] < gaussian.score(left[:5]).min()


def test_the_kernel_density_estimate_ranks_a_far_point_as_less_likely(benign: np.ndarray) -> None:
    detector = KernelDensityDetector(n_samples=200, seed=0).fit(benign)
    assert detector.score(np.full((1, 3), 10.0))[0] > detector.score(benign[:5]).max()


def test_the_kernel_density_estimate_subsamples_to_its_budget(benign: np.ndarray) -> None:
    detector = KernelDensityDetector(n_samples=50, seed=0).fit(benign)
    assert detector.model_ is not None
    assert len(detector.model_.tree_.data) == 50


# --------------------------------------------------------------------------------------
# The control, and the decomposition.
# --------------------------------------------------------------------------------------


def test_the_norm_detector_learns_nothing_from_fitting(benign: np.ndarray) -> None:
    """Fitting is a no-op by design: scores must be identical before and after."""
    detector = NormDetector()
    before = detector.score(benign[:10])
    detector.fit(benign)
    np.testing.assert_array_equal(before, detector.score(benign[:10]))


def test_the_norm_detector_is_the_complexity_proxy(benign: np.ndarray) -> None:
    np.testing.assert_allclose(NormDetector().score(benign), complexity_proxy(benign))


def test_residualising_a_score_against_itself_leaves_nothing() -> None:
    # The identity that makes the decomposition interpretable: a detector that *is* the proxy
    # must have no ranking left once the proxy is removed.
    rng = np.random.default_rng(5)
    proxy = rng.normal(size=500)
    residual = rank_residual(proxy, proxy)
    assert np.abs(residual).max() < 1e-6


def test_residualising_an_independent_score_leaves_it_almost_intact() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(6)
    proxy = rng.normal(size=2000)
    scores = rng.normal(size=2000)
    residual = rank_residual(scores, proxy)
    assert abs(spearmanr(residual, scores).statistic) > 0.95


def test_the_residual_is_monotone_invariant() -> None:
    """A monotone transform of a score is the same detector, so the residual must agree."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(7)
    proxy = rng.normal(size=800)
    scores = rng.normal(size=800)
    plain = rank_residual(scores, proxy)
    squashed = rank_residual(1.0 / (1.0 + np.exp(-scores)), proxy)
    assert spearmanr(plain, squashed).statistic == pytest.approx(1.0)
