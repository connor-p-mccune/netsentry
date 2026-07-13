"""Tests for the differential-privacy accountant and DP-SGD classifier.

The accountant is checked against the closed forms it must reproduce (the Gaussian
mechanism at ``q == 1``, RDP composition, subsampling amplification) and the
monotonicities any correct ε accountant obeys; the classifier is checked for
DP mechanics (clipping/noise), determinism, and that it still learns.
"""

from __future__ import annotations

import math

import numpy as np

from netsentry.robustness.dp import (
    DEFAULT_ORDERS,
    DPClassifier,
    compute_rdp,
    dp_sgd_epsilon,
    rdp_of_step,
    rdp_to_epsilon,
)


def test_rdp_q1_matches_gaussian_closed_form() -> None:
    # No subsampling: the Gaussian mechanism has RDP(alpha) = alpha / (2 sigma^2).
    sigma = 1.7
    for alpha in (2, 5, 16, 64):
        assert rdp_of_step(1.0, sigma, alpha) == alpha / (2.0 * sigma * sigma)


def test_rdp_zero_noise_is_infinite_and_zero_rate_is_free() -> None:
    assert rdp_of_step(0.5, 0.0, 8) == math.inf  # no noise => no privacy
    assert rdp_of_step(0.0, 2.0, 8) == 0.0  # nothing sampled => nothing spent


def test_subsampling_amplifies_privacy() -> None:
    # Subsampling reduces the per-step RDP relative to the full-batch mechanism.
    for alpha in (2, 8, 32):
        assert rdp_of_step(0.1, 2.0, alpha) < rdp_of_step(1.0, 2.0, alpha)


def test_rdp_composition_scales_linearly_in_steps() -> None:
    one = compute_rdp(0.05, 1.5, steps=1)
    ten = compute_rdp(0.05, 1.5, steps=10)
    for alpha in one:
        assert ten[alpha] == 10.0 * one[alpha]


def test_epsilon_decreases_with_more_noise() -> None:
    def eps(sigma: float) -> float:
        return dp_sgd_epsilon(
            sampling_rate=0.01, noise_multiplier=sigma, steps=1000, target_delta=1e-5
        )

    e_low, e_mid, e_high = eps(0.6), eps(1.2), eps(3.0)
    assert e_low > e_mid > e_high > 0.0
    assert math.isfinite(e_low)


def test_epsilon_increases_with_more_steps() -> None:
    def eps(steps: int) -> float:
        return dp_sgd_epsilon(
            sampling_rate=0.01, noise_multiplier=1.0, steps=steps, target_delta=1e-5
        )

    assert eps(200) < eps(1000) < eps(5000)


def test_epsilon_is_a_sound_finite_bound() -> None:
    eps = dp_sgd_epsilon(sampling_rate=0.01, noise_multiplier=1.0, steps=1000, target_delta=1e-5)
    assert 0.0 < eps < 100.0  # a moderate finite guarantee, not degenerate


def test_rdp_to_epsilon_selects_a_valid_order() -> None:
    rdp = compute_rdp(0.02, 1.1, steps=500)
    eps, order = rdp_to_epsilon(rdp, target_delta=1e-5)
    assert order in DEFAULT_ORDERS and order > 1
    assert math.isfinite(eps) and eps > 0.0


def _toy_separable(n: int = 800, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    logits = 1.5 * x[:, 0] - 1.2 * x[:, 1]
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return x, y


def test_non_private_classifier_learns_and_is_infinite_epsilon() -> None:
    x, y = _toy_separable()
    clf = DPClassifier(noise_multiplier=0.0, epochs=40, seed=1).fit(x, y)
    acc = float(np.mean(clf.predict(x) == y))
    assert acc > 0.8
    assert clf.epsilon(1e-5) == math.inf  # non-private reference

    proba = clf.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_private_classifier_still_beats_chance_and_spends_finite_budget() -> None:
    x, y = _toy_separable()
    clf = DPClassifier(noise_multiplier=1.0, l2_clip=1.0, epochs=60, seed=2).fit(x, y)
    acc = float(np.mean(clf.predict(x) == y))
    base = max(float(np.mean(y)), 1.0 - float(np.mean(y)))
    assert acc > base  # DP costs accuracy but the model still learns signal
    assert math.isfinite(clf.epsilon(1e-5)) and clf.epsilon(1e-5) > 0.0
    assert clf.private and clf.steps_ > 0 and 0.0 < clf.sampling_rate_ <= 1.0


def test_dp_fit_is_deterministic_given_the_seed() -> None:
    x, y = _toy_separable()
    a = DPClassifier(noise_multiplier=1.0, seed=7).fit(x, y).decision_scores(x)
    b = DPClassifier(noise_multiplier=1.0, seed=7).fit(x, y).decision_scores(x)
    assert np.array_equal(a, b)
    c = DPClassifier(noise_multiplier=1.0, seed=8).fit(x, y).decision_scores(x)
    assert not np.array_equal(a, c)  # a different seed draws different noise


def test_more_noise_increases_run_to_run_variance() -> None:
    # The mechanism at work: the injected Gaussian noise makes the fit itself a
    # random variable, so heavier noise widens the spread of scores across seeds.
    # (DP-SGD noise is unbiased, so on an easy problem accuracy can survive — the
    # real utility cost appears at the operating point, which the study measures.)
    x, y = _toy_separable(n=1200)

    def scores(sigma: float, seed: int) -> np.ndarray:
        return (
            DPClassifier(noise_multiplier=sigma, epochs=40, seed=seed).fit(x, y).decision_scores(x)
        )

    low = np.stack([scores(0.5, s) for s in range(5)])
    high = np.stack([scores(8.0, s) for s in range(5)])
    assert float(high.std(axis=0).mean()) > float(low.std(axis=0).mean())
