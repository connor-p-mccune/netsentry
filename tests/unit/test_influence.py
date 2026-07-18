"""Influence functions: the gradient/Hessian math, the LOO prediction, and self-influence.

The report's central claim is that a closed-form inverse-Hessian estimate predicts what
actually happens when a training point is removed. These pin that on a tiny logistic problem
where the ground truth is a real retrain, plus a finite-difference check of the derivatives
and the self-influence outlier signal — none of which needs the dataset.
"""

from __future__ import annotations

import numpy as np

from netsentry.explain.influence import (
    _augment,
    _pm,
    _pointwise_grads,
    fit_logistic,
    hessian,
    influence_on_test,
    self_influence,
)


def _toy(n: int = 400, d: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    logits = x @ w + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return x, y


def test_gradient_matches_finite_difference() -> None:
    x, y = _toy(60, 3, seed=1)
    xa = _augment(x)
    theta = fit_logistic(x, y, l2=1.0, seed=1)
    ypm = _pm(y)
    grads = _pointwise_grads(theta, xa, ypm)
    # Finite-difference the per-example loss wrt theta and compare to the analytic gradient.
    eps = 1e-6

    def loss(t: np.ndarray, i: int) -> float:
        return float(np.log1p(np.exp(-ypm[i] * (xa[i] @ t))))

    for i in (0, 17, 41):
        num = np.zeros_like(theta)
        for k in range(len(theta)):
            step = np.zeros_like(theta)
            step[k] = eps
            num[k] = (loss(theta + step, i) - loss(theta - step, i)) / (2 * eps)
        assert np.allclose(grads[i], num, atol=1e-5)


def test_hessian_is_symmetric_positive_definite() -> None:
    x, y = _toy(200, 4, seed=2)
    xa = _augment(x)
    theta = fit_logistic(x, y, l2=1.0, seed=2)
    h = hessian(theta, xa, l2=1.0)
    assert np.allclose(h, h.T)
    assert np.all(np.linalg.eigvalsh(h) > 0)  # invertible, PD


def test_influence_predicts_actual_leave_one_out() -> None:
    # The killer test: on a small problem, the closed-form influence must correlate almost
    # perfectly with the true change in a test loss when each point is genuinely removed.
    x, y = _toy(300, 3, seed=3)
    xa = _augment(x)
    xtest, ytest = _toy(40, 3, seed=99)
    xta = _augment(xtest)
    ytpm = _pm(ytest)

    theta = fit_logistic(x, y, l2=1.0, seed=3)
    h_inv = np.linalg.inv(hessian(theta, xa, l2=1.0))
    g_train = _pointwise_grads(theta, xa, _pm(y))
    g_test = _pointwise_grads(theta, xta, ytpm).mean(axis=0)
    n = len(x)
    pred = influence_on_test(theta, h_inv, g_train, g_test, n)

    def test_loss(t: np.ndarray) -> float:
        return float(np.mean(np.log1p(np.exp(-ytpm * (xta @ t)))))

    base = test_loss(theta)
    rng = np.random.default_rng(4)
    sample = rng.choice(n, size=30, replace=False)
    true_delta = []
    for i in sample:
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        theta_i = fit_logistic(x[keep], y[keep], l2=1.0, seed=3)
        true_delta.append(test_loss(theta_i) - base)
    r = np.corrcoef(pred[sample], true_delta)[0, 1]
    assert r > 0.9  # the approximation tracks the real retrain


def test_self_influence_flags_a_planted_outlier() -> None:
    # A cluster with one clearly mislabelled point; its self-influence should stand out.
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal(-2, 0.3, (150, 2)), rng.normal(2, 0.3, (150, 2))])
    y = np.array([0] * 150 + [1] * 150)
    y[0] = 1  # a benign-region point mislabelled attack
    xa = _augment(x)
    theta = fit_logistic(x, y, l2=1.0, seed=5)
    h_inv = np.linalg.inv(hessian(theta, xa, l2=1.0))
    scores = self_influence(h_inv, _pointwise_grads(theta, xa, _pm(y)))
    # The flipped point sits in the top few by self-influence.
    assert 0 in np.argsort(-scores)[:10]


def test_helpful_point_has_positive_influence_on_its_own_class() -> None:
    # A test attack; a training attack near it should *help* (positive influence, removing it
    # raises the attack-loss), a training benign should *hurt* (negative).
    rng = np.random.default_rng(6)
    x = np.vstack([rng.normal(-2, 0.3, (100, 2)), rng.normal(2, 0.3, (100, 2))])
    y = np.array([0] * 100 + [1] * 100)
    xa = _augment(x)
    theta = fit_logistic(x, y, l2=1.0, seed=6)
    h_inv = np.linalg.inv(hessian(theta, xa, l2=1.0))
    g_train = _pointwise_grads(theta, xa, _pm(y))
    x_query = _augment(np.array([[2.0, 2.0]]))  # deep in the attack region
    g_test = _pointwise_grads(theta, x_query, _pm(np.array([1])))[0]
    infl = influence_on_test(theta, h_inv, g_train, g_test, len(x))
    # Attack training points (indices >= 100) should skew positive; benign ones negative.
    assert infl[100:].mean() > infl[:100].mean()
