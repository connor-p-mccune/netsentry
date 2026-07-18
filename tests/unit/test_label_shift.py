"""Label-shift estimators: BBSE, MLLS/EM, correction, and the resampling harness.

These pin the two properties the report leans on — the estimators recover a *known* planted
target prior from unlabelled data, and correction is a monotone (rank-preserving) reweighting
— plus the closed-form identities and degenerate guards that keep the study honest.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.label_shift import (
    bbse_weights,
    correct_posteriors,
    joint_confusion,
    mlls_em_prior,
    prior_from_weights,
    resample_to_prior,
)


def test_joint_confusion_sums_to_one_and_recovers_a_perfect_predictor() -> None:
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = y_true.copy()  # perfect predictor
    c = joint_confusion(y_true, y_pred, n_classes=2)
    assert np.isclose(c.sum(), 1.0)
    # Perfect predictor: mass only on the diagonal, columns = true-class priors.
    assert np.isclose(c[0, 1], 0.0) and np.isclose(c[1, 0], 0.0)
    assert np.isclose(c.sum(axis=0)[1], 0.4)  # P(true = attack) = 2/5


def _planted(prior: float, tpr: float, fpr: float, n: int, rng: np.random.Generator):
    """A synthetic (y, hard_pred, soft_posterior) triple with known rates and prior."""
    y = (rng.random(n) < prior).astype(int)
    p_attack = np.where(y == 1, tpr, fpr)  # P(pred = attack | y)
    pred = (rng.random(n) < p_attack).astype(int)
    # Soft posterior consistent-ish with the label, for the EM estimator.
    eta1 = np.where(y == 1, 0.85, 0.15) + rng.normal(0, 0.05, n)
    eta1 = np.clip(eta1, 0.01, 0.99)
    posteriors = np.column_stack([1 - eta1, eta1])
    return y, pred, posteriors


def test_bbse_recovers_a_planted_target_prior() -> None:
    rng = np.random.default_rng(0)
    tpr, fpr = 0.8, 0.1
    # Source confusion matrix estimated at the SOURCE prior.
    ys, preds, _ = _planted(0.3, tpr, fpr, 40000, rng)
    confusion = joint_confusion(ys, preds)
    source_prior = confusion.sum(axis=0)
    # A target with a very different prior; same class-conditional rates (label shift).
    for q_true in (0.05, 0.5, 0.8):
        _, pred_t, _ = _planted(q_true, tpr, fpr, 40000, rng)
        mu = np.array([np.mean(pred_t == 0), np.mean(pred_t == 1)])
        w = bbse_weights(confusion, mu)
        q_hat = prior_from_weights(w, source_prior)
        assert abs(q_hat[1] - q_true) < 0.03, f"q_true={q_true}"


def test_bbse_reduces_to_no_shift_when_target_equals_source() -> None:
    rng = np.random.default_rng(1)
    ys, preds, _ = _planted(0.3, 0.8, 0.1, 40000, rng)
    confusion = joint_confusion(ys, preds)
    source_prior = confusion.sum(axis=0)
    mu = np.array([np.mean(preds == 0), np.mean(preds == 1)])
    w = bbse_weights(confusion, mu)
    # No shift => weights ~1 and recovered prior ~ source prior.
    assert np.allclose(w, 1.0, atol=0.05)
    assert abs(prior_from_weights(w, source_prior)[1] - source_prior[1]) < 0.02


def test_mlls_em_recovers_a_planted_prior_under_calibrated_posteriors() -> None:
    # Proper MLLS setup: two Gaussians fix p(x|y); the model outputs the SOURCE-prior
    # posterior eta(x); target data is drawn at a different prior q. MLLS must recover q.
    rng = np.random.default_rng(2)
    p_source = 0.3
    source_prior = np.array([1 - p_source, p_source])
    for q_true in (0.1, 0.5, 0.85):
        n = 40000
        y = (rng.random(n) < q_true).astype(int)
        s = rng.normal(np.where(y == 1, 2.0, 0.0), 1.0)  # f0 = N(0,1), f1 = N(2,1)
        lr = np.exp(2.0 * s - 2.0)  # f1(s) / f0(s)
        eta1 = (p_source * lr) / (p_source * lr + (1 - p_source))  # source-calibrated posterior
        posteriors = np.column_stack([1 - eta1, eta1])
        q_hat, n_iter = mlls_em_prior(posteriors, source_prior, max_iter=1000, tol=1e-9)
        assert abs(q_hat[1] - q_true) < 0.03, f"q_true={q_true}"
        assert n_iter >= 1


def test_correct_posteriors_is_monotone_and_renormalised() -> None:
    # Reweighting by class importance must preserve the attack-score ranking (so PR-AUC is
    # unchanged) and keep each row a valid distribution.
    rng = np.random.default_rng(3)
    eta1 = np.sort(rng.random(50))
    posteriors = np.column_stack([1 - eta1, eta1])
    corrected = correct_posteriors(posteriors, np.array([0.5, 3.0]))
    assert np.allclose(corrected.sum(axis=1), 1.0)
    # Monotone in the original attack score => same ordering.
    assert np.all(np.diff(corrected[:, 1]) >= -1e-12)


def test_correct_posteriors_with_unit_weights_is_identity() -> None:
    rng = np.random.default_rng(4)
    eta1 = rng.random(20)
    posteriors = np.column_stack([1 - eta1, eta1])
    assert np.allclose(correct_posteriors(posteriors, np.array([1.0, 1.0])), posteriors)


def test_bbse_weights_are_nonnegative_on_a_singular_matrix() -> None:
    # A predictor no better than random gives a rank-deficient confusion matrix; the solver
    # must fall back gracefully and never return negative importance weights.
    confusion = np.array([[0.25, 0.25], [0.25, 0.25]])
    w = bbse_weights(confusion, np.array([0.5, 0.5]))
    assert (w >= 0).all()


def test_resample_hits_the_target_prior_exactly() -> None:
    rng = np.random.default_rng(5)
    y = np.array([1] * 300 + [0] * 700)
    idx = resample_to_prior(y, target_prior=0.2, size=1000, rng=rng)
    assert len(idx) == 1000
    assert np.isclose(y[idx].mean(), 0.2, atol=1e-9)  # exact by construction
