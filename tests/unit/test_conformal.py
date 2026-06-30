"""Split-conformal prediction: the coverage guarantee and set construction."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.conformal import (
    class_conditional_thresholds,
    conformal_quantile,
    evaluate_conformal,
    prediction_sets,
)


def test_conformal_quantile_finite_sample_correction() -> None:
    scores = np.linspace(0, 1, 100)
    # With the (n+1) correction the (1-alpha) quantile is slightly conservative.
    q = conformal_quantile(scores, alpha=0.1)
    assert 0.89 <= q <= 0.92
    assert conformal_quantile(np.array([]), 0.1) == float("inf")


def test_prediction_sets_membership() -> None:
    p = np.array([0.05, 0.5, 0.95])
    in_benign, in_attack = prediction_sets(p, tau_benign=0.3, tau_attack=0.3)
    # p<=0.3 -> benign; (1-p)<=0.3 i.e. p>=0.7 -> attack.
    assert in_benign.tolist() == [True, False, False]
    assert in_attack.tolist() == [False, False, True]


def test_thresholds_are_class_conditional() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
    tau_b, tau_a = class_conditional_thresholds(p, y, alpha=0.1)
    assert 0.0 <= tau_b <= 1.0
    assert 0.0 <= tau_a <= 1.0


def test_coverage_guarantee_holds_on_fresh_data() -> None:
    # The core promise: empirical class-conditional coverage >= 1 - alpha on test.
    rng = np.random.default_rng(7)

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = (rng.uniform(size=n) < 0.4).astype(int)
        p = np.clip(rng.normal(np.where(y == 1, 0.7, 0.3), 0.15), 0, 1)
        return p, y

    p_cal, y_cal = sample(4000)
    p_test, y_test = sample(4000)
    alpha = 0.1
    rep = evaluate_conformal(p_cal, y_cal, p_test, y_test, alpha)
    # Allow a small finite-sample slack below the 1-alpha target.
    assert rep.coverage_benign >= 1 - alpha - 0.03
    assert rep.coverage_attack >= 1 - alpha - 0.03


def test_smaller_alpha_gives_higher_coverage() -> None:
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=4000) < 0.4).astype(int)
    p = np.clip(rng.normal(np.where(y == 1, 0.7, 0.3), 0.2), 0, 1)
    tight = evaluate_conformal(p, y, p, y, alpha=0.01)
    loose = evaluate_conformal(p, y, p, y, alpha=0.2)
    assert tight.coverage_attack >= loose.coverage_attack
    # Tighter coverage costs more abstention (larger/ambiguous sets).
    assert (tight.rate_ambiguous + tight.rate_empty) >= (
        loose.rate_ambiguous + loose.rate_empty
    ) - 1e-9
