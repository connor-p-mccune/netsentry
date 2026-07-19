"""Conformal alert selection: the p-value construction, BH step-up, and FDR control.

The report's guarantee is a property of the pure machinery, provable without any model: the
conformal p-values are super-uniform under the benign null, Benjamini-Hochberg selects the
right step-up set, and BH on the (dependent-but-PRDS) conformal p-values controls the realized
false-discovery proportion in expectation. These pin exactly that.
"""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.alert_fdr import (
    benjamini_hochberg,
    conformal_pvalues,
    power,
    realized_fdp,
    resample_to_prevalence,
)


def test_conformal_pvalue_matches_the_smoothed_rank_by_hand() -> None:
    cal = np.array([0.1, 0.2, 0.3, 0.4])  # n_cal = 4
    # s = 0.35: one cal value (0.4) >= it, so p = (1 + 1) / (4 + 1) = 0.4.
    # s = 0.5: zero cal values >= it, so p = (1 + 0) / 5 = 0.2 (most anomalous).
    # s = 0.05: all four >= it, so p = (1 + 4) / 5 = 1.0.
    p = conformal_pvalues(cal, np.array([0.35, 0.5, 0.05]))
    assert np.allclose(p, [0.4, 0.2, 1.0])


def test_conformal_pvalues_are_super_uniform_under_the_null() -> None:
    # Test flows drawn from the SAME distribution as calibration: p-values ~ Uniform(0, 1),
    # so P(p <= t) <= t must hold (super-uniformity), the property BH relies on.
    rng = np.random.default_rng(0)
    cal = rng.normal(size=5000)
    test = rng.normal(size=5000)  # exchangeable with cal (all benign)
    p = conformal_pvalues(cal, test)
    for t in (0.05, 0.1, 0.2, 0.5):
        assert np.mean(p <= t) <= t + 0.02  # super-uniform within sampling slack


def test_conformal_pvalue_is_small_for_clear_anomalies() -> None:
    cal = np.random.default_rng(1).normal(size=1000)
    # A far-out-of-support high score is the most anomalous -> the minimal p-value.
    p = conformal_pvalues(cal, np.array([100.0]))
    assert p[0] == 1.0 / (1000 + 1)


def test_benjamini_hochberg_selects_a_strict_subset() -> None:
    # p = [.005, .01, .5, .5], q = .05, m = 4. crit = [.0125, .025, .0375, .05].
    # Sorted .005, .01 pass; .5, .5 fail. Step-up cut at .01 -> the two small ones only.
    p = np.array([0.005, 0.01, 0.5, 0.5])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False]


def test_benjamini_hochberg_step_up_rescues_a_middle_pvalue() -> None:
    # p sorted = [.001, .03, .035, .9], q = .05, m = 4. crit = [.0125, .025, .0375, .05].
    # .03 fails its own crit (.03 > .025) but the later .035 <= .0375 passes, so the step-up
    # threshold is .035 and .03 is rescued into the selection — the defining BH behaviour.
    p = np.array([0.001, 0.03, 0.035, 0.9])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, True, False]


def test_benjamini_hochberg_rejects_nothing_when_no_pvalue_passes() -> None:
    p = np.array([0.6, 0.7, 0.8, 0.9])
    assert not benjamini_hochberg(p, 0.05).any()


def test_benjamini_hochberg_all_tiny_rejects_everything() -> None:
    p = np.array([1e-6, 1e-6, 1e-6])
    assert benjamini_hochberg(p, 0.05).all()


def test_benjamini_hochberg_controls_fdr_on_a_mixture() -> None:
    # 200 nulls (Uniform) + 50 strong signals (near 0). Averaged over many draws the realized
    # FDP must sit at or under q — the guarantee, checked empirically on synthetic p-values.
    rng = np.random.default_rng(2)
    q = 0.1
    fdps = []
    for _ in range(300):
        nulls = rng.uniform(size=200)
        signals = rng.uniform(0, 1e-3, size=50)
        p = np.concatenate([nulls, signals])
        is_signal = np.concatenate([np.zeros(200, bool), np.ones(50, bool)])
        selected = benjamini_hochberg(p, q)
        fdps.append(realized_fdp(selected, is_signal))
    assert np.mean(fdps) <= q + 0.02


def test_realized_fdp_and_power_are_the_confusion_fractions() -> None:
    selected = np.array([True, True, True, False])
    is_attack = np.array([True, False, False, True])
    # 3 selected, 2 of them benign -> FDP 2/3; 1 of 2 attacks caught -> power 0.5.
    assert np.isclose(realized_fdp(selected, is_attack), 2.0 / 3.0)
    assert np.isclose(power(selected, is_attack), 0.5)


def test_realized_fdp_is_zero_on_an_empty_selection() -> None:
    assert realized_fdp(np.zeros(5, bool), np.ones(5, bool)) == 0.0
    assert power(np.zeros(5, bool), np.zeros(5, bool)) == 0.0


def test_resample_to_prevalence_hits_the_target_prior() -> None:
    is_attack = np.array([True] * 100 + [False] * 900)
    rng = np.random.default_rng(3)
    idx = resample_to_prevalence(is_attack, 0.2, rng, size=2000)
    assert len(idx) == 2000
    assert np.isclose(np.mean(is_attack[idx]), 0.2, atol=1e-9)  # exact by construction
