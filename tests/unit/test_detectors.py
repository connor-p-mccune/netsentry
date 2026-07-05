"""Statistical / online drift detectors, validated against planted shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.monitoring.detectors import (
    benjamini_hochberg,
    ddm,
    ks_feature_tests,
    page_hinkley,
)


def test_benjamini_hochberg_controls_multiplicity() -> None:
    # One tiny p-value among many nulls: BH should reject it and not the nulls.
    pvalues = np.array([0.001, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    significant, crit = benjamini_hochberg(pvalues, alpha=0.05)
    assert significant[0] and not significant[1:].any()
    assert crit >= 0.001

    # All-null p-values: nothing is declared significant.
    none_sig, _ = benjamini_hochberg(np.linspace(0.2, 0.95, 10), alpha=0.05)
    assert not none_sig.any()


def test_benjamini_hochberg_is_less_strict_than_bonferroni() -> None:
    # A borderline p-value BH accepts but per-comparison Bonferroni (alpha/m) rejects.
    pvalues = np.array([0.001, 0.012, 0.9, 0.9, 0.9])
    significant, _ = benjamini_hochberg(pvalues, alpha=0.05)
    assert significant[:2].all()  # both small p-values survive BH
    assert 0.05 / len(pvalues) < 0.012  # but would fail Bonferroni


def test_ks_flags_shifted_feature_only() -> None:
    rng = np.random.default_rng(0)
    n = 500
    reference = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(0, 1, n)})
    current = pd.DataFrame(
        {"stable": rng.normal(0, 1, n), "shifted": rng.normal(3, 1, n)}  # mean shift
    )
    results = {r.feature: r for r in ks_feature_tests(reference, current, ["stable", "shifted"])}
    assert results["shifted"].significant
    assert not results["stable"].significant
    assert results["shifted"].statistic > results["stable"].statistic


def test_ks_skips_absent_and_degenerate_columns() -> None:
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0], "constant": [np.nan, np.nan, np.nan]})
    current = pd.DataFrame({"a": [1.0, 2.0, 3.0], "constant": [np.nan, np.nan, np.nan]})
    results = ks_feature_tests(reference, current, ["a", "constant", "missing"])
    assert [r.feature for r in results] == ["a"]  # degenerate + missing dropped


def test_page_hinkley_detects_mean_increase_after_change_point() -> None:
    rng = np.random.default_rng(1)
    stream = np.concatenate([rng.normal(0.0, 0.05, 300), rng.normal(1.0, 0.05, 300)])
    idx = page_hinkley(stream, delta=0.005, lam=5.0)
    assert idx is not None
    assert 300 <= idx <= 360  # fires shortly after the shift, not before

    # A stationary stream never alarms.
    assert page_hinkley(rng.normal(0.0, 0.05, 600), delta=0.005, lam=5.0) is None


def test_ddm_warns_then_drifts_when_error_rate_climbs() -> None:
    # Evenly-spaced errors give a perfectly stable cumulative baseline (no random
    # clustering), so the detector's response to the *real* change is unambiguous.
    low = np.zeros(400)
    low[::20] = 1  # 5% baseline error rate
    high = np.zeros(300)
    high[::2] = 1  # 50% after the change point at index 400
    result = ddm(np.concatenate([low, high]), warn_level=2.0, drift_level=3.0, min_samples=50)
    assert result.drift_index is not None and result.drift_index >= 400  # only after the jump
    if result.warning_index is not None:
        assert result.warning_index <= result.drift_index  # warning precedes drift

    # A stable error rate raises no drift alarm.
    assert ddm(low, min_samples=50).drift_index is None
