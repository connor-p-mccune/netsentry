"""Randomized-smoothing certification math: the Clopper-Pearson lower bound and Cohen's
per-flow certified radius, plus the certified-accuracy bookkeeping."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from netsentry.robustness.certify import (
    ABSTAIN,
    CertRow,
    SigmaResult,
    certified_radius,
    clopper_pearson_lower,
)


def test_clopper_pearson_lower_is_conservative_and_monotone() -> None:
    # More successes -> a higher (but still < phat) lower bound.
    low = clopper_pearson_lower(60, 100, alpha=0.05)
    high = clopper_pearson_lower(90, 100, alpha=0.05)
    assert 0.0 < low < 0.60  # strictly below the point estimate
    assert low < high < 0.90
    # Edge cases are safe.
    assert clopper_pearson_lower(0, 100, 0.05) == 0.0
    assert clopper_pearson_lower(100, 100, 0.05) > 0.9


def test_certify_abstains_on_a_split_vote() -> None:
    # A near-50/50 vote cannot clear the p_A > 1/2 confidence bar.
    predicted, radius = certified_radius(attack_votes=501, n=1000, sigma=0.5, alpha=0.001)
    assert predicted == ABSTAIN and radius == 0.0


def test_certify_returns_the_majority_class_and_scaled_radius() -> None:
    # A lopsided attack vote certifies class 1 with a positive radius.
    predicted, radius = certified_radius(attack_votes=990, n=1000, sigma=0.5, alpha=0.001)
    assert predicted == 1 and radius > 0.0
    # The radius equals sigma * Phi^-1(p_A) for the computed lower bound.
    p_lower = clopper_pearson_lower(990, 1000, 0.001)
    assert np.isclose(radius, 0.5 * norm.ppf(p_lower))

    # A lopsided benign vote certifies class 0.
    predicted0, radius0 = certified_radius(attack_votes=10, n=1000, sigma=0.5, alpha=0.001)
    assert predicted0 == 0 and radius0 > 0.0


def test_radius_scales_linearly_with_sigma() -> None:
    _, r1 = certified_radius(950, 1000, sigma=0.5, alpha=0.001)
    _, r2 = certified_radius(950, 1000, sigma=1.0, alpha=0.001)
    assert np.isclose(r2, 2.0 * r1)


def test_sigma_result_accounting() -> None:
    rows = [
        CertRow(true_label=1, predicted=1, radius=1.2),  # correct, robust to 1.2
        CertRow(true_label=1, predicted=ABSTAIN, radius=0.0),  # abstained
        CertRow(true_label=0, predicted=1, radius=0.8),  # wrong
        CertRow(true_label=0, predicted=0, radius=0.4),  # correct, robust to 0.4
    ]
    result = SigmaResult(sigma=0.5, rows=rows, radii_grid=[0.0, 0.5, 1.0])
    assert result.clean_accuracy == 0.5  # 2 of 4 correct
    assert result.abstain_rate == 0.25
    assert result.certified_accuracy(0.0) == 0.5  # both correct rows have radius >= 0
    assert result.certified_accuracy(0.5) == 0.25  # only the radius-1.2 row survives
    assert result.certified_accuracy(1.0) == 0.25
    assert np.isclose(result.median_radius, np.median([1.2, 0.8, 0.4]))
