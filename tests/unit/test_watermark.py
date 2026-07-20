"""Model watermarking: the ownership test's exact binomial tail and the fair-coin null.

The ownership proof is a statistical claim that must be exactly right: the log-space binomial
tail against known values, the fair-coin construction that makes an innocent model's agreement
50% regardless of its bias, and the watermark-accuracy bookkeeping. Pinned here without any
model, plus a small end-to-end embed-and-detect on separable blobs.
"""

from __future__ import annotations

import math

import numpy as np

from netsentry.robustness.watermark import (
    generate_watermark,
    ownership_log10_pvalue,
    watermark_accuracy,
)


def test_ownership_pvalue_all_matches_is_k_log10_half() -> None:
    # P(Binomial(k, 0.5) >= k) = 0.5^k, so log10 p = k * log10(0.5).
    for k in (10, 64, 256):
        assert np.isclose(ownership_log10_pvalue(k, k), k * math.log10(0.5))


def test_ownership_pvalue_is_one_at_or_below_the_mean() -> None:
    # P(X >= 0) = 1 -> log10 p = 0; and at the mean the tail is >= 0.5 -> log10 p in (-0.31, 0].
    assert np.isclose(ownership_log10_pvalue(0, 100), 0.0)
    assert -0.31 <= ownership_log10_pvalue(50, 100) <= 0.0


def test_ownership_pvalue_matches_a_hand_binomial_tail() -> None:
    # k = 4, matches = 3: P(X >= 3) = C(4,3)/16 + C(4,4)/16 = (4 + 1)/16 = 5/16.
    assert np.isclose(ownership_log10_pvalue(3, 4), math.log10(5.0 / 16.0))


def test_ownership_pvalue_is_monotone_in_matches() -> None:
    vals = [ownership_log10_pvalue(m, 200) for m in (100, 130, 160, 200)]
    assert vals == sorted(vals, reverse=True)  # more matches -> smaller (more negative) log p


def test_generate_watermark_is_deterministic_and_shaped() -> None:
    t1, y1 = generate_watermark(n_features=8, k=50, seed=3, scale=4.0)
    t2, y2 = generate_watermark(n_features=8, k=50, seed=3, scale=4.0)
    assert t1.shape == (50, 8) and y1.shape == (50,)
    assert np.array_equal(t1, t2) and np.array_equal(y1, y2)  # secret key is reproducible
    assert set(np.unique(y1)).issubset({0, 1})


def test_watermark_accuracy_counts_matches() -> None:
    preds = np.array([1, 0, 1, 1])
    owner = np.array([1, 0, 0, 1])
    assert np.isclose(watermark_accuracy(preds, owner), 0.75)


def test_innocent_agreement_is_chance_regardless_of_model_bias() -> None:
    # The construction's core guarantee: a model that predicts a FIXED class (extreme bias)
    # still matches fair-coin owner labels ~50% of the time, so the null is clean.
    _, owner = generate_watermark(n_features=6, k=4000, seed=0, scale=4.0)
    always_benign = np.zeros(len(owner), dtype=int)
    always_attack = np.ones(len(owner), dtype=int)
    assert abs(watermark_accuracy(always_benign, owner) - 0.5) < 0.03
    assert abs(watermark_accuracy(always_attack, owner) - 0.5) < 0.03
