"""Conformal test martingale: the martingale/Ville property under the null, growth under drift.

The whole guarantee rests on two facts — the betting process is a (super)martingale with
M_0 = 1 under exchangeability (so Ville bounds the false-alarm rate), and it grows without
bound when the stream stops being exchangeable. Both are pinned here on synthetic p-value
streams, plus the conformal p-values' uniformity under exchangeability.
"""

from __future__ import annotations

import numpy as np

from netsentry.monitoring.exchangeability import (
    _mixture,
    detection_time,
    online_conformal_pvalues,
    power_martingale_mixture,
)


def test_false_alarm_rate_respects_ville_bound() -> None:
    # Under IID-uniform p-values (the exchangeable null) the mixture is a (super)martingale,
    # so P(ever cross 1/alpha) <= alpha. Estimate that rate over many independent streams.
    rng = np.random.default_rng(0)
    eps = _mixture(19)
    alpha = 0.05
    threshold = 1.0 / alpha
    n_streams, length = 400, 200
    crossings = 0
    for _ in range(n_streams):
        p = rng.random(length)
        path = power_martingale_mixture(p, eps)
        if detection_time(path, threshold) is not None:
            crossings += 1
    assert crossings / n_streams <= alpha + 0.02  # Ville, with finite-sample slack


def test_martingale_shrinks_on_neutral_pvalues() -> None:
    # p = 0.5 everywhere is the "no evidence" stream: betting against a fair coin loses.
    path = power_martingale_mixture(np.full(60, 0.5), _mixture(19))
    assert path[-1] < 1.0


def test_martingale_explodes_on_small_pvalues() -> None:
    # A run of tiny p-values is what drift produces; the martingale must cross fast.
    path = power_martingale_mixture(np.full(100, 0.01), _mixture(19))
    hit = detection_time(path, 100.0)
    assert hit is not None and hit < 100


def test_online_conformal_pvalues_uniform_under_exchangeability() -> None:
    # An IID stream is exchangeable, so the online conformal p-values are ~Uniform(0,1).
    rng = np.random.default_rng(3)
    p = online_conformal_pvalues(rng.standard_normal(3000), rng)
    assert 0.45 < float(np.mean(p)) < 0.55
    assert float(np.min(p)) >= 0.0 and float(np.max(p)) <= 1.0


def test_online_conformal_pvalues_flag_a_monotone_trend() -> None:
    # A strictly increasing nonconformity stream is maximally non-exchangeable: each new
    # point is the largest so far, so its p-value is tiny.
    rng = np.random.default_rng(4)
    p = online_conformal_pvalues(np.arange(500, dtype=float), rng)
    assert float(np.mean(p[-50:])) < 0.1


def test_detection_time_reports_first_crossing() -> None:
    path = np.array([1.0, 5.0, 20.0, 150.0, 3.0])
    assert detection_time(path, 100.0) == 4
    assert detection_time(path, 1000.0) is None
