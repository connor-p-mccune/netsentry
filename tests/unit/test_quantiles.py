"""Streaming quantile estimators, checked against exact quantiles rather than each other.

Every estimator here has the same failure mode: it returns a plausible number. So each one is
graded against `numpy.quantile` on the same data, and the algorithm-specific invariants that
would otherwise fail silently — P-squared's marker ordering, the t-digest's tail resolution,
the histogram's interpolation — get their own tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.monitoring.quantiles import (
    HistogramQuantile,
    P2Quantile,
    ReservoirQuantile,
    TDigest,
)


def _stream(n: int = 20000, seed: int = 0) -> np.ndarray:
    """A skewed stream: detection scores pile up near zero with a thin upper tail."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.beta(0.5, 12.0, n), 0.0, 1.0)


def _feed(estimator: object, values: np.ndarray) -> object:
    for value in values:
        estimator.update(float(value))  # type: ignore[attr-defined]
    return estimator


# --------------------------------------------------------------------------------------
# Reservoir sampling.
# --------------------------------------------------------------------------------------


def test_a_reservoir_holds_at_most_its_size() -> None:
    estimator = ReservoirQuantile(500, np.random.default_rng(1))
    _feed(estimator, _stream(5000))
    assert len(estimator.samples) == 500
    assert estimator.seen == 5000
    assert estimator.memory_bytes == 4000


def test_a_reservoir_keeps_everything_while_it_is_filling() -> None:
    values = _stream(100)
    estimator = ReservoirQuantile(500, np.random.default_rng(2))
    _feed(estimator, values)
    assert sorted(estimator.samples) == sorted(values.tolist())


def test_a_reservoir_sample_is_representative_of_the_median() -> None:
    values = _stream(40000, seed=3)
    estimator = ReservoirQuantile(4000, np.random.default_rng(3))
    _feed(estimator, values)
    assert estimator.quantile(0.5) == pytest.approx(float(np.quantile(values, 0.5)), abs=0.01)


# --------------------------------------------------------------------------------------
# P-squared.
# --------------------------------------------------------------------------------------


def test_p_squared_tracks_a_high_quantile_in_five_numbers() -> None:
    values = _stream(50000, seed=4)
    estimator = P2Quantile(0.99)
    _feed(estimator, values)
    truth = float(np.quantile(values, 0.99))
    assert estimator.quantile(0.99) == pytest.approx(truth, abs=0.02)
    assert estimator.memory_bytes < 200  # the entire point: constant, tiny memory


def test_p_squared_markers_stay_ordered() -> None:
    # The parabolic prediction can put a marker out of order; the algorithm is only correct
    # because it detects that and falls back to linear interpolation.
    for seed in range(5):
        estimator = P2Quantile(0.999)
        _feed(estimator, _stream(5000, seed=seed))
        assert estimator.heights == sorted(estimator.heights)


def test_p_squared_is_exact_on_its_first_five_observations() -> None:
    estimator = P2Quantile(0.5)
    _feed(estimator, np.array([5.0, 1.0, 3.0, 2.0, 4.0]))
    assert estimator.heights == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert estimator.quantile(0.5) == pytest.approx(3.0)


def test_p_squared_handles_a_constant_stream() -> None:
    estimator = P2Quantile(0.9)
    _feed(estimator, np.full(1000, 0.42))
    assert estimator.quantile(0.9) == pytest.approx(0.42)


# --------------------------------------------------------------------------------------
# t-digest.
# --------------------------------------------------------------------------------------


def test_the_digest_compresses_and_still_finds_the_median() -> None:
    values = _stream(30000, seed=5)
    digest = TDigest(compression=100.0)
    _feed(digest, values)
    assert digest.quantile(0.5) == pytest.approx(float(np.quantile(values, 0.5)), abs=0.02)
    # Compression is bounded by the scale function rather than by a fixed count: what the
    # invariant guarantees is that centroids grow far slower than observations.
    assert len(digest.centroids) < len(values) // 20


def test_the_digest_keeps_the_tail_finer_than_the_middle() -> None:
    # The scale function's whole purpose: centroid weight is bounded by q(1-q), so centroids
    # near the extremes stay small while the middle merges freely.
    digest = TDigest(compression=100.0)
    _feed(digest, _stream(40000, seed=6))
    weights = [c.weight for c in digest.centroids]
    edge = np.mean(weights[:5] + weights[-5:])
    middle = np.mean(weights[len(weights) // 2 - 3 : len(weights) // 2 + 3])
    assert edge < middle


def test_a_higher_compression_keeps_more_centroids() -> None:
    values = _stream(20000, seed=7)
    coarse = _feed(TDigest(compression=20.0), values)
    fine = _feed(TDigest(compression=500.0), values)
    assert len(coarse.centroids) < len(fine.centroids)  # type: ignore[attr-defined]


def test_the_digest_is_accurate_in_the_tail_it_is_built_for() -> None:
    values = _stream(50000, seed=8)
    digest = _feed(TDigest(compression=200.0), values)
    truth = float(np.quantile(values, 0.999))
    assert digest.quantile(0.999) == pytest.approx(truth, abs=0.05)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# The histogram, and the interpolation that makes it usable.
# --------------------------------------------------------------------------------------


def test_the_histogram_is_accurate_to_about_a_bin_width() -> None:
    values = _stream(50000, seed=9)
    histogram = _feed(HistogramQuantile(1000), values)
    truth = float(np.quantile(values, 0.999))
    assert histogram.quantile(0.999) == pytest.approx(truth, abs=2 / 1000)  # type: ignore[attr-defined]


def test_more_bins_means_less_error() -> None:
    values = _stream(50000, seed=10)
    truth = float(np.quantile(values, 0.99))
    coarse = abs(_feed(HistogramQuantile(50), values).quantile(0.99) - truth)  # type: ignore[attr-defined]
    fine = abs(_feed(HistogramQuantile(20000), values).quantile(0.99) - truth)  # type: ignore[attr-defined]
    assert fine < coarse


def test_the_histogram_interpolates_inside_the_bin() -> None:
    # Returning the bin's lower edge is a systematic underestimate; with everything inside one
    # bin, the interpolated 90th percentile must sit near the top of it rather than at the base.
    histogram = HistogramQuantile(10)
    _feed(histogram, np.full(1000, 0.55))
    estimate = histogram.quantile(0.9)
    assert 0.5 < estimate <= 0.6
    assert estimate > 0.58


def test_an_empty_estimator_returns_zero_rather_than_raising() -> None:
    assert HistogramQuantile(10).quantile(0.5) == 0.0
    assert TDigest().quantile(0.5) == 0.0
    assert P2Quantile(0.5).quantile(0.5) == 0.0
    assert ReservoirQuantile(10, np.random.default_rng(0)).quantile(0.5) == 0.0


# --------------------------------------------------------------------------------------
# The property that motivates the whole report.
# --------------------------------------------------------------------------------------


def test_every_estimator_lands_within_a_hair_of_the_exact_tail_quantile() -> None:
    """All four agree with `numpy.quantile` at the operating point, on the same stream.

    This is the report's premise as a test: the estimators are operationally interchangeable
    on a bounded score, and the differences between them are memory and update cost rather
    than accuracy.
    """
    values = _stream(60000, seed=11)
    truth = float(np.quantile(values, 0.999))
    estimators = {
        "reservoir": _feed(ReservoirQuantile(20000, np.random.default_rng(11)), values),
        "p2": _feed(P2Quantile(0.999), values),
        "tdigest": _feed(TDigest(200.0), values),
        "histogram": _feed(HistogramQuantile(10000), values),
    }
    for name, estimator in estimators.items():
        estimate = estimator.quantile(0.999)  # type: ignore[attr-defined]
        assert abs(estimate - truth) < 0.05, f"{name} drifted: {estimate} vs {truth}"
