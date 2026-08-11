"""Sketches: every structure's guarantee, checked rather than cited.

These are approximate data structures, so "it returned a number" proves nothing. What each one
promises is specific — Count-Min never undercounts, HyperLogLog's error scales as
1.04/sqrt(m), Misra-Gries keeps every 1/k-share key, a reservoir is uniform — and each promise
is what the tests below assert.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from netsentry.intel.sketches import (
    CountMin,
    HyperLogLog,
    MisraGries,
    reservoir_sample,
    synthesize_stream,
)


# --------------------------------------------------------------------------------------
# Count-Min
# --------------------------------------------------------------------------------------
def test_count_min_never_undercounts() -> None:
    """The one-sided guarantee, and the only reason this is safe to detect with."""
    sketch = CountMin(width=64, depth=4)
    truth = {f"host-{i}": (i % 7) + 1 for i in range(500)}
    for key, count in truth.items():
        for _ in range(count):
            sketch.add(key)
    assert all(sketch.estimate(key) >= count for key, count in truth.items())


def test_count_min_is_exact_when_nothing_collides() -> None:
    sketch = CountMin(width=4096, depth=3)
    for key in ("a", "b", "b", "c", "c", "c"):
        sketch.add(key)
    assert (sketch.estimate("a"), sketch.estimate("b"), sketch.estimate("c")) == (1, 2, 3)


def test_the_epsilon_n_bound_holds_for_almost_every_key() -> None:
    """The actual theorem: overestimate <= epsilon * N with probability at least 1 - delta."""
    epsilon, delta = 0.01, 0.01
    sketch = CountMin.for_guarantee(epsilon, delta)
    rng = np.random.default_rng(0)
    keys = [f"h{k}" for k in rng.integers(0, 5000, size=20_000)]
    truth: dict[str, int] = {}
    for key in keys:
        sketch.add(key)
        truth[key] = truth.get(key, 0) + 1
    allowance = epsilon * sketch.total
    within = np.mean([sketch.estimate(k) - v <= allowance for k, v in truth.items()])
    assert within >= 1 - delta


def test_sizing_from_a_guarantee_gets_wider_as_epsilon_tightens() -> None:
    loose = CountMin.for_guarantee(0.01, 0.01)
    tight = CountMin.for_guarantee(0.001, 0.01)
    assert tight.width > loose.width
    assert tight.depth == loose.depth  # depth answers delta, width answers epsilon


def test_memory_does_not_grow_with_the_number_of_keys() -> None:
    """The property the whole report is about."""
    sketch = CountMin(width=256, depth=4)
    before = sketch.n_bytes()
    for i in range(100_000):
        sketch.add(f"host-{i}")
    assert sketch.n_bytes() == before


# --------------------------------------------------------------------------------------
# HyperLogLog
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("cardinality", [100, 5_000, 100_000])
def test_hyperloglog_lands_within_a_few_standard_errors(cardinality: int) -> None:
    hll = HyperLogLog(precision=12)
    for i in range(cardinality):
        hll.add(f"dst-{i}")
    error = abs(hll.estimate() - cardinality) / cardinality
    assert error < 3 * 1.04 / math.sqrt(hll.m)


def test_error_falls_as_registers_grow() -> None:
    """The 1.04/sqrt(m) scaling, observed rather than assumed."""
    errors = []
    for precision in (6, 10, 14):
        hll = HyperLogLog(precision=precision)
        for i in range(50_000):
            hll.add(f"dst-{i}")
        errors.append(abs(hll.estimate() - 50_000) / 50_000)
    assert errors[-1] < errors[0]


def test_duplicates_do_not_move_the_estimate() -> None:
    hll = HyperLogLog(precision=10)
    for i in range(1000):
        hll.add(f"dst-{i}")
    before = hll.estimate()
    for _ in range(20):
        for i in range(1000):
            hll.add(f"dst-{i}")
    assert hll.estimate() == pytest.approx(before)


def test_the_small_range_correction_saves_low_cardinalities() -> None:
    """Without linear counting the raw estimator is badly biased exactly where scans start."""
    hll = HyperLogLog(precision=12)
    for i in range(30):
        hll.add(f"dst-{i}")
    assert 20 <= hll.estimate() <= 45


def test_an_empty_sketch_estimates_nothing() -> None:
    assert HyperLogLog(precision=8).estimate() == pytest.approx(0.0)


def test_two_sketches_of_the_same_set_agree_exactly() -> None:
    """Determinism: the hash is keyed and stable, not Python's randomised one."""
    a, b = HyperLogLog(precision=10), HyperLogLog(precision=10)
    for i in range(5000):
        a.add(f"x{i}")
    for i in reversed(range(5000)):
        b.add(f"x{i}")
    assert a.estimate() == pytest.approx(b.estimate())


# --------------------------------------------------------------------------------------
# Misra-Gries
# --------------------------------------------------------------------------------------
def test_every_heavy_hitter_survives() -> None:
    """The guarantee: anything above a 1/k share of the stream is in the summary."""
    k = 8
    summary = MisraGries(k=k)
    stream = ["whale"] * 400 + ["shark"] * 300 + [f"minnow-{i}" for i in range(300)]
    for key in stream:
        summary.add(key)
    kept = {key for key, _ in summary.top(k)}
    threshold = len(stream) / k
    for key in ("whale", "shark"):
        assert stream.count(key) > threshold and key in kept


def test_the_counter_budget_is_respected() -> None:
    summary = MisraGries(k=5)
    for i in range(10_000):
        summary.add(f"key-{i}")
    assert len(summary.counters) <= 4


def test_counts_are_never_overstated() -> None:
    """Misra-Gries under-counts by design; the error runs one way."""
    summary = MisraGries(k=6)
    stream = ["a"] * 50 + ["b"] * 30 + [f"c{i}" for i in range(40)]
    for key in stream:
        summary.add(key)
    for key, count in summary.counters.items():
        assert count <= stream.count(key)


# --------------------------------------------------------------------------------------
# Reservoir sampling
# --------------------------------------------------------------------------------------
def test_the_reservoir_fills_and_then_stops_growing() -> None:
    rng = np.random.default_rng(0)
    assert len(reservoir_sample([str(i) for i in range(10_000)], 100, rng)) == 100
    assert len(reservoir_sample(["a", "b"], 100, rng)) == 2


def test_the_sample_is_uniform_over_the_stream() -> None:
    """Every item equally likely to survive — the property that makes one pass acceptable."""
    rng = np.random.default_rng(1)
    stream = [str(i) for i in range(50)]
    counts = np.zeros(50)
    for _ in range(4000):
        for item in reservoir_sample(stream, 5, rng):
            counts[int(item)] += 1
    expected = counts.sum() / 50
    assert np.all(np.abs(counts - expected) < 0.35 * expected)


# --------------------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------------------
def test_the_stream_is_skewed_and_contains_its_planted_scanners() -> None:
    stream = synthesize_stream(20_000, 500, 1.1, scanners=2, scanner_targets=300, seed=0)
    sources = [s for s, _ in stream]
    scanners = {s for s in sources if s.startswith("203.0.113.")}
    assert len(scanners) == 2
    for scanner in scanners:
        assert len({d for s, d in stream if s == scanner}) == 300
    counts = sorted((sources.count(s) for s in set(sources)), reverse=True)
    assert counts[0] > 5 * counts[len(counts) // 2]  # heavy-tailed, not uniform


def test_the_stream_is_reproducible_and_shuffled() -> None:
    first = synthesize_stream(2_000, 100, 1.1, 1, 50, seed=3)
    assert first == synthesize_stream(2_000, 100, 1.1, 1, 50, seed=3)
    scanner_positions = [i for i, (s, _) in enumerate(first) if s.startswith("203.0.113.")]
    assert max(scanner_positions) - min(scanner_positions) > len(first) // 4
