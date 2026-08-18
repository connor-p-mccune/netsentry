"""Slice discovery: the search, and the three defences that keep it honest.

The dangerous failure mode of a slice finder is not a crash, it is a plausible region that
does not exist. So the tests plant a region that *does* exist and check the search finds it,
then run the whole machinery on noise and check it finds nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.slice_discovery import (
    Literal,
    beam_search,
    build_literals,
    literal_mask,
    normal_sf,
    score_slice,
)
from netsentry.monitoring.detectors import benjamini_hochberg


def _planted(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Three features and a loss that is elevated only where f0 is low and f1 is high."""
    rng = np.random.default_rng(seed)
    matrix = rng.random((n, 3))
    region = (matrix[:, 0] < 0.2) & (matrix[:, 1] > 0.8)
    loss = rng.normal(1.0, 0.2, n) + 2.0 * region
    return matrix, ["f0", "f1", "f2"], loss


# --------------------------------------------------------------------------------------
# Literals.
# --------------------------------------------------------------------------------------


def test_literals_partition_every_row_exactly_once_per_feature() -> None:
    matrix, names, _ = _planted(n=500)
    membership, literals = build_literals(matrix, names, n_bins=5)
    for name in names:
        columns = [i for i, literal in enumerate(literals) if literal.feature == name]
        covered = membership[:, columns].sum(axis=1)
        assert set(np.unique(covered).tolist()) == {1}  # a partition, not an overlap


def test_literal_bounds_are_observed_values_not_interpolations() -> None:
    # Bounds are what the confirmation half re-applies, and an interpolated bound on an
    # integer-valued feature is both unreadable and not a real cut point.
    values = np.repeat(np.arange(10.0), 50)
    matrix = values.reshape(-1, 1)
    _, literals = build_literals(matrix, ["packets"], n_bins=5)
    for literal in literals:
        if np.isfinite(literal.low):
            assert literal.low in set(values.tolist())


def test_constant_features_produce_no_literals() -> None:
    matrix = np.column_stack([np.ones(200), np.random.default_rng(1).random(200)])
    _, literals = build_literals(matrix, ["constant", "varying"], n_bins=4)
    assert all(literal.feature == "varying" for literal in literals)


def test_a_literal_can_be_reapplied_to_different_rows() -> None:
    literal = Literal(feature="f", index=0, low=0.2, high=0.5)
    matrix = np.array([[0.1], [0.2], [0.3], [0.5], [0.6]])
    assert literal_mask(matrix, literal).tolist() == [False, False, True, True, False]


# --------------------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------------------


def test_score_slice_recovers_a_planted_effect() -> None:
    rng = np.random.default_rng(2)
    loss = rng.normal(1.0, 0.1, 2000)
    mask = np.zeros(2000, dtype=bool)
    mask[:300] = True
    loss[mask] += 0.5
    score = score_slice(mask, loss)
    assert score.support == 300
    assert score.effect == pytest.approx(0.5, abs=0.05)
    assert score.p_value < 1e-6


def test_score_slice_is_one_sided() -> None:
    # A slice where the model does *better* is not a finding; its p-value must stay large or
    # the search will spend its false-discovery budget on good news.
    rng = np.random.default_rng(3)
    loss = rng.normal(1.0, 0.1, 2000)
    mask = np.zeros(2000, dtype=bool)
    mask[:300] = True
    loss[mask] -= 0.5
    assert score_slice(mask, loss).p_value > 0.99


def test_score_slice_handles_degenerate_masks() -> None:
    loss = np.ones(10)
    assert score_slice(np.zeros(10, dtype=bool), loss).p_value == 1.0
    assert score_slice(np.ones(10, dtype=bool), loss).p_value == 1.0


def test_normal_sf_matches_known_values() -> None:
    assert normal_sf(0.0) == pytest.approx(0.5)
    assert normal_sf(1.96) == pytest.approx(0.025, abs=1e-3)
    assert normal_sf(-1.96) == pytest.approx(0.975, abs=1e-3)


# --------------------------------------------------------------------------------------
# The search.
# --------------------------------------------------------------------------------------


def test_the_search_finds_a_planted_conjunction() -> None:
    matrix, names, loss = _planted()
    membership, literals = build_literals(matrix, names, n_bins=5)
    found, tested = beam_search(membership, literals, loss, depth=2, beam=10, min_support=20)
    assert tested > 0
    best = max(found, key=lambda s: s.score.effect)
    features = {literal.feature for literal in best.literals}
    assert features == {"f0", "f1"}  # the planted region, and not the irrelevant f2
    assert best.score.effect > 1.0


def test_the_search_deduplicates_permuted_conjunctions() -> None:
    matrix, names, loss = _planted(n=1500, seed=4)
    membership, literals = build_literals(matrix, names, n_bins=4)
    found, _ = beam_search(membership, literals, loss, depth=2, beam=8, min_support=20)
    keys = [frozenset(literal.describe() for literal in s.literals) for s in found]
    assert len(keys) == len(set(keys))  # "A AND B" and "B AND A" are one slice


def test_the_search_never_conjoins_two_bins_of_one_feature() -> None:
    # Two bins of the same feature are disjoint by construction, so their conjunction is
    # always empty and would waste the whole beam.
    matrix, names, loss = _planted(n=1200, seed=5)
    membership, literals = build_literals(matrix, names, n_bins=4)
    found, _ = beam_search(membership, literals, loss, depth=2, beam=8, min_support=10)
    for slice_ in found:
        features = [literal.feature for literal in slice_.literals]
        assert len(features) == len(set(features))


def test_support_floor_is_respected() -> None:
    matrix, names, loss = _planted(n=1000, seed=6)
    membership, literals = build_literals(matrix, names, n_bins=10)
    found, _ = beam_search(membership, literals, loss, depth=2, beam=6, min_support=200)
    assert all(s.score.support >= 200 for s in found)


# --------------------------------------------------------------------------------------
# The defence that matters: nothing real should be found in noise.
# --------------------------------------------------------------------------------------


def test_the_search_finds_nothing_significant_in_pure_noise() -> None:
    """The null calibration, as a test.

    A search over thousands of candidate regions on random loss will produce plenty of
    uncorrected 'significant' slices; after Benjamini-Hochberg it must produce essentially
    none, or every finding in the report is unfalsifiable.
    """
    rng = np.random.default_rng(7)
    matrix = rng.random((3000, 8))
    loss = rng.normal(1.0, 0.2, 3000)  # no structure whatsoever
    membership, literals = build_literals(matrix, [f"f{i}" for i in range(8)], n_bins=5)
    found, tested = beam_search(membership, literals, loss, depth=2, beam=15, min_support=40)
    assert tested > 500  # the search really did look at many candidates
    p_values = np.array([s.score.p_value for s in found], dtype=float)
    raw = int(np.sum(p_values <= 0.05))
    adjusted, _ = benjamini_hochberg(p_values, 0.05)
    assert raw > 0  # uncorrected testing does flag regions in noise ...
    assert int(adjusted.sum()) == 0  # ... and the correction removes them
