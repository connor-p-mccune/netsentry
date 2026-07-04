"""Novelty-distance tests: NN distances on known geometry, quantile bin edges,
per-bin detection (including the closed last bin and empty-bin NaN), and the
composition counterfactual that decomposes the split gap."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.novelty import (
    NoveltyBin,
    bin_detection,
    composition_counterfactual,
    nn_distances,
    quantile_edges,
)


def test_nn_distances_on_known_geometry() -> None:
    reference = np.array([[0.0, 0.0], [10.0, 10.0]])
    queries = np.array([[0.0, 1.0], [10.0, 10.0], [6.0, 6.0]])
    d = nn_distances(reference, queries)
    assert d == pytest.approx([1.0, 0.0, np.sqrt(32.0)])


def test_quantile_edges_are_monotone_and_cover_the_range() -> None:
    rng = np.random.default_rng(0)
    distances = rng.exponential(size=500)
    edges = quantile_edges(distances, n_bins=5)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == pytest.approx(distances.min())
    assert edges[-1] == pytest.approx(distances.max())


def test_quantile_edges_degenerate_distances_yield_one_bin() -> None:
    edges = quantile_edges(np.full(10, 2.5), n_bins=4)
    assert len(edges) == 2 and edges[0] == 2.5


def test_bin_detection_per_bin_rates_and_closed_last_bin() -> None:
    distances = np.array([0.1, 0.2, 5.0, 10.0])  # 10.0 sits ON the last edge
    detected = np.array([True, True, False, True])
    bins = bin_detection(distances, detected, np.array([0.0, 1.0, 10.0]))
    assert [b.n for b in bins] == [2, 2]
    assert bins[0].detection == 1.0
    assert bins[1].detection == 0.5  # the on-edge point is included in the last bin


def test_bin_detection_empty_bin_is_nan() -> None:
    bins = bin_detection(np.array([5.0]), np.array([True]), np.array([0.0, 1.0, 10.0]))
    assert bins[0].n == 0 and np.isnan(bins[0].detection)
    assert bins[1].detection == 1.0


def test_composition_counterfactual_reweights_source_rates_to_target_mix() -> None:
    source = [NoveltyBin(0, 1, 50, 0.9), NoveltyBin(1, 2, 50, 0.1)]
    target = [NoveltyBin(0, 1, 25, 0.5), NoveltyBin(1, 2, 75, 0.0)]
    # 25% of the target mix at 0.9 plus 75% at 0.1.
    assert composition_counterfactual(source, target) == pytest.approx(0.25 * 0.9 + 0.75 * 0.1)


def test_composition_counterfactual_skips_undefined_bins() -> None:
    source = [NoveltyBin(0, 1, 50, float("nan")), NoveltyBin(1, 2, 50, 0.2)]
    target = [NoveltyBin(0, 1, 60, 0.5), NoveltyBin(1, 2, 40, 0.1)]
    # Only the second bin is usable; its weight renormalizes to 1.
    assert composition_counterfactual(source, target) == pytest.approx(0.2)


def test_composition_counterfactual_with_no_common_bins_is_nan() -> None:
    source = [NoveltyBin(0, 1, 10, float("nan"))]
    target = [NoveltyBin(0, 1, 0, float("nan"))]
    assert np.isnan(composition_counterfactual(source, target))
