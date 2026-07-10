"""Exemplar retrieval: index balance, exact k-NN, support votes, payload round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.explain.exemplars import ExemplarIndex, build_exemplar_index, exemplar_support


def _index(n_per_class: int = 50, d: int = 4) -> ExemplarIndex:
    rng = np.random.default_rng(0)
    matrix = np.vstack([rng.normal(0, 1, (n_per_class, d)), rng.normal(5, 1, (n_per_class, d))])
    labels = np.array(["BENIGN"] * n_per_class + ["DDoS"] * n_per_class)
    days = np.array(["Monday"] * (2 * n_per_class))
    return ExemplarIndex(matrix=matrix.astype(np.float32), labels=labels, days=days)


def test_build_index_caps_every_class_at_per_class() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(0, 1, (700, 3))
    labels = np.array(["BENIGN"] * 500 + ["DoS Hulk"] * 150 + ["Heartbleed"] * 50)
    days = np.array(["Monday"] * 700)
    index = build_exemplar_index(matrix, labels, days, per_class=100, seed=42)
    counts = {label: int((index.labels == label).sum()) for label in set(labels)}
    assert counts["BENIGN"] == 100  # capped
    assert counts["DoS Hulk"] == 100  # capped
    assert counts["Heartbleed"] == 50  # rarer than the cap: kept whole
    assert index.matrix.dtype == np.float32


def test_query_returns_self_at_distance_zero() -> None:
    index = _index()
    distances, idx = index.query(index.matrix[[3]], k=1)
    assert idx[0, 0] == 3
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-4)


def test_query_orders_neighbours_nearest_first() -> None:
    index = _index()
    rng = np.random.default_rng(2)
    distances, _idx = index.query(rng.normal(2.5, 1, (5, 4)), k=4)
    assert np.all(np.diff(distances, axis=1) >= 0)  # sorted ascending per row
    # A query deep inside the DDoS cluster must retrieve DDoS-labeled cases.
    _d2, i2 = index.query(np.full((1, 4), 5.0), k=3)
    assert all(index.labels[j] == "DDoS" for j in i2[0])


def test_query_k_larger_than_index_is_clamped() -> None:
    index = _index(n_per_class=2)
    distances, _idx = index.query(np.zeros((1, 4)), k=50)
    assert distances.shape == (1, 4)  # 4 exemplars total


def test_exemplar_support_majority_with_conservative_ties() -> None:
    assert exemplar_support(np.array([[1, 1, 1, 0, 0]])).tolist() == [True]  # 3/5 attack
    assert exemplar_support(np.array([[1, 1, 0, 0, 0]])).tolist() == [False]  # 2/5
    assert exemplar_support(np.array([[1, 1, 0, 0]])).tolist() == [False]  # tie: no support


def test_payload_round_trip_preserves_the_index() -> None:
    index = _index(n_per_class=5)
    clone = ExemplarIndex.from_payload(index.to_payload())
    assert np.array_equal(clone.matrix, index.matrix)
    assert clone.labels.tolist() == index.labels.tolist()
    assert clone.days.tolist() == index.days.tolist()
