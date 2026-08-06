"""Attack-family discovery: the label-free protocol and the grading applied afterwards.

The methodological claim is that labels never influence the clustering — they only grade it
— so the tests exercise the pieces on either side of that line separately: k selection and
naming (label-free), then purity, discovery, and the random control (label-aware, applied
after the fact).
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.discovery import (
    choose_k,
    cluster_purity,
    discovered_families,
    nearest_known_class,
    random_baseline_ari,
)


def _three_blobs(seed: int = 0, n: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return np.vstack([c + rng.normal(0, 0.3, size=(n // 3, 2)) for c in centres])


def test_k_selection_finds_the_true_number_of_blobs_without_labels() -> None:
    k, scores = choose_k(_three_blobs(), [2, 3, 4, 5], seed=0, sample=0)
    assert k == 3
    assert scores[3] > scores[2] and scores[3] > scores[4]


def test_k_selection_reports_a_score_for_every_feasible_candidate() -> None:
    _, scores = choose_k(_three_blobs(), [2, 3, 4], seed=0, sample=0)
    assert set(scores) == {2, 3, 4}


def test_k_selection_skips_candidates_larger_than_the_dataset() -> None:
    k, scores = choose_k(_three_blobs(n=6), [2, 3, 100], seed=0, sample=0)
    assert 100 not in scores and k in {2, 3}


def test_k_selection_is_deterministic() -> None:
    x = _three_blobs()
    assert (
        choose_k(x, [2, 3, 4], seed=7, sample=0)[0] == choose_k(x, [2, 3, 4], seed=7, sample=0)[0]
    )


def test_purity_is_one_for_a_perfectly_separated_clustering() -> None:
    clusters = np.array([0, 0, 1, 1])
    truth = np.array(["DoS", "DoS", "PortScan", "PortScan"])
    assert cluster_purity(clusters, truth) == 1.0


def test_purity_measures_the_majority_share_of_each_cluster() -> None:
    clusters = np.array([0, 0, 0, 0])
    truth = np.array(["DoS", "DoS", "DoS", "PortScan"])
    assert cluster_purity(clusters, truth) == 0.75


def test_purity_of_an_empty_clustering_is_zero() -> None:
    assert cluster_purity(np.zeros(0), np.zeros(0)) == 0.0


def test_singleton_clusters_score_perfect_purity_which_is_why_ari_is_reported_too() -> None:
    # Purity's known weakness, pinned so nobody reads it as the whole story.
    clusters = np.arange(4)
    truth = np.array(["a", "b", "c", "d"])
    assert cluster_purity(clusters, truth) == 1.0


def test_a_family_counts_as_discovered_when_a_cluster_is_mostly_made_of_it() -> None:
    clusters = np.array([0] * 25 + [1] * 25)
    truth = np.array(["DoS Hulk"] * 25 + ["PortScan"] * 25)
    assert discovered_families(clusters, truth, min_purity=0.6, min_size=20) == [
        "DoS Hulk",
        "PortScan",
    ]


def test_a_cluster_too_small_to_open_does_not_count_as_a_discovery() -> None:
    clusters = np.zeros(5, dtype=int)
    truth = np.array(["Heartbleed"] * 5)
    assert discovered_families(clusters, truth, min_purity=0.6, min_size=20) == []


def test_a_muddled_cluster_does_not_count_as_a_discovery() -> None:
    clusters = np.zeros(40, dtype=int)
    truth = np.array(["DoS"] * 20 + ["PortScan"] * 20)  # 50% dominant, below the 60% bar
    assert discovered_families(clusters, truth, min_purity=0.6, min_size=20) == []


def test_random_assignment_scores_near_zero_ari() -> None:
    truth = np.repeat(["a", "b", "c"], 100)
    assert abs(random_baseline_ari(truth, k=3, seed=0, trials=20)) < 0.02


def test_nearest_known_class_picks_the_closest_centroid() -> None:
    centroids = {"DoS": np.array([0.0, 0.0]), "PortScan": np.array([10.0, 0.0])}
    name, dist = nearest_known_class(np.array([9.0, 0.0]), centroids)
    assert name == "PortScan" and dist == pytest.approx(1.0)


def test_a_cluster_far_from_everything_still_reports_its_distance() -> None:
    # The distance is what marks a candidate new family; the name alone would mislead.
    centroids = {"DoS": np.array([0.0, 0.0])}
    name, dist = nearest_known_class(np.array([500.0, 0.0]), centroids)
    assert name == "DoS" and dist == pytest.approx(500.0)


def test_an_empty_catalogue_names_nothing() -> None:
    name, dist = nearest_known_class(np.array([1.0]), {})
    assert name == "unknown" and dist == float("inf")
