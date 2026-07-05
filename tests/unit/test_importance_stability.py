"""Feature-importance stability metrics, validated against known importance matrices."""

from __future__ import annotations

import numpy as np

from netsentry.explain.importance_stability import stability_metrics

_NAMES = ["a", "b", "c", "d"]


def test_identical_runs_are_perfectly_stable() -> None:
    matrix = np.tile([0.4, 0.3, 0.2, 0.1], (5, 1))  # every refit identical
    result = stability_metrics(matrix, _NAMES, top_k=2)
    assert result.rank_correlation == 1.0
    assert result.topk_jaccard == 1.0
    assert result.features[0].feature == "a"  # highest mean importance leads
    for f in result.features:
        assert f.rank_std == 0.0  # no rank movement
    # 'a' and 'b' are always the top-2; 'c' and 'd' never are.
    freq = {f.feature: f.topk_frequency for f in result.features}
    assert freq["a"] == 1.0 and freq["b"] == 1.0
    assert freq["c"] == 0.0 and freq["d"] == 0.0


def test_reversed_rankings_anticorrelate() -> None:
    matrix = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    result = stability_metrics(matrix, _NAMES, top_k=2)
    assert result.rank_correlation < 0  # exactly reversed -> Spearman -1
    assert result.topk_jaccard == 0.0  # disjoint top-2 sets


def test_topk_frequency_counts_membership() -> None:
    # 'a' leads twice, 'b' leads once; top_k=1 tracks the single leader per run.
    matrix = np.array([[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]])
    result = stability_metrics(matrix, _NAMES, top_k=1)
    freq = {f.feature: f.topk_frequency for f in result.features}
    assert freq["a"] == 2 / 3
    assert freq["b"] == 1 / 3


def test_top_k_is_clamped_to_feature_count() -> None:
    matrix = np.tile([0.5, 0.5], (3, 1))
    result = stability_metrics(matrix, ["x", "y"], top_k=10)
    assert result.top_k == 2  # clamped, no index error
    assert result.topk_jaccard == 1.0
