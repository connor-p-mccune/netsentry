"""The private synthesiser, tested where it would be wrong in a way that still looks fine.

Three failure modes matter here and none of them shows up in the output data: bin edges that
secretly came from the data (a leak before any noise is added), a mechanism whose noise does
not actually scale with epsilon (a guarantee in the docstring only), and a budget that is
described one way and spent another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings
from netsentry.data.clean import BINARY_TARGET
from netsentry.data.dp_synth import (
    BayesNet,
    PrivacyBudget,
    chow_liu_structure,
    discretize,
    family_structure,
    fit_network,
    inverse_signed_log,
    joint_counts,
    laplace_probabilities,
    n_levels,
    public_bin_edges,
    sample_within_bins,
    signed_log,
    synthesize,
)


def _edges(n_bins: int = 8) -> np.ndarray:
    return public_bin_edges(n_bins, 1e6, -1.0)


# --------------------------------------------------------------------------------------
# The grid, which must not depend on the data.
# --------------------------------------------------------------------------------------


def test_bin_edges_are_a_function_of_the_declared_domain_only() -> None:
    # If this test ever needs data to pass, the release has leaked before the noise is added.
    assert np.array_equal(_edges(), _edges())
    assert np.all(np.diff(_edges()) > 0)
    assert len(_edges(12)) == 13


def test_signed_log_round_trips_through_zero_and_negatives() -> None:
    values = np.array([-1.0, 0.0, 1.0, 1e3, 1e9])
    assert np.allclose(inverse_signed_log(signed_log(values)), values)


def test_missing_values_get_their_own_level() -> None:
    edges = _edges()
    matrix = np.array([[np.nan, 0.0], [np.inf, 1e5]])
    levels = discretize(matrix, edges)
    assert levels[0, 0] == 0 and levels[1, 0] == 0  # NaN and Inf are both "missing"
    assert levels[0, 1] > 0 and levels[1, 1] > 0
    assert n_levels(edges) == len(edges)


def test_out_of_domain_values_clip_rather_than_overflow() -> None:
    edges = _edges(8)
    huge = discretize(np.array([[1e15]]), edges)
    tiny = discretize(np.array([[-1e9]]), edges)
    assert 1 <= int(huge[0, 0]) <= n_levels(edges) - 1
    assert 1 <= int(tiny[0, 0]) <= n_levels(edges) - 1


def test_sampling_stays_inside_the_chosen_bin() -> None:
    edges = _edges(6)
    rng = np.random.default_rng(0)
    levels = np.array([[0, 1, 3, 6]])
    values = sample_within_bins(levels, edges, rng)
    assert np.isnan(values[0, 0])  # level 0 decodes back to missing
    transformed = signed_log(values[0, 1:])
    assert np.all(transformed >= edges[np.array([0, 2, 5])] - 1e-9)
    assert np.all(transformed <= edges[np.array([1, 3, 6])] + 1e-9)


# --------------------------------------------------------------------------------------
# The mechanism.
# --------------------------------------------------------------------------------------


def test_released_histograms_are_distributions() -> None:
    rng = np.random.default_rng(1)
    counts = np.array([100.0, 0.0, 5.0, 40.0])
    for epsilon in (0.1, 1.0, float("inf")):
        probabilities = laplace_probabilities(counts, epsilon, rng)
        assert probabilities.min() >= 0.0
        assert probabilities.sum() == pytest.approx(1.0)


def test_infinite_epsilon_is_the_empirical_distribution() -> None:
    # The no-privacy control has to be exactly the data, or the "what does DP cost" comparison
    # is measuring the control's own error too.
    counts = np.array([3.0, 7.0, 10.0])
    exact = laplace_probabilities(counts, float("inf"), np.random.default_rng(2))
    assert np.allclose(exact, counts / counts.sum())


def test_noise_scales_the_way_the_guarantee_says_it_does() -> None:
    # Laplace(1/epsilon) on counts: halving epsilon should roughly double the error. Measured
    # over repeats rather than asserted, because a mechanism whose noise does not move with
    # epsilon satisfies its docstring and nothing else.
    rng = np.random.default_rng(3)
    counts = np.full(16, 500.0)
    truth = counts / counts.sum()

    def _error(epsilon: float) -> float:
        draws = [
            float(np.abs(laplace_probabilities(counts, epsilon, rng) - truth).sum())
            for _ in range(60)
        ]
        return float(np.mean(draws))

    tight, loose = _error(0.05), _error(0.4)
    assert tight > loose
    assert 4.0 < tight / loose < 16.0  # 8x expected; wide band so the test is not flaky


# --------------------------------------------------------------------------------------
# Structure.
# --------------------------------------------------------------------------------------


def test_the_public_structure_is_a_forest_within_families() -> None:
    from netsentry.features.feature_sets import feature_group, numeric_features

    features = numeric_features()
    parents = family_structure(features)
    for index, parent in enumerate(parents):
        assert parent < index  # strictly earlier => acyclic by construction
        if parent >= 0:
            assert feature_group(features[parent]) == feature_group(features[index])


def test_chow_liu_finds_the_dependence_it_is_shown() -> None:
    rng = np.random.default_rng(4)
    base = rng.integers(0, 5, size=800)
    bins = np.column_stack([base, base, rng.integers(0, 5, size=800)]).astype(np.int16)
    parents = chow_liu_structure(bins, 5)
    # Feature 1 is a copy of feature 0, so the maximum-information edge is between them.
    assert parents[1] == 0 or parents[0] == 1


def test_joint_counts_match_a_naive_cross_tabulation() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(0, 4, size=200)
    b = rng.integers(0, 4, size=200)
    table = joint_counts(a, b, 4)
    naive = np.zeros((4, 4))
    for x, y in zip(a, b, strict=True):
        naive[x, y] += 1
    assert np.array_equal(table, naive)


def test_a_cyclic_structure_is_rejected_rather_than_looping_forever() -> None:
    net = BayesNet(parents=[1, 0], tables=[np.ones(2) / 2, np.ones((2, 2)) / 2], n_bins=2)
    with pytest.raises(ValueError):
        net.sample(4, np.random.default_rng(0))


def test_sampling_reproduces_the_released_tables() -> None:
    # With no noise the sampler must reproduce the empirical distribution, or every downstream
    # number is measuring the sampler instead of the mechanism.
    rng = np.random.default_rng(6)
    bins = np.column_stack(
        [rng.choice(4, size=4000, p=[0.7, 0.1, 0.1, 0.1]), rng.integers(0, 4, size=4000)]
    ).astype(np.int16)
    net = fit_network(bins, [-1, 0], 4, float("inf"), rng)
    drawn = net.sample(20000, rng)
    empirical = np.bincount(bins[:, 0], minlength=4) / len(bins)
    sampled = np.bincount(drawn[:, 0], minlength=4) / len(drawn)
    assert np.max(np.abs(empirical - sampled)) < 0.02


# --------------------------------------------------------------------------------------
# The release, end to end.
# --------------------------------------------------------------------------------------


def test_the_release_carries_labels_and_both_classes(
    settings: Settings, clean_synth: pd.DataFrame
) -> None:
    from netsentry.features.feature_sets import numeric_features

    features = numeric_features()[:8]
    frame = clean_synth[[*features, BINARY_TARGET]].copy()
    release = synthesize(
        frame,
        features,
        settings=settings,
        epsilon=4.0,
        structure="independent",
        n_rows=500,
        rng=np.random.default_rng(7),
    )
    assert set(release.frame.columns) == {*features, BINARY_TARGET}
    assert release.frame[BINARY_TARGET].nunique() == 2
    assert 400 <= len(release.frame) <= 600


def test_the_budget_description_adds_up() -> None:
    budget = PrivacyBudget(total=4.0, prior=0.2, per_node=3.8 / 76, nodes=76, classes=2)
    assert budget.prior + budget.nodes * budget.per_node == pytest.approx(budget.total)
    assert "parallel across 2 classes" in budget.describe()
    assert "no guarantee" in PrivacyBudget(float("inf"), 0.0, 0.0, 0, 2).describe()
