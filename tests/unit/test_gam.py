"""The additive fitter, and the three claims the report makes about it.

An additive model is only worth reporting if its curves mean what the report says they mean.
Three things get their own test because each of them can be wrong while the score stays fine:
the fitter has to recover a function it was given, correlated features must **split** their
credit rather than each taking all of it (which is what refreshing the gradient per feature
buys), and the pairwise extension has to earn its keep on a function no additive model can
represent at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.models.gam import (
    AdditiveModel,
    Binner,
    CapacityRow,
    EditRow,
    _rank_edits,
    _sigmoid,
    best_rung,
    fit_additive,
    fit_pairs,
    rank_pairs,
)


def _fit(features: np.ndarray, labels: np.ndarray, *, n_bins: int = 16, rounds: int = 120):
    binner = Binner.fit(features, n_bins)
    model = fit_additive(
        binner.transform(features), labels, binner, rounds=rounds, learning_rate=0.2, l2=1.0
    )
    return model, binner


# --------------------------------------------------------------------------------------
# Binning.
# --------------------------------------------------------------------------------------


def test_the_binner_learns_quantile_edges_and_reports_its_own_size() -> None:
    values = np.arange(100, dtype=float).reshape(-1, 1)
    binner = Binner.fit(values, 10)
    assert binner.sizes == [10]
    assert len(np.unique(binner.transform(values))) == 10


def test_values_beyond_the_training_range_fall_into_the_end_bins() -> None:
    """Flat extrapolation is a safety property, so it is pinned rather than assumed."""
    binner = Binner.fit(np.arange(100, dtype=float).reshape(-1, 1), 8)
    extreme = np.array([[-1e9], [1e9]])
    assert binner.transform(extreme).ravel().tolist() == [0, binner.sizes[0] - 1]


def test_a_constant_column_collapses_to_a_single_bin() -> None:
    """Degenerate columns must not produce a bin no row can reach."""
    binner = Binner.fit(np.zeros((50, 1)), 8)
    assert binner.sizes == [1]
    assert binner.transform(np.zeros((3, 1))).ravel().tolist() == [0, 0, 0]


def test_a_zero_inflated_column_opens_no_empty_bins() -> None:
    """Flow features pile up at zero; a dead bin is a flat segment nobody can act on."""
    column = np.concatenate([np.zeros(700), np.linspace(1.0, 10.0, 300)]).reshape(-1, 1)
    binner = Binner.fit(column, 10)
    occupied = np.unique(binner.transform(column))
    assert len(occupied) == binner.sizes[0]


# --------------------------------------------------------------------------------------
# Fitting.
# --------------------------------------------------------------------------------------


def test_the_intercept_is_the_base_rate_when_no_feature_carries_signal() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(2000, 2))
    labels = (rng.random(2000) < 0.2).astype(int)
    model, _ = _fit(features, labels, rounds=1)
    assert _sigmoid(np.array([model.intercept]))[0] == pytest.approx(labels.mean(), abs=0.02)


def test_the_fitter_recovers_a_step_it_was_given() -> None:
    rng = np.random.default_rng(1)
    features = rng.uniform(-3.0, 3.0, size=(6000, 2))
    truth = 3.0 * (features[:, 0] > 0.0) - 1.5
    labels = (rng.random(6000) < _sigmoid(truth)).astype(int)
    model, binner = _fit(features, labels)
    recovered = model.shapes[0][binner.transform(features)[:, 0]]
    left = recovered[features[:, 0] < -0.2].mean()
    right = recovered[features[:, 0] > 0.2].mean()
    assert right - left == pytest.approx(3.0, abs=0.6)
    # ...and the feature with no signal stays inside the noise floor.
    assert model.contribution_range(1) < model.contribution_range(0) / 2.0


def test_correlated_features_split_their_credit_rather_than_doubling_it() -> None:
    """The reason the gradient is refreshed after every feature and not once per round.

    Two identical columns carry one effect. Fitting them against a stale gradient lets each
    take the *whole* effect, and the shapes an operator reads then claim twice the influence
    that exists. Refreshing after each feature makes them share it.
    """
    rng = np.random.default_rng(2)
    base = rng.uniform(-3.0, 3.0, size=6000)
    features = np.column_stack([base, base])
    truth = 2.0 * base
    labels = (rng.random(6000) < _sigmoid(truth)).astype(int)
    model, _ = _fit(features, labels)
    first, second = model.contribution_range(0), model.contribution_range(1)
    combined_swing = first + second
    assert first == pytest.approx(second, rel=0.35)  # neither column monopolises the effect
    assert max(first, second) < combined_swing * 0.75


def test_the_model_is_exactly_the_sum_of_its_tables() -> None:
    """The claim that the model *is* its explanation, checked arithmetically."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(500, 4))
    labels = (rng.random(500) < 0.3).astype(int)
    model, binner = _fit(features, labels, rounds=20)
    bins = binner.transform(features)
    by_hand = model.intercept + sum(
        model.shapes[index][bins[:, index]] for index in range(features.shape[1])
    )
    assert np.allclose(model.margin(bins), by_hand)


# --------------------------------------------------------------------------------------
# Interactions: the capacity an additive model structurally does not have.
# --------------------------------------------------------------------------------------


def _xor(n: int = 6000, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A function no additive model can represent: each marginal is uninformative."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=(n, 3))
    truth = 4.0 * np.sign(features[:, 0] * features[:, 1])
    labels = (rng.random(n) < _sigmoid(truth)).astype(int)
    return features, labels


def test_an_additive_model_cannot_fit_an_interaction() -> None:
    features, labels = _xor()
    model, binner = _fit(features, labels)
    scores = model.predict_proba(binner.transform(features))
    assert np.mean((scores >= 0.5).astype(int) == labels) < 0.6  # barely better than a coin


def test_the_pair_ranking_finds_the_interacting_pair_first() -> None:
    features, labels = _xor()
    model, binner = _fit(features, labels)
    ranked = rank_pairs(model, binner.transform(features), labels, candidates=[0, 1, 2])
    assert ranked[0][0] == (0, 1)


def test_one_pairwise_term_recovers_what_the_additive_model_could_not() -> None:
    features, labels = _xor()
    model, binner = _fit(features, labels)
    bins = binner.transform(features)
    extended = fit_pairs(model, bins, labels, [(0, 1)], rounds=60, learning_rate=0.2, l2=1.0)
    accuracy = np.mean((extended.predict_proba(bins) >= 0.5).astype(int) == labels)
    assert accuracy > 0.85


# --------------------------------------------------------------------------------------
# Editing.
# --------------------------------------------------------------------------------------


def test_clamping_a_bin_changes_only_the_flows_in_that_bin() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(800, 3))
    labels = (rng.random(800) < 0.3).astype(int)
    model, binner = _fit(features, labels, rounds=30)
    bins = binner.transform(features)
    before = model.predict_proba(bins)
    after = model.clamp(1, 4).predict_proba(bins)
    touched = bins[:, 1] == 4
    assert np.allclose(before[~touched], after[~touched])
    assert touched.sum() > 0
    assert not np.allclose(before[touched], after[touched])


def test_clamping_leaves_the_original_model_untouched() -> None:
    """`clamp` returns a copy: an operator experimenting must not mutate what is deployed."""
    rng = np.random.default_rng(6)
    features = rng.normal(size=(300, 2))
    labels = (rng.random(300) < 0.4).astype(int)
    model, _ = _fit(features, labels, rounds=10)
    original = model.shapes[0].copy()
    model.clamp(0, 2)
    assert np.array_equal(model.shapes[0], original)


def test_proposed_edits_report_the_counts_they_actually_clear() -> None:
    """The ranking is only honest if its numbers survive an independent recomputation."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3000, 4))
    labels = (rng.random(3000) < (0.15 + 0.3 * (features[:, 0] > 1.0))).astype(int)
    model, binner = _fit(features, labels, rounds=40)
    bins = binner.transform(features)
    threshold = float(np.quantile(model.predict_proba(bins), 0.9))
    proposals = _rank_edits(model, bins, labels, threshold, top_n=3, min_removed=1)
    assert proposals
    for index, bin_index, removed, lost in proposals:
        before = model.predict_proba(bins) >= threshold
        after = model.clamp(index, bin_index).predict_proba(bins) >= threshold
        cleared = before & ~after
        assert int(np.sum(cleared & (labels == 0))) == removed
        assert int(np.sum(cleared & (labels == 1))) == lost


def test_a_bin_that_lowers_risk_is_never_proposed() -> None:
    """Clamping a protective region to zero can only *raise* scores, so it cannot be an edit."""
    rng = np.random.default_rng(8)
    features = rng.normal(size=(2000, 3))
    labels = (rng.random(2000) < 0.3).astype(int)
    model, binner = _fit(features, labels, rounds=30)
    bins = binner.transform(features)
    threshold = float(np.quantile(model.predict_proba(bins), 0.8))
    for index, bin_index, _, _ in _rank_edits(
        model, bins, labels, threshold, top_n=10, min_removed=1
    ):
        assert model.shapes[index][bin_index] > 0.0


# --------------------------------------------------------------------------------------
# The records the report reads from.
# --------------------------------------------------------------------------------------


def test_the_exchange_rate_prices_alarms_against_missed_attacks() -> None:
    row = EditRow(
        budget=0.01,
        feature="x",
        low=0.0,
        high=1.0,
        validation_removed=10,
        validation_lost=1,
        removed=40,
        lost=2,
        benign_share=0.05,
    )
    assert row.exchange_rate == pytest.approx(20.0)
    assert not row.free


def test_a_costless_edit_is_flagged_as_free() -> None:
    row = EditRow(
        budget=0.01,
        feature="x",
        low=0.0,
        high=1.0,
        validation_removed=5,
        validation_lost=0,
        removed=9,
        lost=0,
        benign_share=0.05,
    )
    assert row.free and row.exchange_rate == pytest.approx(9.0)


def test_the_ladder_reports_the_rung_each_column_would_have_chosen() -> None:
    ladder = [
        CapacityRow("small", 10, train=0.4, validation=0.5, test=0.3),
        CapacityRow("medium", 20, train=0.6, validation=0.7, test=0.5),
        CapacityRow("large", 40, train=0.9, validation=0.6, test=0.55),
    ]
    validation_pick = best_rung(ladder, "validation")
    test_pick = best_rung(ladder, "test")
    assert validation_pick is not None and validation_pick.label == "medium"
    assert test_pick is not None and test_pick.label == "large"
    assert best_rung([], "test") is None


def test_an_empty_model_reports_a_zero_swing() -> None:
    model = AdditiveModel(intercept=0.0, shapes=[np.array([])], binner=Binner(edges=[]))
    assert model.contribution_range(0) == 0.0
