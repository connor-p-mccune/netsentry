"""The streaming learner and its drift detector, tested against their own guarantees.

Neither of these is checked against a reference implementation, because the point of both is a
*property*: the tree must not split before its confidence bound allows it, and the window must
hold its false-alarm rate on stationary data while still catching a change quickly. Those are
the things a test can pin, and they are what would break silently if the arithmetic drifted.
"""

from __future__ import annotations

import numpy as np

from netsentry.models.hoeffding import (
    ADWIN,
    HoeffdingTree,
    entropy,
    hoeffding_bound,
    information_gain,
)

# --------------------------------------------------------------------------------------
# Split criterion and the bound
# --------------------------------------------------------------------------------------


def test_entropy_is_zero_for_a_pure_node_and_one_bit_for_a_balanced_one() -> None:
    assert entropy(np.array([10.0, 0.0])) == 0.0
    assert np.isclose(entropy(np.array([5.0, 5.0])), 1.0)
    assert np.isclose(entropy(np.array([1.0, 1.0, 1.0, 1.0])), 2.0)


def test_a_perfect_split_recovers_the_whole_parent_entropy() -> None:
    parent = np.array([50.0, 50.0])
    gain = information_gain(parent, np.array([50.0, 0.0]), np.array([0.0, 50.0]))
    assert np.isclose(gain, 1.0)


def test_a_useless_split_gains_nothing() -> None:
    parent = np.array([50.0, 50.0])
    gain = information_gain(parent, np.array([25.0, 25.0]), np.array([25.0, 25.0]))
    assert np.isclose(gain, 0.0)


def test_hoeffding_bound_matches_its_closed_form_and_shrinks_with_n() -> None:
    # The bound is the whole justification for splitting early, so it is checked numerically
    # rather than only for monotonicity.
    expected = np.sqrt(1.0 * np.log(1.0 / 1e-6) / (2.0 * 200))
    assert np.isclose(hoeffding_bound(1.0, 1e-6, 200), expected)
    assert hoeffding_bound(1.0, 1e-6, 800) < hoeffding_bound(1.0, 1e-6, 200)
    assert np.isclose(
        hoeffding_bound(1.0, 1e-6, 800), hoeffding_bound(1.0, 1e-6, 200) / 2, rtol=1e-9
    )
    assert hoeffding_bound(1.0, 1e-6, 0) == float("inf")


# --------------------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------------------


def _separable(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((n, 4))
    y = (x[:, 0] > 0.7).astype(int)
    return x, y


def test_the_tree_does_not_split_before_its_grace_period() -> None:
    rng = np.random.default_rng(0)
    x, y = _separable(150, rng)
    tree = HoeffdingTree(n_features=4, grace_period=200)
    tree.learn_many(x, y)
    assert tree.n_splits == 0
    assert tree.n_nodes() == 1


def test_the_tree_grows_on_a_separable_stream_and_ranks_it() -> None:
    rng = np.random.default_rng(1)
    x, y = _separable(3000, rng)
    tree = HoeffdingTree(n_features=4, grace_period=100, leaf_prediction="mc")
    tree.learn_many(x[:2000], y[:2000])
    scores = np.array([tree.score_one(row) for row in x[2000:]])
    assert tree.n_splits >= 1
    positives = scores[y[2000:] == 1]
    negatives = scores[y[2000:] == 0]
    assert positives.mean() > negatives.mean() + 0.3


def test_memory_is_bounded_by_the_structure_not_the_stream() -> None:
    """The property that makes a streaming learner deployable: no dependence on stream length."""
    rng = np.random.default_rng(2)
    x, y = _separable(6000, rng)
    tree = HoeffdingTree(n_features=4, grace_period=200, max_depth=3)
    tree.learn_many(x, y)
    assert tree.n_nodes() == 2 * tree.n_splits + 1
    assert tree.n_leaves() == tree.n_splits + 1
    assert tree.memory_bytes() == tree.n_leaves() * 8 * (2 * 4 * 2 + 2 * 4 + 2)


def test_depth_limit_is_respected() -> None:
    rng = np.random.default_rng(3)
    x, y = _separable(8000, rng)
    shallow = HoeffdingTree(n_features=4, grace_period=100, max_depth=1)
    shallow.learn_many(x, y)
    assert shallow.n_splits <= 1


def test_posteriors_are_probability_vectors() -> None:
    rng = np.random.default_rng(4)
    x, y = _separable(1200, rng)
    for leaf_rule in ("mc", "nb"):
        tree = HoeffdingTree(n_features=4, grace_period=200, leaf_prediction=leaf_rule)
        tree.learn_many(x, y)
        proba = tree.predict_proba(x[:50])
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all()


def test_an_untouched_tree_predicts_the_uniform_prior() -> None:
    tree = HoeffdingTree(n_features=3)
    assert np.allclose(tree.predict_proba_one(np.zeros(3)), [0.5, 0.5])


def test_the_same_stream_builds_the_same_tree() -> None:
    rng = np.random.default_rng(5)
    x, y = _separable(2000, rng)
    first = HoeffdingTree(n_features=4, grace_period=100).learn_many(x, y)
    second = HoeffdingTree(n_features=4, grace_period=100).learn_many(x, y)
    assert first.n_splits == second.n_splits
    assert np.allclose(first.predict_proba(x[:20]), second.predict_proba(x[:20]))


# --------------------------------------------------------------------------------------
# ADWIN
# --------------------------------------------------------------------------------------


def test_adwin_does_not_fire_on_a_stationary_stream() -> None:
    rng = np.random.default_rng(6)
    detector = ADWIN(delta=0.002)
    fired = sum(detector.update(float(v)) for v in rng.binomial(1, 0.5, 3000))
    assert fired == 0
    assert detector.width == 3000  # nothing was discarded
    assert abs(detector.estimation - 0.5) < 0.05


def test_adwin_detects_an_abrupt_change_and_shrinks_its_window() -> None:
    rng = np.random.default_rng(7)
    detector = ADWIN(delta=0.002)
    for value in rng.binomial(1, 0.2, 1000):
        detector.update(float(value))
    detected_at = None
    width_before = width_after = 0
    for i, value in enumerate(rng.binomial(1, 0.8, 1000)):
        previous = detector.width
        if detector.update(float(value)) and detected_at is None:
            detected_at, width_before, width_after = i, previous, detector.width
    assert detected_at is not None and detected_at < 200
    assert width_after < width_before  # the stale half was dropped at the cut
    assert detector.width < 1500  # and the window never regrows to include the old regime
    assert detector.estimation > 0.6  # the estimate followed the new regime


def test_adwin_memory_grows_logarithmically() -> None:
    """The exponential histogram is what makes the window affordable: buckets, not values."""
    rng = np.random.default_rng(8)
    detector = ADWIN()
    for value in rng.random(20_000):
        detector.update(float(value))
    assert detector.width == 20_000
    assert detector.n_buckets() < 120  # against 20,000 stored values


def test_adwin_reset_forgets_everything() -> None:
    detector = ADWIN()
    for value in np.linspace(0, 1, 500):
        detector.update(float(value))
    detector.reset()
    assert detector.width == 0 and detector.estimation == 0.0 and detector.n_buckets() == 0
