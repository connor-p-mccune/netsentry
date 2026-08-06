"""Federated averaging: aggregation, clipping, the DP noise step, and the skew measure.

FedAvg's correctness is entirely in its aggregation algebra — a sample-size-weighted mean of
site weights, an L2 clip that bounds one site's influence, and Gaussian noise scaled to that
bound. Each is checked in closed form here, along with the property that makes the whole
thing worth building: sites that hold complementary data produce an average that beats any
one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.training.federated import (
    Weights,
    add_aggregate_noise,
    clip_to_norm,
    federated_average,
    initial_weights,
    label_skew,
    local_train,
)


def test_initial_weights_are_zero_so_every_site_starts_identical() -> None:
    w = initial_weights(4)
    assert w.coef.tolist() == [0.0] * 4 and w.intercept == 0.0


def test_federated_average_weights_sites_by_sample_count() -> None:
    a = Weights(np.array([0.0, 0.0]), 0.0)
    b = Weights(np.array([1.0, 2.0]), 4.0)
    # 300 rows at b against 100 at a -> b carries three quarters of the average.
    out = federated_average([a, b], [100, 300])
    assert np.allclose(out.coef, [0.75, 1.5])
    assert out.intercept == pytest.approx(3.0)


def test_federated_average_of_identical_sites_is_that_model() -> None:
    w = Weights(np.array([0.3, -0.7]), 1.1)
    out = federated_average([w.copy(), w.copy(), w.copy()], [10, 20, 30])
    assert np.allclose(out.coef, w.coef) and out.intercept == pytest.approx(w.intercept)


def test_a_site_with_no_rows_cannot_move_the_average() -> None:
    a = Weights(np.array([1.0]), 1.0)
    b = Weights(np.array([99.0]), 99.0)
    out = federated_average([a, b], [50, 0])
    assert np.allclose(out.coef, a.coef) and out.intercept == pytest.approx(a.intercept)


def test_federated_average_rejects_an_empty_federation() -> None:
    with pytest.raises(ValueError, match="no site updates"):
        federated_average([], [])


def test_clipping_leaves_a_small_update_untouched() -> None:
    w = Weights(np.array([0.3, 0.4]), 0.0)  # norm 0.5
    assert np.allclose(clip_to_norm(w, 5.0).coef, w.coef)


def test_clipping_projects_a_large_update_onto_the_ball() -> None:
    w = Weights(np.array([3.0, 4.0]), 0.0)  # norm 5
    out = clip_to_norm(w, 1.0)
    assert np.sqrt(np.sum(out.coef**2) + out.intercept**2) == pytest.approx(1.0)
    # Direction is preserved; only the magnitude is bounded.
    assert np.allclose(out.coef, [0.6, 0.8])


def test_clipping_a_zero_update_does_not_divide_by_zero() -> None:
    out = clip_to_norm(Weights(np.zeros(3), 0.0), 1.0)
    assert np.allclose(out.coef, 0.0)


def test_no_noise_multiplier_returns_the_aggregate_unchanged() -> None:
    w = Weights(np.array([1.0, 2.0]), 3.0)
    out = add_aggregate_noise(
        w, clip_norm=1.0, noise_multiplier=0.0, n_sites=3, rng=np.random.default_rng(0)
    )
    assert np.allclose(out.coef, w.coef) and out.intercept == w.intercept


def test_noise_scale_follows_the_per_site_sensitivity() -> None:
    # Sensitivity is clip_norm / n_sites, so more sites means less noise per site's worth
    # of influence -- the reason federation and DP compose well at scale.
    rng = np.random.default_rng(0)
    few = add_aggregate_noise(
        Weights(np.zeros(4000), 0.0), clip_norm=1.0, noise_multiplier=1.0, n_sites=2, rng=rng
    )
    many = add_aggregate_noise(
        Weights(np.zeros(4000), 0.0), clip_norm=1.0, noise_multiplier=1.0, n_sites=100, rng=rng
    )
    assert few.coef.std() > many.coef.std() * 10


def test_label_skew_is_zero_when_every_site_matches_the_global_prior() -> None:
    assert label_skew([0.25, 0.25, 0.25], 0.25) == 0.0


def test_label_skew_grows_as_sites_disagree() -> None:
    mild = label_skew([0.2, 0.3], 0.25)
    severe = label_skew([0.0, 0.5], 0.25)
    assert severe > mild > 0


def test_label_skew_of_an_empty_federation_is_zero() -> None:
    assert label_skew([], 0.3) == 0.0


def test_local_training_on_an_empty_site_returns_the_global_model() -> None:
    w = Weights(np.array([0.5, -0.5]), 0.1)
    out = local_train(
        w,
        np.zeros((0, 2)),
        np.zeros(0),
        epochs=3,
        batch_size=8,
        learning_rate=0.1,
        l2=0.0,
        seed=0,
    )
    assert np.allclose(out.coef, w.coef) and out.intercept == pytest.approx(w.intercept)


def test_local_training_is_deterministic_under_a_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.integers(0, 2, size=200).astype(float)
    kwargs = {"epochs": 2, "batch_size": 32, "learning_rate": 0.1, "l2": 0.0, "seed": 7}
    a = local_train(initial_weights(3), x, y, **kwargs)  # type: ignore[arg-type]
    b = local_train(initial_weights(3), x, y, **kwargs)  # type: ignore[arg-type]
    assert np.allclose(a.coef, b.coef) and a.intercept == pytest.approx(b.intercept)


def test_averaging_complementary_sites_beats_either_alone() -> None:
    # The premise of the whole exercise. The true rule is x0 + x1 > 0, but site A's traffic
    # only ever exercises feature 0 (its feature 1 is flat) and site B's only feature 1.
    # Each learns half the rule and is blind to the other half; the average learns both.
    rng = np.random.default_rng(3)
    n = 600
    xa = np.column_stack([rng.normal(size=n), np.zeros(n)])
    xb = np.column_stack([np.zeros(n), rng.normal(size=n)])
    ya = (xa.sum(axis=1) > 0).astype(float)
    yb = (xb.sum(axis=1) > 0).astype(float)
    kwargs = {"epochs": 8, "batch_size": 64, "learning_rate": 0.5, "l2": 0.0}
    wa = local_train(initial_weights(2), xa, ya, seed=1, **kwargs)  # type: ignore[arg-type]
    wb = local_train(initial_weights(2), xb, yb, seed=2, **kwargs)  # type: ignore[arg-type]
    avg = federated_average([wa, wb], [n, n])

    x_test = rng.normal(size=(3000, 2))
    y_test = (x_test.sum(axis=1) > 0).astype(float)

    def accuracy(w: Weights) -> float:
        return float(((w.scores(x_test) >= 0.5).astype(float) == y_test).mean())

    assert accuracy(avg) > max(accuracy(wa), accuracy(wb)) + 0.1
