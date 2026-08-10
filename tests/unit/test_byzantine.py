"""Byzantine aggregation: the vulnerability of the mean, and the bounded influence of the fixes.

The headline claim is that a single participant can move FedAvg anywhere, so the first test
constructs the update that does it — proving the vulnerability rather than describing it.
The rest pin the property each defence is chosen for: a minority of arbitrary values,
including infinities, must not move the result at all.
"""

from __future__ import annotations

import numpy as np

from netsentry.training.byzantine import (
    aggregate,
    coordinate_median,
    gaussian_update,
    krum,
    sign_flip,
    trimmed_mean,
)
from netsentry.training.federated import Weights, federated_average


def _w(*values: float) -> Weights:
    """A tiny two-parameter update: one coefficient plus an intercept."""
    return Weights(np.array(values[:-1], dtype=float), float(values[-1]))


HONEST = [_w(1.0, 0.0), _w(1.1, 0.1), _w(0.9, -0.1), _w(1.0, 0.05), _w(1.05, 0.0)]


def test_one_site_can_move_fedavg_to_any_target_it_likes() -> None:
    """The vulnerability, constructed. Averaging has no bounded influence, so solve for it."""
    honest = HONEST[:4]
    target = 42.0
    n = len(honest) + 1
    # Choose the liar's coefficient so the plain mean lands exactly on the target.
    needed = target * n - sum(float(w.coef[0]) for w in honest)
    liar = _w(needed, 0.0)
    result = federated_average([*honest, liar], [1] * n)
    assert np.isclose(float(result.coef[0]), target)


def test_the_median_ignores_that_same_site_entirely() -> None:
    liar = _w(1e9, -1e9)
    with_liar = coordinate_median([*HONEST, liar])
    without = coordinate_median([*HONEST, HONEST[0]])
    assert abs(float(with_liar.coef[0]) - float(without.coef[0])) < 0.2


def test_the_median_survives_an_infinite_update() -> None:
    """A mean returns NaN here; the median does not notice."""
    result = coordinate_median([*HONEST, _w(np.inf, -np.inf)])
    assert np.isfinite(result.coef[0]) and 0.9 <= float(result.coef[0]) <= 1.1


def test_trimmed_mean_discards_the_extremes_from_both_ends() -> None:
    updates = [_w(0.0, 0.0), _w(1.0, 0.0), _w(2.0, 0.0), _w(3.0, 0.0), _w(100.0, 0.0)]
    trimmed = trimmed_mean(updates, trim=1)
    # Drops 0.0 and 100.0, averaging 1, 2, 3.
    assert np.isclose(float(trimmed.coef[0]), 2.0)


def test_trimmed_mean_with_no_trimming_is_exactly_the_plain_mean() -> None:
    plain = trimmed_mean(HONEST, trim=0)
    mean = federated_average(HONEST, [1] * len(HONEST))
    assert np.allclose(plain.coef, mean.coef)


def test_trimmed_mean_never_trims_away_every_update() -> None:
    result = trimmed_mean(HONEST, trim=99)
    assert np.isfinite(result.coef[0])


def test_krum_elects_an_honest_update_when_the_liars_do_not_cluster() -> None:
    liars = [_w(50.0, 50.0), _w(-70.0, 30.0)]
    chosen = krum([*HONEST, *liars], n_byzantine=2)
    # The elected vector must be one of the honest ones, not an average of anything.
    assert any(np.allclose(chosen.coef, w.coef) and chosen.intercept == w.intercept for w in HONEST)


def test_krum_returns_an_actual_submitted_update_not_a_blend() -> None:
    chosen = krum(HONEST, n_byzantine=1)
    assert any(np.allclose(chosen.coef, w.coef) for w in HONEST)


def test_krum_falls_back_when_there_are_too_few_sites_to_say_anything() -> None:
    # n - f - 2 < 1: the rule has no neighbours to score against, so it degrades to a median.
    chosen = krum(HONEST[:3], n_byzantine=2)
    assert np.isfinite(chosen.coef[0])


def test_sign_flip_negates_and_amplifies_the_honest_update() -> None:
    flipped = sign_flip(_w(2.0, -1.0), scale=10.0)
    assert np.allclose(flipped.coef, [-20.0]) and flipped.intercept == 10.0


def test_gaussian_update_is_reproducible_from_its_seed() -> None:
    a = gaussian_update(4, 1.0, np.random.default_rng(0))
    b = gaussian_update(4, 1.0, np.random.default_rng(0))
    assert np.allclose(a.coef, b.coef) and a.intercept == b.intercept


def test_aggregate_dispatches_to_the_named_rule() -> None:
    sizes = [1] * len(HONEST)
    assert np.allclose(
        aggregate("coordinate median", HONEST, sizes, 1, 1).coef, coordinate_median(HONEST).coef
    )
    assert np.allclose(
        aggregate("trimmed mean", HONEST, sizes, 1, 1).coef, trimmed_mean(HONEST, 1).coef
    )
    assert np.allclose(
        aggregate("FedAvg (mean)", HONEST, sizes, 1, 1).coef,
        federated_average(HONEST, sizes).coef,
    )


def test_every_robust_rule_resists_a_minority_of_arbitrary_updates() -> None:
    """The shared property: a minority of liars cannot move the result far, whatever they send."""
    rng = np.random.default_rng(1)
    honest = [Weights(rng.normal(1.0, 0.05, size=6), 0.0) for _ in range(9)]
    liars = [Weights(rng.normal(0.0, 500.0, size=6), 100.0) for _ in range(4)]
    clean_median = coordinate_median(honest)
    for name in ("coordinate median", "trimmed mean", "Krum"):
        attacked = aggregate(name, [*honest, *liars], [1] * 13, 4, 4)
        assert float(np.max(np.abs(attacked.coef - clean_median.coef))) < 0.5
