"""SISA machine unlearning: sharding, deletion cost, and the exact-forgetting guarantee.

The load-bearing claim is *exactness* — after unlearning a batch, the ensemble must be identical
to one trained from scratch on the surviving rows. That is a property of the sharding + isolated
training, provable on a tiny model here without the real dataset, alongside the shard-assignment
determinism, the shards-touched accounting, and the coupon-collector cost expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config import Settings
from netsentry.training.unlearn import (
    SisaEnsemble,
    assign_shards,
    expected_shards_touched,
    shards_touched,
)


def test_assign_shards_is_balanced_and_deterministic() -> None:
    a = assign_shards(100, 4, seed=7)
    b = assign_shards(100, 4, seed=7)
    assert np.array_equal(a, b)  # same seed -> same assignment (exactness needs this)
    counts = np.bincount(a, minlength=4)
    assert counts.tolist() == [25, 25, 25, 25]  # balanced


def test_assign_shards_changes_with_shard_count() -> None:
    assert not np.array_equal(assign_shards(60, 3, seed=1), assign_shards(60, 6, seed=1))


def test_shards_touched_returns_the_distinct_holding_shards() -> None:
    shard_of = np.array([0, 1, 2, 0, 1, 2, 3])
    # Deleting rows 0 (shard 0), 4 (shard 1), 6 (shard 3) touches shards {0, 1, 3}.
    assert shards_touched(shard_of, np.array([0, 4, 6])) == [0, 1, 3]


def test_expected_shards_touched_matches_the_coupon_collector_formula() -> None:
    # One deletion always hits exactly one shard.
    assert np.isclose(expected_shards_touched(8, 1), 1.0)
    # S(1 - (1 - 1/S)^k) by hand for S=4, k=2: 4(1 - (3/4)^2) = 4(1 - 9/16) = 1.75.
    assert np.isclose(expected_shards_touched(4, 2), 1.75)
    # Enough deletions saturate every shard.
    assert expected_shards_touched(4, 10000) > 3.99


def test_expected_shards_touched_is_monotone_in_batch_size() -> None:
    vals = [expected_shards_touched(8, k) for k in (1, 5, 20, 100)]
    assert vals == sorted(vals)  # more deletions -> more shards, never fewer


@pytest.fixture
def tiny_settings() -> Settings:
    s = Settings()
    s.mlflow.enabled = False
    s.supervised.n_estimators = 15
    return s


def _blobs(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Two separable Gaussian blobs -> a learnable binary problem for the submodels."""
    x_pos = rng.normal(loc=2.0, size=(n // 2, 4))
    x_neg = rng.normal(loc=-2.0, size=(n - n // 2, 4))
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n - n // 2, dtype=int)])
    order = rng.permutation(n)
    return x[order], y[order]


def test_unlearning_is_exactly_a_fresh_ensemble_without_the_deleted_rows(
    tiny_settings: Settings,
) -> None:
    # The SISA guarantee: unlearn(delete) == train-from-scratch(surviving), bit for bit.
    rng = np.random.default_rng(0)
    x, y = _blobs(rng, 400)
    x_val, y_val = _blobs(rng, 120)
    n_shards = 4
    shard_of = assign_shards(len(y), n_shards, tiny_settings.seed)
    delete_idx = rng.choice(len(y), size=25, replace=False)
    keep = np.ones(len(y), dtype=bool)
    keep[delete_idx] = False

    trained = SisaEnsemble(tiny_settings, n_shards).fit(x, y, shard_of, (x_val, y_val))
    trained.unlearn(shards_touched(shard_of, delete_idx), x, y, shard_of, keep, (x_val, y_val))

    fresh = SisaEnsemble(tiny_settings, n_shards)
    for shard in range(n_shards):
        mask = (shard_of == shard) & keep
        fresh.fit_shard(shard, x[mask], y[mask], (x_val, y_val))

    diff = np.max(np.abs(trained.attack_proba(x_val) - fresh.attack_proba(x_val)))
    assert diff < 1e-9  # exact: no residue of the deleted rows survives


def test_unlearn_leaves_untouched_shards_byte_identical(tiny_settings: Settings) -> None:
    # Only the shards holding a deleted row may change; the rest must be the same object-state.
    rng = np.random.default_rng(1)
    x, y = _blobs(rng, 320)
    x_val, y_val = _blobs(rng, 100)
    n_shards = 4
    shard_of = assign_shards(len(y), n_shards, tiny_settings.seed)
    ens = SisaEnsemble(tiny_settings, n_shards).fit(x, y, shard_of, (x_val, y_val))

    # Delete only rows that live in shard 0.
    in_shard0 = np.flatnonzero(shard_of == 0)
    delete_idx = in_shard0[:5]
    keep = np.ones(len(y), dtype=bool)
    keep[delete_idx] = False
    untouched_preds = {s: ens.models[s].predict_proba(x_val) for s in range(n_shards) if s != 0}
    ens.unlearn([0], x, y, shard_of, keep, (x_val, y_val))
    for s in range(1, n_shards):
        assert np.array_equal(ens.models[s].predict_proba(x_val), untouched_preds[s])
