"""The self-supervised pretext tasks, and the arithmetic the report leans on.

The corruption operator is the piece worth testing hard: both pretext tasks are defined by it,
and a corruption that samples replacements from the wrong axis produces a task that is easy for
the wrong reason (marginally implausible rows) while still training, still converging, and
still yielding a plausible-looking curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.training.pretrain import certification_floor, corrupt
from netsentry.utils.optional import is_available


def test_corruption_replaces_about_the_configured_fraction() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 10)).astype(np.float32)
    _, mask = corrupt(x, 0.3, rng)
    assert 0.28 < float(mask.mean()) < 0.32


def test_corruption_draws_replacements_from_the_same_column() -> None:
    # Each column holds a disjoint range of values. If a replacement ever came from another
    # column the corrupted row would be marginally impossible and the pretext task trivial.
    rng = np.random.default_rng(1)
    x = np.column_stack([np.full(500, 10.0), np.full(500, 20.0), np.full(500, 30.0)]).astype(
        np.float32
    )
    x += rng.random(x.shape).astype(np.float32) * 0.5
    corrupted, _ = corrupt(x, 0.9, rng)
    assert corrupted[:, 0].max() < 11.0
    assert corrupted[:, 1].min() >= 20.0 and corrupted[:, 1].max() < 21.0
    assert corrupted[:, 2].min() >= 30.0


def test_uncorrupted_entries_are_left_exactly_alone() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 6)).astype(np.float32)
    corrupted, mask = corrupt(x, 0.4, rng)
    untouched = mask == 0
    assert np.array_equal(corrupted[untouched], x[untouched])


def test_zero_corruption_is_the_identity() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 4)).astype(np.float32)
    corrupted, mask = corrupt(x, 0.0, rng)
    assert np.array_equal(corrupted, x)
    assert mask.sum() == 0


def test_the_certification_floor_matches_the_order_statistic_bound() -> None:
    # n >= log(delta) / log(1 - alpha). At a 1% budget and 95% confidence that is 299 benign
    # flows; at 0.1% it is nearly 3,000 -- the arithmetic behind the report's caveat.
    assert certification_floor(0.01, 0.95) == 299
    assert certification_floor(0.001, 0.95) == 2995
    assert certification_floor(0.01, 0.99) > certification_floor(0.01, 0.95)


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_the_pretext_tasks_produce_embeddings_of_the_declared_width() -> None:
    from netsentry.config import Settings
    from netsentry.training.pretrain import (
        pca_encoder,
        pretrain_contrastive,
        pretrain_masked,
        random_encoder,
    )

    cfg = Settings().pretrain
    cfg.epochs = 1
    cfg.embedding_dim = 8
    cfg.hidden_sizes = [16]
    rng = np.random.default_rng(4)
    pool = rng.standard_normal((256, 12)).astype(np.float32)
    for result in (
        pretrain_masked(pool, cfg, 0, "masked"),
        pretrain_contrastive(pool, cfg, 0, "contrastive"),
        random_encoder(12, cfg, 0, "random"),
        pca_encoder(pool, cfg, 0, "pca"),
    ):
        embedded = result.embed(pool[:16])
        assert embedded.shape == (16, 8)
        assert np.isfinite(embedded).all()


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_masked_pretraining_actually_reduces_its_own_loss() -> None:
    # A pretext task that does not learn its own objective cannot be said to have learned a
    # representation, whatever the downstream curve does.
    from netsentry.config import Settings
    from netsentry.training.pretrain import pretrain_masked

    cfg = Settings().pretrain
    cfg.embedding_dim = 8
    cfg.hidden_sizes = [16]
    rng = np.random.default_rng(5)
    base = rng.standard_normal((512, 3)).astype(np.float32)
    pool = np.hstack([base, base * 2.0, base - 1.0]).astype(np.float32)  # learnable structure

    cfg.epochs = 1
    early = pretrain_masked(pool, cfg, 0, "early").final_loss
    cfg.epochs = 15
    late = pretrain_masked(pool, cfg, 0, "late").final_loss
    assert late < early


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_pretraining_is_reproducible_from_the_seed() -> None:
    from netsentry.config import Settings
    from netsentry.training.pretrain import pretrain_masked

    cfg = Settings().pretrain
    cfg.epochs = 2
    cfg.embedding_dim = 8
    cfg.hidden_sizes = [16]
    pool = np.random.default_rng(6).standard_normal((256, 10)).astype(np.float32)
    first = pretrain_masked(pool, cfg, 42, "a").embed(pool[:8])
    second = pretrain_masked(pool, cfg, 42, "b").embed(pool[:8])
    assert np.array_equal(first, second)
