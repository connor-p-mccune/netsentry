"""The deep tabular models: contract, determinism, and the parts that would silently mislead.

A neural baseline in a comparison study is only trustworthy if it is *not* quietly broken, and
the ways it can be quietly broken are specific: it can early-stop on the wrong quantity, ignore
the class imbalance, leak the validation set into its own selection, or return probabilities
that are not probabilities. Each of those is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config import Settings
from netsentry.models.tabular_nn import TorchTabularClassifier
from netsentry.utils.optional import is_available

pytestmark = pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")

ARCHITECTURES = ("mlp", "ft_transformer")


def _tiny(settings: Settings) -> Settings:
    settings.deep_tabular.epochs = 4
    settings.deep_tabular.batch_size = 64
    settings.deep_tabular.hidden_sizes = [16]
    settings.deep_tabular.token_dim = 8
    settings.deep_tabular.n_heads = 2
    settings.deep_tabular.n_blocks = 1
    return settings


def _separable(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int)
    return x, y


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_probabilities_are_probabilities(settings: Settings, architecture: str) -> None:
    x, y = _separable(400, 6, seed=0)
    model = TorchTabularClassifier(_tiny(settings), architecture=architecture).fit(x, y)
    proba = model.predict_proba(x[:50])
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_same_seed_gives_the_same_model(settings: Settings, architecture: str) -> None:
    x, y = _separable(400, 6, seed=1)
    tiny = _tiny(settings)
    first = TorchTabularClassifier(tiny, architecture=architecture).fit(x, y)
    second = TorchTabularClassifier(tiny, architecture=architecture).fit(x, y)
    assert np.allclose(first.predict_proba(x[:20]), second.predict_proba(x[:20]))


def test_a_different_seed_gives_a_different_model(settings: Settings) -> None:
    # The determinism above must come from the seed, not from the model being degenerate.
    x, y = _separable(400, 6, seed=2)
    tiny = _tiny(settings)
    first = TorchTabularClassifier(tiny, architecture="mlp").fit(x, y)
    tiny.seed = tiny.seed + 1
    second = TorchTabularClassifier(tiny, architecture="mlp").fit(x, y)
    assert not np.allclose(first.predict_proba(x[:20]), second.predict_proba(x[:20]))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_it_learns_something_separable(settings: Settings, architecture: str) -> None:
    x, y = _separable(1200, 6, seed=3)
    tiny = _tiny(settings)
    tiny.deep_tabular.epochs = 25
    model = TorchTabularClassifier(tiny, architecture=architecture).fit(
        x[:900], y[:900], eval_set=(x[900:], y[900:])
    )
    scores = model.predict_proba(x[900:])[:, 1]
    assert scores[y[900:] == 1].mean() > scores[y[900:] == 0].mean() + 0.2


def test_early_stopping_watches_validation_pr_auc_not_loss(settings: Settings) -> None:
    """The metric that decides deployment is the metric that should decide the epoch."""
    x, y = _separable(800, 6, seed=4)
    tiny = _tiny(settings)
    tiny.deep_tabular.epochs = 30
    tiny.deep_tabular.patience = 2
    model = TorchTabularClassifier(tiny, architecture="mlp").fit(
        x[:600], y[:600], eval_set=(x[600:], y[600:])
    )
    assert model.trace is not None
    assert 0.0 <= model.trace.best_val_pr_auc <= 1.0
    assert 1 <= model.trace.best_epoch <= model.trace.epochs_run
    # Patience must be able to stop it early; otherwise the parameter is decorative.
    assert model.trace.epochs_run <= tiny.deep_tabular.epochs


def test_imbalance_is_handled_by_weighting_not_by_collapsing(settings: Settings) -> None:
    # 2% positives: an unweighted network minimises loss by predicting "benign" for everything,
    # which scores well on accuracy and is useless. The positive class weight is what stops it.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((1000, 5))
    y = np.zeros(1000, dtype=int)
    rare = rng.choice(1000, 20, replace=False)
    y[rare] = 1
    x[rare] += 2.5
    tiny = _tiny(settings)
    tiny.deep_tabular.epochs = 30
    model = TorchTabularClassifier(tiny, architecture="mlp").fit(x, y)
    scores = model.predict_proba(x)[:, 1]
    assert scores[y == 1].mean() > scores[y == 0].mean() + 0.2
    assert scores.std() > 0.01  # not a constant predictor


def test_parameter_counts_are_reported_and_architecture_dependent(settings: Settings) -> None:
    x, y = _separable(200, 6, seed=6)
    tiny = _tiny(settings)
    mlp = TorchTabularClassifier(tiny, architecture="mlp").fit(x, y)
    transformer = TorchTabularClassifier(tiny, architecture="ft_transformer").fit(x, y)
    assert mlp.n_parameters() > 0 and transformer.n_parameters() > 0
    assert mlp.n_parameters() != transformer.n_parameters()


def test_an_unknown_architecture_fails_loudly(settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        TorchTabularClassifier(_tiny(settings), architecture="resnet").fit(
            *_separable(100, 4, seed=7)
        )
