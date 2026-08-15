"""The partial-AUC metric and its differentiable surrogate.

Two things have to be true for the study that uses these to mean anything: the metric has to
agree with what "TPR inside a false-positive budget" means on cases where the answer is obvious,
and the surrogate has to actually be a relaxation of it -- lower when the ranking is better,
sensitive only to the negatives inside the budget, and differentiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.models.pauc import pairwise_pauc_surrogate, partial_auc, top_k_negatives
from netsentry.utils.optional import is_available


def test_top_k_is_clamped_to_a_usable_range() -> None:
    assert top_k_negatives(0, 100) == 1  # never zero: an empty top-k has no gradient
    assert top_k_negatives(5, 100) == 5
    assert top_k_negatives(500, 100) == 100  # never more negatives than exist


def test_perfect_separation_scores_one_and_random_scores_half() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200, dtype=int), np.zeros(2000, dtype=int)]
    perfect = np.r_[rng.uniform(0.6, 1.0, 200), rng.uniform(0.0, 0.4, 2000)]
    assert np.isclose(partial_auc(y, perfect, alpha=0.01), 1.0, atol=0.02)
    noise = rng.uniform(size=2200)
    assert abs(partial_auc(y, noise, alpha=0.05) - 0.5) < 0.2
    # And the normalisation is what makes budgets comparable: a random ranker sits at 0.5 at
    # every budget, where the raw area would shrink with alpha for purely arithmetic reasons.
    assert abs(partial_auc(y, noise, alpha=0.001) - 0.5) < 0.3


def test_partial_auc_only_looks_inside_the_budget() -> None:
    """A model made worse *outside* the budget must not change the partial AUC inside it."""
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(300, dtype=int), np.zeros(3000, dtype=int)]
    scores = np.r_[rng.uniform(0.5, 1.0, 300), rng.uniform(0.0, 0.5, 3000)]
    before = partial_auc(y, scores, alpha=0.005)
    damaged = scores.copy()
    # Push the *lowest*-scoring negatives around: they are nowhere near the 0.5% threshold.
    low = np.argsort(damaged)[:500]
    damaged[low] = rng.uniform(0.0, 0.05, len(low))
    assert np.isclose(partial_auc(y, damaged, alpha=0.005), before, atol=1e-9)


def test_partial_auc_is_undefined_without_both_classes() -> None:
    assert np.isnan(partial_auc(np.ones(10, dtype=int), np.linspace(0, 1, 10), alpha=0.01))


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_the_surrogate_falls_as_the_ranking_improves() -> None:
    import torch

    negatives = torch.linspace(0.0, 1.0, 100)
    bad = torch.zeros(20)  # every positive below every negative
    good = torch.full((20,), 5.0)  # every positive above every negative
    kwargs = {"alpha": 0.1, "temperature": 0.5}
    assert pairwise_pauc_surrogate(good, negatives, **kwargs) < pairwise_pauc_surrogate(
        bad, negatives, **kwargs
    )
    assert 0.0 <= float(pairwise_pauc_surrogate(good, negatives, **kwargs)) <= 1.0


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_the_surrogate_ignores_negatives_outside_the_budget() -> None:
    """The property that makes it an operating-point objective rather than a ranking one."""
    import torch

    negatives = torch.linspace(0.0, 1.0, 200)
    positives = torch.full((10,), 0.99)
    before = float(pairwise_pauc_surrogate(positives, negatives, alpha=0.05, temperature=0.5))
    lowered = negatives.clone()
    lowered[:100] -= 10.0  # the easy half gets much easier; the top decile is untouched
    after = float(pairwise_pauc_surrogate(positives, lowered, alpha=0.05, temperature=0.5))
    assert np.isclose(before, after)


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_the_surrogate_has_gradients_that_push_positives_up() -> None:
    import torch

    positives = torch.tensor([0.1, 0.2], requires_grad=True)
    negatives = torch.tensor([0.5, 0.6, 0.7])
    loss = pairwise_pauc_surrogate(positives, negatives, alpha=0.5, temperature=0.5)
    loss.backward()
    assert positives.grad is not None
    assert (positives.grad < 0).all()  # descending the loss raises the positives' scores


@pytest.mark.skipif(not is_available("torch"), reason="torch (ae extra) not installed")
def test_the_surrogate_is_defined_on_a_batch_with_no_positives() -> None:
    # Under 20% prevalence a minibatch can legitimately contain none, and a loss that raised
    # there would abort training somewhere between epochs one and two.
    import torch

    loss = pairwise_pauc_surrogate(
        torch.empty(0), torch.linspace(0, 1, 50), alpha=0.1, temperature=0.5
    )
    assert float(loss.detach()) == 0.0
