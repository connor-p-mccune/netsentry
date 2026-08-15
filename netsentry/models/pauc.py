"""Train for the operating point, not for the loss.

Every report in this project says the same thing about evaluation: the number that matters is the
detection rate at a fixed false-positive budget, because that is what a SOC deploys. And every
model in this project is trained to minimise cross-entropy over the whole distribution, which is
a different objective in a way that is easy to state: log-loss spends capacity on being right
about the eighty percent of traffic that is obviously benign, while the operating point is
decided entirely by the handful of benign flows that score highest. Those flows are where the
threshold lands. Nothing in the training objective knows they are special.

The **partial AUC** is the metric that does know. Where ROC-AUC integrates the ROC curve over all
false-positive rates, partial AUC integrates it only over ``[0, alpha]`` -- the region the budget
allows -- and maximising it is equivalent to caring only about how positives rank against the
*top-scoring* negatives (Narasimhan & Agarwal 2013). That gives a surrogate with a clean form:

    take the k = ceil(alpha * n_negatives) highest-scoring negatives,
    and penalise every positive that fails to outrank them.

with the step function relaxed to a sigmoid so it has gradients. The relaxation is the standard
pairwise one; the top-k restriction is what makes it an *operating-point* objective rather than a
ranking objective, and it is the part that matters here.

Two honest caveats live in this module rather than in a footnote:

- **A minibatch's top-k is not the population's top-k.** The estimator sees the hardest negatives
  *in the batch*, which at a 0.1% budget and a batch of a thousand is a single flow. That biases
  the objective toward easier negatives than the deployed threshold will actually meet, and the
  fix is a larger batch rather than a cleverer loss -- so the batch size is part of the objective's
  specification here, not a performance knob.
- **Optimising a region trades away the rest of the curve.** A model tuned for 0.1% FPR has no
  reason to be good at 5%, and the study that uses this module measures exactly that by scoring
  every trained model at every budget.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def top_k_negatives(k: int, n_negatives: int) -> int:
    """How many negatives define the threshold at a budget -- at least one, at most all of them."""
    return int(min(max(k, 1), max(n_negatives, 1)))


def partial_auc(
    y_true: np.ndarray, scores: np.ndarray, alpha: float, *, n_grid: int = 512
) -> float:
    """Normalised partial AUC over ``[0, alpha]`` of the ROC curve -- the metric being surrogated.

    Normalised the standard way (McClish 1989): the raw area inside the strip lies between
    ``alpha^2 / 2`` (the diagonal -- a random ranker) and ``alpha`` (a perfect one), and rescaling
    to ``[0.5, 1]`` against those bounds is what makes budgets comparable to each other and to
    ROC-AUC. Without it the raw number shrinks with the budget for arithmetic reasons and a
    reader would compare 0.03 at 5% against 0.0007 at 0.1% as though the model had got worse.

    Computed by interpolating the empirical ROC curve on a grid rather than by integrating its
    steps, so ties in the score do not silently inflate it.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    positives, negatives = s[y == 1], s[y == 0]
    if len(positives) == 0 or len(negatives) == 0 or alpha <= 0:
        return float("nan")
    grid = np.linspace(0.0, alpha, n_grid)
    thresholds = np.quantile(negatives, 1.0 - grid)  # FPR = x  =>  threshold at that quantile
    tpr = np.array([float(np.mean(positives >= t)) for t in thresholds], dtype=float)
    area = float(np.trapezoid(tpr, grid))
    chance = alpha * alpha / 2.0
    return 0.5 * (1.0 + (area - chance) / max(alpha - chance, 1e-12))


def pairwise_pauc_surrogate(
    positive_scores: Any, negative_scores: Any, *, alpha: float, temperature: float
) -> Any:
    """Differentiable partial-AUC loss: penalise positives that fail to outrank the top negatives.

    Returns ``1 - mean sigmoid((s_pos - s_neg_topk) / temperature)``, which is a smooth upper
    bound on the fraction of (positive, hard-negative) pairs the model gets the wrong way round.
    The temperature is the only tuning knob and it means something concrete: the score margin at
    which a pair stops contributing gradient.

    Takes and returns torch tensors; imported lazily so the module is importable without torch.
    """
    import torch

    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return torch.zeros((), dtype=positive_scores.dtype, requires_grad=True)
    k = top_k_negatives(math.ceil(alpha * negative_scores.numel()), negative_scores.numel())
    hardest = torch.topk(negative_scores, k).values
    margins = positive_scores[:, None] - hardest[None, :]
    return 1.0 - torch.sigmoid(margins / temperature).mean()
