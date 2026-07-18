"""Weak supervision: the vote encoding, the EM label model, and the matched-volume stats.

The report's claims rest on the generative label model doing something surprising —
estimating each signature's precision *without labels* — so the tests plant labeling
functions with known accuracies and assert the EM recovers them, alongside the polarity,
determinism, and degenerate-input contracts the study leans on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from netsentry.config.settings import RuleClause, RuleDefinition, WeakSupervisionConfig
from netsentry.training.weak_supervision import (
    VOTE_ABSTAIN,
    VOTE_ATTACK,
    build_label_model,
    cofire_rows,
    fit_label_model,
    noise_aware_labels,
    top_k_stats,
    votes_from_rules,
)


def _cfg(**overrides: float) -> WeakSupervisionConfig:
    return WeakSupervisionConfig(**overrides)  # type: ignore[arg-type]


def _planted_votes(
    y: np.ndarray, fire_on_attack: list[float], fire_on_benign: list[float], seed: int = 0
) -> np.ndarray:
    """Votes for LFs that fire (vote attack) with per-class probabilities, else abstain."""
    rng = np.random.default_rng(seed)
    votes = np.full((len(y), len(fire_on_attack)), VOTE_ABSTAIN, dtype=np.int8)
    for j, (pa, pb) in enumerate(zip(fire_on_attack, fire_on_benign, strict=True)):
        fire_prob = np.where(y == 1, pa, pb)
        votes[rng.random(len(y)) < fire_prob, j] = VOTE_ATTACK
    return votes


def test_votes_from_rules_fire_and_abstain() -> None:
    definitions = [
        RuleDefinition(
            name="high-rate",
            description="",
            clauses=[RuleClause(feature="Flow Packets/s", op="ge", value=100.0)],
        ),
        RuleDefinition(
            name="short-flow",
            description="",
            clauses=[RuleClause(feature="Flow Duration", op="le", value=10.0)],
        ),
    ]
    df = pd.DataFrame({"Flow Packets/s": [500.0, 1.0, 200.0], "Flow Duration": [5.0, 50.0, 50.0]})
    votes = votes_from_rules(df, definitions)
    assert votes.shape == (3, 2)
    assert votes[0].tolist() == [VOTE_ATTACK, VOTE_ATTACK]
    assert votes[1].tolist() == [VOTE_ABSTAIN, VOTE_ABSTAIN]  # silence, never a benign vote
    assert votes[2].tolist() == [VOTE_ATTACK, VOTE_ABSTAIN]


def test_em_recovers_planted_lf_precisions_without_labels() -> None:
    # Four signatures with very different reliabilities; the label model must rank and
    # roughly locate each one's precision from vote agreement alone. The class balance
    # is supplied (it is unidentifiable from attack-or-abstain votes, by design).
    rng = np.random.default_rng(7)
    y = (rng.random(8000) < 0.3).astype(int)
    fire_attack = [0.6, 0.35, 0.25, 0.45]
    fire_benign = [0.01, 0.03, 0.15, 0.02]
    votes = _planted_votes(y, fire_attack, fire_benign, seed=71)  # decoupled from the y draw
    model = fit_label_model(votes, _cfg(class_prior=0.3))

    pi = y.mean()
    for j, (pa, pb) in enumerate(zip(fire_attack, fire_benign, strict=True)):
        true_precision = pi * pa / (pi * pa + (1 - pi) * pb)
        # Cast-votes-only likelihood trades some sharpness for dependence-robustness, so
        # the tolerance is honest about the smoothing shrinkage, not a point estimate.
        assert abs(model.implied_precision(j) - true_precision) < 0.15, f"LF {j}"
    assert model.prior == 0.3  # fixed, never re-estimated
    # The noisy LF (index 2) must be recognised as the least precise of the four.
    precisions = [model.implied_precision(j) for j in range(4)]
    assert int(np.argmin(precisions)) == 2


def test_misspecified_prior_still_ranks_the_lfs() -> None:
    # Halving the assumed class balance shifts the precision estimates but must not
    # scramble which signatures the model trusts — the ranking is what training uses.
    rng = np.random.default_rng(19)
    y = (rng.random(8000) < 0.3).astype(int)
    votes = _planted_votes(y, [0.6, 0.35, 0.25, 0.45], [0.01, 0.03, 0.15, 0.02], seed=191)
    good = fit_label_model(votes, _cfg(class_prior=0.3))
    off = fit_label_model(votes, _cfg(class_prior=0.15))
    p_good = [good.implied_precision(j) for j in range(4)]
    p_off = [off.implied_precision(j) for j in range(4)]
    # The invariant training depends on: the noisy LF (2) stays clearly least trusted
    # under both priors. (Total orders over-assert — the three precise LFs are near-tied.)
    assert int(np.argmin(p_good)) == 2 and int(np.argmin(p_off)) == 2
    assert max(p_good) - p_good[2] > 0.2 and max(p_off) - p_off[2] > 0.2


def test_posterior_separates_classes_better_than_the_best_single_lf() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(11)
    y = (rng.random(6000) < 0.25).astype(int)
    votes = _planted_votes(y, [0.5, 0.3, 0.3], [0.02, 0.05, 0.08], seed=113)
    model = fit_label_model(votes, _cfg())
    posterior = model.posterior(votes)
    post_auc = roc_auc_score(y, posterior)
    best_single = max(
        roc_auc_score(y, (votes[:, j] == VOTE_ATTACK).astype(float)) for j in range(3)
    )
    assert post_auc > best_single  # weighing all votes beats trusting any one signature


def test_uninformative_abstention_returns_the_prior_on_silent_rows() -> None:
    # LFs abstain at the same rate on both classes, so silence carries no evidence and
    # an all-abstain row's posterior must collapse to the assumed prior.
    rng = np.random.default_rng(3)
    y = (rng.random(6000) < 0.4).astype(int)
    votes = np.full((len(y), 3), VOTE_ABSTAIN, dtype=np.int8)
    for j in range(3):
        speaks = rng.random(len(y)) < 0.5  # class-independent propensity
        correct = rng.random(len(y)) < 0.9
        vote_val = np.where(correct, y, 1 - y)  # attack vote = 1, benign vote = 0
        votes[speaks, j] = vote_val[speaks].astype(np.int8)
    model = fit_label_model(votes, _cfg(class_prior=0.4))
    silent = (votes == VOTE_ABSTAIN).all(axis=1)
    assert silent.any()
    posterior = model.posterior(votes)
    assert np.allclose(posterior[silent], model.prior, atol=0.05)


def test_agreement_gate_refuses_to_fit_on_disjoint_votes() -> None:
    # Three disjoint one-sided signatures: accuracies are unidentifiable without
    # agreement, so the gate must state the configured trust instead of fitting.
    votes = np.full((300, 3), VOTE_ABSTAIN, dtype=np.int8)
    votes[0:50, 0] = VOTE_ATTACK
    votes[50:80, 1] = VOTE_ATTACK
    votes[80:100, 2] = VOTE_ATTACK
    votes[0:10, 1] = VOTE_ATTACK  # a whisker of co-fire, well under the gate
    assert cofire_rows(votes) == 10
    model, mode = build_label_model(votes, _cfg(class_prior=0.2, signature_trust=0.8))
    assert mode == "prior belief"
    post = model.posterior(votes)
    assert np.allclose(post[10:50], 0.8, atol=1e-9)  # a single fire is exactly the trust
    assert np.allclose(post[100:], 0.2, atol=1e-9)  # silence is exactly the prior
    assert (post[0:10] > 0.9).all()  # two fires compose by naive-Bayes odds
    for j in range(3):
        assert np.isclose(model.implied_precision(j), 0.8)


def test_agreement_gate_fits_em_when_overlap_exists() -> None:
    rng = np.random.default_rng(29)
    y = (rng.random(4000) < 0.3).astype(int)
    votes = _planted_votes(y, [0.6, 0.5, 0.4], [0.02, 0.03, 0.1], seed=291)
    assert cofire_rows(votes) > 50
    model, mode = build_label_model(votes, _cfg(class_prior=0.3))
    assert mode == "agreement (EM)"
    assert model.n_iter > 0  # a genuine EM fit, not the combiner


def test_label_model_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.3).astype(int)
    votes = _planted_votes(y, [0.5, 0.2], [0.02, 0.1], seed=53)
    a = fit_label_model(votes, _cfg())
    b = fit_label_model(votes, _cfg())
    assert a.prior == b.prior
    assert np.array_equal(a.vote_probs, b.vote_probs)


def test_noise_aware_labels_floor_and_polarity() -> None:
    posterior = np.array([0.95, 0.5, 0.1, 0.52])
    y, weights, keep = noise_aware_labels(posterior, min_weight=0.05)
    assert y.tolist() == [1, 1, 0, 1]
    assert np.allclose(weights, [0.9, 0.0, 0.8, 0.04])
    assert keep.tolist() == [True, False, True, False]  # agnostic rows teach nothing


def test_top_k_stats_hand_checked() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    y = np.array([1, 0, 1, 1, 0])
    silent = np.array([False, False, True, True, False])  # attacks 2 and 3 are rule-silent
    precision, recall, silent_recall = top_k_stats(scores, y, k=3, silent_mask=silent)
    assert precision == 2 / 3  # rows 0, 1, 2 alerted; two are attacks
    assert recall == 2 / 3  # of three attacks, rows 0 and 2 are caught
    assert silent_recall == 1 / 2  # of the two silent attacks, only row 2 is in the top-3


def test_top_k_stats_zero_volume_is_safe() -> None:
    precision, recall, silent_recall = top_k_stats(
        np.array([0.5]), np.array([1]), k=0, silent_mask=np.array([False])
    )
    assert np.isnan(precision) and recall == 0.0 and silent_recall == 0.0
