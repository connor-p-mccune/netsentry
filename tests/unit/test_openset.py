"""Open-set recognition: the novelty scores, the Mahalanobis scorer, and the OSCR metric.

Every piece the report leans on has an answer that can be derived by hand — the entropy of a
uniform distribution, the Mahalanobis distance under a known covariance, the OSCR area for a
perfectly separated pair, Scheirer's openness on his own worked numbers — so they are pinned
here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.openset import (
    MahalanobisScorer,
    detection_at_fpr,
    entropy_novelty,
    margin_novelty,
    msp_novelty,
    openness,
    openset_auroc,
    oscr_auc,
    oscr_curve,
    percentile_rank,
    rank_average,
)


def test_msp_novelty_is_zero_for_a_confident_prediction() -> None:
    proba = np.array([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2]])
    assert np.allclose(msp_novelty(proba), [0.0, 0.5])


def test_entropy_novelty_is_maximal_for_a_uniform_distribution() -> None:
    k = 4
    uniform = np.full((1, k), 1.0 / k)
    one_hot = np.eye(k)[:1]
    assert np.isclose(entropy_novelty(uniform)[0], np.log(k))
    assert entropy_novelty(one_hot)[0] < 1e-9


def test_margin_novelty_is_one_minus_the_top_two_gap() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    assert np.allclose(margin_novelty(proba), [1.0 - 0.5, 1.0 - 0.05])


def test_percentile_rank_places_values_in_the_reference_distribution() -> None:
    reference = np.arange(100, dtype=float)  # 0..99
    ranks = percentile_rank(np.array([-5.0, 0.0, 50.0, 200.0]), reference)
    assert np.allclose(ranks, [0.0, 0.0, 0.5, 1.0])


def test_percentile_rank_rejects_an_empty_reference() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        percentile_rank(np.zeros(3), np.array([]))


def test_rank_average_is_invariant_to_a_monotone_rescaling() -> None:
    # Fusion must not let whichever score has the widest numeric range dominate.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    ref_a, ref_b = np.linspace(0, 5, 50), np.linspace(0, 50, 50)
    assert np.allclose(
        rank_average([a, b], [ref_a, ref_b]),
        rank_average([a, np.exp(b)], [ref_a, np.exp(ref_b)]),
    )


def test_rank_average_keeps_block_separation_the_within_block_ranking_destroys() -> None:
    # The bug this guards: ranking each block against itself makes both blocks uniform on
    # [0, 1], so a fused score loses the very separation its members had. Against a shared
    # reference the separation survives.
    reference = np.linspace(0.0, 1.0, 200)
    known, unknown = np.linspace(0.0, 0.2, 40), np.linspace(0.8, 1.0, 40)
    fused_known = rank_average([known, known], [reference, reference])
    fused_unknown = rank_average([unknown, unknown], [reference, reference])
    assert fused_unknown.min() > fused_known.max()


def test_rank_average_rejects_ragged_inputs() -> None:
    ref = [np.arange(5.0), np.arange(5.0)]
    with pytest.raises(ValueError, match="equal-length"):
        rank_average([np.zeros(3), np.zeros(4)], ref)


def test_rank_average_rejects_a_missing_reference() -> None:
    with pytest.raises(ValueError, match="one reference per"):
        rank_average([np.zeros(3), np.zeros(3)], [np.arange(5.0)])


def test_mahalanobis_recovers_the_analytic_distance_under_an_identity_covariance() -> None:
    # Two unit-variance classes at +/-3 in one dimension: with no shrinkage the squared
    # distance from the origin to the nearest mean is 9.
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3.0, 1.0, (4000, 1)), rng.normal(3.0, 1.0, (4000, 1))])
    y = np.array(["a"] * 4000 + ["b"] * 4000)
    scorer = MahalanobisScorer(shrinkage=0.0).fit(x, y)
    assert scorer.score(np.array([[0.0]]))[0] == pytest.approx(9.0, rel=0.1)
    assert scorer.score(np.array([[-3.0]]))[0] == pytest.approx(0.0, abs=0.05)


def test_mahalanobis_scores_an_unseen_region_above_a_training_class() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 3))
    y = np.zeros(500, dtype=int)
    scorer = MahalanobisScorer(shrinkage=0.1).fit(x, y)
    near = scorer.score(rng.normal(size=(50, 3)))
    far = scorer.score(rng.normal(loc=8.0, size=(50, 3)))
    assert far.min() > near.max()


def test_mahalanobis_rejects_an_out_of_range_shrinkage() -> None:
    with pytest.raises(ValueError, match="shrinkage"):
        MahalanobisScorer(shrinkage=1.5)


def test_openness_is_zero_for_a_closed_set_problem() -> None:
    # Same classes at train and test, and nothing else to emit: a closed-set protocol.
    assert openness(10, 10, 10) == pytest.approx(0.0)


def test_openness_grows_as_unknown_classes_are_added() -> None:
    closed = openness(5, 5, 5)
    somewhat = openness(5, 8, 5)
    very = openness(5, 20, 5)
    assert closed < somewhat < very < 1.0


def test_openness_rejects_nonpositive_class_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        openness(0, 4)


def test_detection_at_fpr_sets_the_threshold_on_known_flows_only() -> None:
    # Known scores uniform on [0, 1): the 99th percentile is ~0.99, so only unknowns
    # above it count as detected.
    known = np.linspace(0.0, 1.0, 1000, endpoint=False)
    unknown = np.array([0.5, 0.995, 0.999, 2.0])
    udr, threshold = detection_at_fpr(known, unknown, 0.01)
    assert threshold == pytest.approx(0.99, abs=0.01)
    assert udr == pytest.approx(0.75)


def test_openset_auroc_is_one_for_perfect_separation() -> None:
    assert openset_auroc(np.zeros(50), np.ones(50)) == pytest.approx(1.0)


def test_openset_auroc_is_half_for_identical_distributions() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    assert openset_auroc(a, b) == pytest.approx(0.5, abs=0.03)


def test_oscr_auc_is_one_when_separation_is_perfect_and_the_known_task_is_solved() -> None:
    known_correct = np.ones(20, dtype=bool)
    novelty_known = np.zeros(20)
    novelty_unknown = np.ones(20)
    assert oscr_auc(known_correct, novelty_known, novelty_unknown) == pytest.approx(1.0)


def test_oscr_charges_for_closed_set_errors_that_auroc_ignores() -> None:
    # Identical, perfect novelty separation; the only difference is that half the known
    # flows are misclassified. AUROC cannot see it; OSCR halves.
    novelty_known, novelty_unknown = np.zeros(20), np.ones(20)
    all_right = np.ones(20, dtype=bool)
    half_right = np.array([True] * 10 + [False] * 10)
    assert openset_auroc(novelty_known, novelty_unknown) == pytest.approx(1.0)
    assert oscr_auc(half_right, novelty_known, novelty_unknown) == pytest.approx(
        0.5 * oscr_auc(all_right, novelty_known, novelty_unknown)
    )


def test_oscr_curve_is_monotone_in_fpr_and_bounded() -> None:
    rng = np.random.default_rng(4)
    known = rng.normal(size=200)
    unknown = rng.normal(loc=1.5, size=200)
    correct = rng.random(200) < 0.8
    fpr, ccr = oscr_curve(correct, known, unknown)
    assert np.all(np.diff(fpr) >= -1e-12)  # sorted by construction
    assert np.all((ccr >= 0.0) & (ccr <= 1.0))
    assert ccr[-1] == pytest.approx(float(np.mean(correct)))  # accept everything
