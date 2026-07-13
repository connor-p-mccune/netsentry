"""Membership-inference attack primitives: the confidence signal, feature construction,
advantage/AUC arithmetic, and the overfit-reference config."""

from __future__ import annotations

import numpy as np

from netsentry.config import Settings
from netsentry.robustness.membership import (
    _overfit_settings,
    attack_scores,
    confidence_features,
    membership_advantage,
    true_class_probability,
)


def test_true_class_probability_selects_the_label_column() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    classes = np.array(["BENIGN", "DDoS", "PortScan"])
    y = np.array(["BENIGN", "DDoS"])
    assert np.allclose(true_class_probability(proba, classes, y), [0.7, 0.8])


def test_true_class_probability_is_zero_for_an_absent_class() -> None:
    # A class the model never saw (missing from a subsample) has no column -> 0.0.
    proba = np.array([[0.6, 0.4]])
    classes = np.array(["BENIGN", "DDoS"])
    assert true_class_probability(proba, classes, np.array(["Heartbleed"]))[0] == 0.0


def test_confidence_features_shape_and_true_prob_column() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6]])
    classes = np.array(["BENIGN", "DDoS", "PortScan"])
    y = np.array(["BENIGN", "PortScan"])
    feats = confidence_features(proba, classes, y, top_k=3)
    # top_k sorted probs (3) + true-prob + correctness + entropy = 6 columns.
    assert feats.shape == (2, 6)
    assert np.allclose(feats[:, 0], [0.7, 0.6])  # highest prob first
    assert np.allclose(feats[:, 3], [0.7, 0.6])  # true-class prob column
    assert np.allclose(feats[:, 4], [1.0, 1.0])  # both predictions correct


def test_confidence_features_pad_narrow_models_to_stable_width() -> None:
    proba = np.array([[0.6, 0.4]])  # only 2 classes, but top_k=3 requested
    feats = confidence_features(proba, np.array(["BENIGN", "DDoS"]), np.array(["BENIGN"]), top_k=3)
    assert feats.shape == (1, 6)
    assert feats[0, 2] == 0.0  # third "top" prob padded with zero


def test_membership_advantage_is_zero_for_indistinguishable_scores() -> None:
    is_member = np.array([1, 1, 0, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert membership_advantage(is_member, scores) == 0.0


def test_membership_advantage_is_one_for_perfect_separation() -> None:
    is_member = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])  # members strictly higher
    assert membership_advantage(is_member, scores) == 1.0


def test_attack_scores_reports_chance_when_one_class_present() -> None:
    auc, adv, tpr, _fpr_arr, _tpr_arr = attack_scores(
        np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]), fpr_budget=0.01
    )
    assert auc == 0.5 and adv == 0.0 and tpr == 0.0


def test_attack_scores_perfect_attack_hits_auc_one() -> None:
    is_member = np.concatenate([np.ones(50), np.zeros(50)])
    scores = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])
    auc, adv, tpr, _, _ = attack_scores(is_member, scores, fpr_budget=0.02)
    assert auc == 1.0 and adv == 1.0 and tpr == 1.0


def test_overfit_settings_removes_regularisation() -> None:
    base = Settings()
    over = _overfit_settings(base)
    assert over.supervised.min_child_samples < base.supervised.min_child_samples
    assert over.supervised.reg_lambda == 0.0
    assert over.supervised.num_leaves >= base.supervised.num_leaves
    assert base.supervised.reg_lambda > 0.0  # the base is left untouched
