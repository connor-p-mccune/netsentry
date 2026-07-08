"""Self-training study: pseudo-label selection and the truth audit."""

from __future__ import annotations

import numpy as np

from netsentry.training.selftrain import audit_pseudo_labels, select_pseudo_labels


def test_select_respects_confidence_band_and_abstains_between() -> None:
    scores = np.array([0.99, 0.985, 0.5, 0.4, 0.01, 0.015, 0.6])
    pseudo = select_pseudo_labels(scores, tau_attack=0.98, tau_benign=0.02, max_per_class=10)
    assert sorted(pseudo.attack_idx.tolist()) == [0, 1]
    assert sorted(pseudo.benign_idx.tolist()) == [4, 5]
    assert pseudo.n == 4  # the three mid-band flows are abstentions


def test_select_caps_each_side_by_confidence() -> None:
    scores = np.array([0.99, 0.999, 0.995, 0.001, 0.005, 0.003])
    pseudo = select_pseudo_labels(scores, tau_attack=0.98, tau_benign=0.01, max_per_class=2)
    # Most confident first: highest scores for attack, lowest for benign.
    assert pseudo.attack_idx.tolist() == [1, 2]
    assert pseudo.benign_idx.tolist() == [3, 5]


def test_audit_counts_absorbed_attacks() -> None:
    # 6 flows: pseudo-attack = {0, 1}, pseudo-benign = {2, 3, 4}, abstain = {5}.
    scores = np.array([0.99, 0.99, 0.01, 0.01, 0.01, 0.5])
    y_true = np.array([1, 0, 1, 1, 0, 1])  # two attacks hide inside pseudo-benign
    pseudo = select_pseudo_labels(scores, tau_attack=0.98, tau_benign=0.02, max_per_class=10)
    audit = audit_pseudo_labels(pseudo, y_true)

    assert audit.n_window == 6
    assert audit.n_attacks_in_window == 4
    assert audit.n_pseudo_attack == 2
    assert audit.n_pseudo_benign == 3
    assert audit.n_abstained == 1
    assert audit.attacks_claimed == 1
    assert audit.attacks_absorbed == 2  # the confirmation-bias cell
    assert audit.attack_precision == 0.5
    assert audit.benign_precision == 1 / 3


def test_audit_handles_empty_sides() -> None:
    scores = np.array([0.5, 0.6])
    pseudo = select_pseudo_labels(scores, tau_attack=0.98, tau_benign=0.02, max_per_class=10)
    audit = audit_pseudo_labels(pseudo, np.array([0, 1]))
    assert pseudo.n == 0
    assert np.isnan(audit.attack_precision) and np.isnan(audit.benign_precision)
    assert audit.n_abstained == 2
