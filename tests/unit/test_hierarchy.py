"""Taxonomy-aware scoring: partial credit must be earned, and the flat case must be exact.

The whole argument rests on hierarchical F1 being a *generalisation* of the flat metric
rather than a softer one. If it inflated every score by a constant it would be worthless,
so the first tests pin it against hand-computed values and against the flat metric on a
flat tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.config import Settings
from netsentry.evaluation.hierarchy import (
    ATTACK_NODE,
    BENIGN_NODE,
    ERROR_KINDS,
    Taxonomy,
    build_taxonomy,
    error_kind,
    error_profile,
    hierarchical_prf,
    playbook_cost,
)

BENIGN = "BENIGN"
LABELS = [BENIGN, "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "PortScan", "Bot"]


@pytest.fixture
def taxonomy() -> Taxonomy:
    return build_taxonomy(LABELS, BENIGN)


# --------------------------------------------------------------------------------------
# The taxonomy
# --------------------------------------------------------------------------------------
def test_the_tree_is_built_from_the_attack_mapping_not_invented(taxonomy: Taxonomy) -> None:
    assert taxonomy.ancestors("DoS Hulk")[:2] == (ATTACK_NODE, "Impact")
    assert taxonomy.ancestors("PortScan")[:2] == (ATTACK_NODE, "Discovery")
    assert taxonomy.ancestors(BENIGN) == (BENIGN_NODE, BENIGN)


def test_siblings_share_a_technique_and_cousins_only_a_tactic(taxonomy: Taxonomy) -> None:
    """Hulk and GoldenEye are both T1499; slowloris is T1499.002 under the same tactic."""
    assert taxonomy.ancestors("DoS Hulk")[2] == taxonomy.ancestors("DoS GoldenEye")[2]
    assert taxonomy.ancestors("DoS Hulk")[2] != taxonomy.ancestors("DoS slowloris")[2]
    assert taxonomy.ancestors("DoS Hulk")[1] == taxonomy.ancestors("DoS slowloris")[1]


def test_an_unmapped_class_still_gets_a_path(taxonomy: Taxonomy) -> None:
    """Degrading resolution beats silently dropping a class from the evaluation."""
    tax = build_taxonomy([BENIGN, "Some New Attack"], BENIGN)
    assert tax.ancestors("Some New Attack") == (ATTACK_NODE, "Some New Attack")


def test_path_distance_grows_with_taxonomic_separation(taxonomy: Taxonomy) -> None:
    same = taxonomy.path_distance("DoS Hulk", "DoS Hulk")
    sibling = taxonomy.path_distance("DoS Hulk", "DoS GoldenEye")
    cousin = taxonomy.path_distance("DoS Hulk", "DoS slowloris")
    stranger = taxonomy.path_distance("DoS Hulk", "PortScan")
    across = taxonomy.path_distance("DoS Hulk", BENIGN)
    assert same == 0
    assert sibling < cousin < stranger <= across


# --------------------------------------------------------------------------------------
# Hierarchical precision / recall / F1
# --------------------------------------------------------------------------------------
def test_a_perfect_prediction_scores_one(taxonomy: Taxonomy) -> None:
    y = np.array(["DoS Hulk", "PortScan", BENIGN])
    assert hierarchical_prf(y, y, taxonomy) == pytest.approx((1.0, 1.0, 1.0))


def test_a_sibling_confusion_earns_partial_credit_not_zero(taxonomy: Taxonomy) -> None:
    """Hand-computed: paths share attack/Impact/T1499, 3 of 4 ancestors each."""
    p, r, f = hierarchical_prf(np.array(["DoS Hulk"]), np.array(["DoS GoldenEye"]), taxonomy)
    assert p == pytest.approx(0.75) and r == pytest.approx(0.75) and f == pytest.approx(0.75)


def test_partial_credit_decreases_as_the_confusion_widens(taxonomy: Taxonomy) -> None:
    truth = np.array(["DoS Hulk"])
    scores = [
        hierarchical_prf(truth, np.array([other]), taxonomy)[2]
        for other in ("DoS GoldenEye", "DoS slowloris", "PortScan", BENIGN)
    ]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] == pytest.approx(0.0)  # benign shares no ancestor with an attack


def test_on_a_flat_taxonomy_it_collapses_to_the_flat_metric() -> None:
    """The property that makes this a generalisation rather than a different metric."""
    flat = Taxonomy(paths={name: (name,) for name in ("a", "b", "c")})
    y_true = np.array(["a", "b", "c", "a"])
    y_pred = np.array(["a", "b", "a", "a"])
    p, r, f = hierarchical_prf(y_true, y_pred, flat)
    accuracy = float(np.mean(y_true == y_pred))
    assert p == pytest.approx(accuracy) and r == pytest.approx(accuracy)
    assert f == pytest.approx(accuracy)


# --------------------------------------------------------------------------------------
# The error decomposition
# --------------------------------------------------------------------------------------
def test_every_outcome_is_classified_into_exactly_one_kind(taxonomy: Taxonomy) -> None:
    cases = [
        ("DoS Hulk", "DoS Hulk", "exact"),
        ("DoS Hulk", "DoS GoldenEye", "within_technique"),
        ("DoS Hulk", "DoS slowloris", "within_tactic"),
        ("DoS Hulk", "PortScan", "cross_tactic"),
        ("DoS Hulk", BENIGN, "missed_attack"),
        (BENIGN, "DoS Hulk", "false_alarm"),
    ]
    for true, pred, expected in cases:
        assert error_kind(true, pred, taxonomy, BENIGN) == expected, (true, pred)


def test_the_profile_is_a_distribution(taxonomy: Taxonomy) -> None:
    y_true = np.array(["DoS Hulk", "DoS Hulk", BENIGN, "PortScan"])
    y_pred = np.array(["DoS GoldenEye", BENIGN, "Bot", "PortScan"])
    profile = error_profile(y_true, y_pred, taxonomy, BENIGN)
    assert set(profile) == set(ERROR_KINDS)
    assert sum(profile.values()) == pytest.approx(1.0)
    assert profile["within_technique"] == pytest.approx(0.25)
    assert profile["missed_attack"] == pytest.approx(0.25)


def test_a_missed_attack_costs_more_than_a_sibling_name(settings: Settings) -> None:
    costs = settings.hierarchy.error_costs()
    assert costs["exact"] == 0.0
    assert costs["within_technique"] < costs["within_tactic"] < costs["cross_tactic"]
    assert costs["cross_tactic"] <= costs["missed_attack"]


def test_cost_is_zero_for_a_perfect_classifier(settings: Settings) -> None:
    perfect = dict.fromkeys(ERROR_KINDS, 0.0) | {"exact": 1.0}
    assert playbook_cost(perfect, settings.hierarchy.error_costs()) == pytest.approx(0.0)


def test_cost_separates_two_models_with_identical_accuracy(settings: Settings) -> None:
    """The point of the whole report: same accuracy, different consequences."""
    costs = settings.hierarchy.error_costs()
    cheap = {"exact": 0.9, "within_technique": 0.1}
    expensive = {"exact": 0.9, "missed_attack": 0.1}
    assert playbook_cost(cheap, costs) < playbook_cost(expensive, costs)
