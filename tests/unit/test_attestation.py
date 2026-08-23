"""The commitment, the certificate, and every forgery the report claims it refuses.

A verification routine that has never rejected anything is indistinguishable from one that
cannot. So each forgery in the report gets a test here, and the two that are easiest to get
wrong get their own: dropping a tree (which the arithmetic check *cannot* see, because a
missing summand leaves a consistent sum) and replaying a certificate onto a neighbouring flow
(which the hash chain cannot see, because a proof is about a leaf region).
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.governance.attestation import (
    Commitment,
    CommittedEnsemble,
    commit_tree,
    fold_merkle_path,
    forge_dropped_tree,
    forge_leaf_value,
    forge_score_only,
    forge_sibling,
    forge_spliced_path,
    forge_threshold,
    leaf_hash,
    merkle_path,
    merkle_root,
    node_hash,
    verify_certificate,
)
from netsentry.robustness.verify_trees import Tree, ensemble_margin


def _stump(feature: int, threshold: float, low: float, high: float) -> Tree:
    """A one-split tree: node 0 branches on ``feature``, children are leaves."""
    return Tree(
        feature=np.array([feature, -1, -1], dtype=np.int32),
        threshold=np.array([threshold, 0.0, 0.0], dtype=np.float64),
        left=np.array([1, -1, -1], dtype=np.int32),
        right=np.array([2, -1, -1], dtype=np.int32),
        value=np.array([0.0, low, high], dtype=np.float64),
    )


def _ensemble() -> CommittedEnsemble:
    return CommittedEnsemble.commit(
        [
            _stump(0, 0.5, -1.0, 2.0),
            _stump(1, -0.25, 0.5, -0.75),
            _stump(2, 1.0, 0.25, 0.125),
        ]
    )


# --------------------------------------------------------------------------------------
# The hash construction.
# --------------------------------------------------------------------------------------


def test_a_leaf_hash_can_never_be_read_as_a_node_hash() -> None:
    """Domain separation: without the tags a float could be presented as a subtree digest."""
    assert leaf_hash(1.0) != node_hash(0, 1.0, b"", b"")


def test_the_tree_commitment_moves_when_any_part_of_the_tree_does() -> None:
    base = commit_tree(_stump(0, 0.5, -1.0, 2.0))[0]
    assert commit_tree(_stump(0, 0.5, -1.0, 2.0))[0] == base  # deterministic
    assert commit_tree(_stump(0, 0.5, -1.0, 2.5))[0] != base  # a leaf value
    assert commit_tree(_stump(0, 0.6, -1.0, 2.0))[0] != base  # a threshold
    assert commit_tree(_stump(1, 0.5, -1.0, 2.0))[0] != base  # a split feature


def test_swapping_two_children_changes_the_commitment() -> None:
    """Order matters, so `x <= t -> a` cannot be silently re-read as `x <= t -> b`."""
    left = commit_tree(_stump(0, 0.5, -1.0, 2.0))[0]
    right = commit_tree(_stump(0, 0.5, 2.0, -1.0))[0]
    assert left != right


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 9])
def test_every_merkle_path_folds_back_to_the_root(size: int) -> None:
    """Odd levels duplicate the last node; the path has to survive that on both sides."""
    leaves = [leaf_hash(float(i)) for i in range(size)]
    root = merkle_root(leaves)
    for index in range(size):
        assert fold_merkle_path(leaves[index], merkle_path(leaves, index)) == root


def test_a_merkle_path_does_not_authenticate_the_wrong_leaf() -> None:
    leaves = [leaf_hash(float(i)) for i in range(6)]
    root = merkle_root(leaves)
    assert fold_merkle_path(leaves[3], merkle_path(leaves, 2)) != root


# --------------------------------------------------------------------------------------
# Certificates.
# --------------------------------------------------------------------------------------


def test_a_certificate_verifies_and_its_margin_matches_an_independent_scorer() -> None:
    """Cross-checked against the interval verifier's own traversal, not against itself."""
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    certificate = ensemble.certify(x)
    assert verify_certificate(certificate, x, ensemble.commitment).ok
    assert certificate.margin == pytest.approx(ensemble_margin(ensemble.trees, x))


def test_the_certificate_carries_one_proof_per_tree() -> None:
    ensemble = _ensemble()
    certificate = ensemble.certify(np.array([1.0, 1.0, 1.0]))
    assert len(certificate.proofs) == len(ensemble.trees) == ensemble.commitment.n_trees
    assert certificate.size_bytes > 0


@pytest.mark.parametrize(
    ("name", "forge"),
    [
        ("a rewritten leaf value", lambda c: forge_leaf_value(c, 1.5)),
        ("a moved threshold", lambda c: forge_threshold(c, 0.25)),
        ("a spliced path", forge_spliced_path),
        ("a rewritten sibling", forge_sibling),
        ("a rewritten score", lambda c: forge_score_only(c, -2.0)),
        ("a dropped tree", forge_dropped_tree),
    ],
)
def test_every_forgery_is_refused(name: str, forge) -> None:  # type: ignore[no-untyped-def]
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    result = verify_certificate(forge(ensemble.certify(x)), x, ensemble.commitment)
    assert not result.ok, f"{name} was accepted"
    assert result.reason != "verified"


def test_a_certificate_from_a_different_model_is_refused() -> None:
    """The rollback case: last week's approved bundle, still in memory."""
    ensemble = _ensemble()
    other = CommittedEnsemble.commit(
        [_stump(0, 0.5, -1.0, 2.0), _stump(1, -0.25, 0.5, -0.75), _stump(2, 1.0, 0.25, 9.0)]
    )
    x = np.array([0.0, 1.0, 5.0])
    assert not verify_certificate(ensemble.certify(x), x, other.commitment).ok


def test_dropping_a_tree_is_caught_only_because_the_size_is_committed() -> None:
    """The arithmetic check cannot see a missing summand: the sum stays consistent."""
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    forged = forge_dropped_tree(ensemble.certify(x))
    assert forged.margin == pytest.approx(sum(p.leaf_value for p in forged.proofs))
    lenient = Commitment(root=ensemble.commitment.root, n_trees=len(forged.proofs))
    assert verify_certificate(forged, x, lenient).ok  # size not committed: it slips through
    assert not verify_certificate(forged, x, ensemble.commitment).ok


# --------------------------------------------------------------------------------------
# What a certificate actually proves.
# --------------------------------------------------------------------------------------


def test_an_unbound_certificate_replays_onto_a_flow_in_the_same_leaf_region() -> None:
    """The structural fact the report is built on: a proof covers a region, not a point."""
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    neighbour = x + np.array([1e-6, 1e-6, 1e-6])  # same side of every split
    unbound = ensemble.certify(x, bind_flow=False)
    assert verify_certificate(unbound, neighbour, ensemble.commitment, require_binding=False).ok


def test_binding_the_flow_digest_reduces_the_region_to_the_flow() -> None:
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    bound = ensemble.certify(x)
    assert verify_certificate(bound, x, ensemble.commitment).ok
    assert not verify_certificate(bound, x + 1e-9, ensemble.commitment).ok


def test_a_flow_across_a_split_is_refused_even_unbound() -> None:
    """Move over a threshold and the predicates stop holding, which is the point."""
    ensemble = _ensemble()
    x = np.array([0.0, 1.0, 5.0])
    across = np.array([1.0, 1.0, 5.0])  # crosses the first stump's split at 0.5
    unbound = ensemble.certify(x, bind_flow=False)
    result = verify_certificate(unbound, across, ensemble.commitment, require_binding=False)
    assert not result.ok
    assert "predicate" in result.reason


def test_the_commitment_of_an_empty_ensemble_is_well_defined() -> None:
    assert Commitment.of([]).n_trees == 0
    assert len(Commitment.of([]).root) == 32
    assert Commitment.of([]).hex == Commitment.of([]).root.hex()
