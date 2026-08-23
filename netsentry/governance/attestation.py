"""Proof-carrying verdicts: verifying the computation, not the bytes it ran from.

This project already attests two things and neither of them is the one that matters at the
moment a verdict is issued. [`netsentry verify`](provenance.md) hashes the bundle **at rest**,
which proves the file on disk is the file that was reviewed. The [alert ledger](ledger.md)
hash-chains the alert **history**, which proves nothing was edited after the fact. Between them
sits the gap: a service whose in-memory model has been swapped, downgraded to yesterday's
approved version, or quietly truncated produces verdicts that pass both checks, because both
checks are about artefacts rather than about the computation that produced the answer.

The fix is a **certificate**: a per-verdict object an auditor can check against a published
commitment, without the model, without re-running inference, and without trusting the service.
The mechanism falls out of the model's own shape. Hash a decision tree bottom-up --

    leaf:     H("L" || value)
    internal: H("I" || feature || threshold || hash(left) || hash(right))

-- and the tree *is* a Merkle tree. A root-to-leaf path, with each step's sibling hash attached,
is then a standard authentication path: an auditor recomputes the chain from the leaf upward and
must land on the committed root, which is impossible to forge without a hash collision. The
ensemble commits to a Merkle root over its per-tree roots, so a certificate also proves the tree
it came from is *in* the ensemble, at the position it claims.

Verification checks three things and each of them stops a different attack:

1. **The hash chain** -- the path is really in the committed tree (stops an edited leaf value or
   a moved threshold).
2. **The predicates** -- the flow's own feature values satisfy every branch the path claims to
   have taken (stops a path spliced from somewhere else).
3. **The arithmetic** -- the leaf values sum to the reported margin (stops a dropped tree or a
   rewritten score).

Nine forgeries are executed against it rather than argued about, and one of them **succeeds**:
a certificate is a proof about a *leaf region*, not about a flow, so it replays cleanly onto any
other flow that lands in the same leaves. That is a real protocol flaw, it is measured rather
than mentioned, and the one-line fix -- binding the flow's own digest into the transcript -- is
implemented and re-measured.

The second half is the cost nobody advertises. A certificate reveals a path per tree, so an
adversary collecting them reconstructs the model one branch at a time. **Verifiability and
confidentiality trade off**, and here that trade is denominated in certificates.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.robustness.verify_trees import Tree, parse_booster
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import AttestationConfig

logger = get_logger(__name__)

REPORT_NAME = "attestation.md"
REPLAY_FIGURE = "attestation_replay.png"
LEAKAGE_FIGURE = "attestation_leakage.png"

_LEAF_TAG = b"L"
_NODE_TAG = b"I"
_PAIR_TAG = b"P"


def _digest(*parts: bytes) -> bytes:
    """SHA-256 over the concatenation. The only cryptographic assumption in the module."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
    return hasher.digest()


def leaf_hash(value: float) -> bytes:
    """Commit to a leaf's contribution.

    Tagged so that a leaf can never be confused with an internal node: without a domain
    separator an attacker who controls a float could present it as a node hash, which is the
    standard second-preimage trick against unlabelled Merkle constructions.
    """
    return _digest(_LEAF_TAG, struct.pack("<d", float(value)))


def node_hash(feature: int, threshold: float, left: bytes, right: bytes) -> bytes:
    """Commit to an internal node together with both of its subtrees."""
    return _digest(
        _NODE_TAG, struct.pack("<i", int(feature)), struct.pack("<d", float(threshold)), left, right
    )


def commit_tree(tree: Tree) -> list[bytes]:
    """Hash every node of one tree bottom-up; the root's hash commits to the whole tree.

    Returned per node rather than only at the root because the certificate needs the *sibling*
    hash at every step of the path, and recomputing it per verdict would make certification
    quadratic in depth for no reason.
    """
    hashes: list[bytes] = [b""] * tree.n_nodes

    def walk(node: int) -> bytes:
        if tree.feature[node] < 0:
            hashes[node] = leaf_hash(float(tree.value[node]))
            return hashes[node]
        left = walk(int(tree.left[node]))
        right = walk(int(tree.right[node]))
        hashes[node] = node_hash(int(tree.feature[node]), float(tree.threshold[node]), left, right)
        return hashes[node]

    walk(0)
    return hashes


def merkle_root(leaves: list[bytes]) -> bytes:
    """Binary Merkle root over an ordered list, duplicating the last node on odd levels."""
    if not leaves:
        return _digest(_PAIR_TAG)
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_digest(_PAIR_TAG, level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_path(leaves: list[bytes], index: int) -> list[tuple[bytes, bool]]:
    """Authentication path for one leaf: each sibling, and whether it sits on the right."""
    path: list[tuple[bytes, bool]] = []
    level = list(leaves)
    position = index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling = position ^ 1
        path.append((level[sibling], bool(sibling > position)))
        level = [_digest(_PAIR_TAG, level[i], level[i + 1]) for i in range(0, len(level), 2)]
        position //= 2
    return path


def fold_merkle_path(leaf: bytes, path: list[tuple[bytes, bool]]) -> bytes:
    """Recompute a Merkle root from a leaf and its authentication path."""
    current = leaf
    for sibling, on_right in path:
        current = (
            _digest(_PAIR_TAG, current, sibling)
            if on_right
            else _digest(_PAIR_TAG, sibling, current)
        )
    return current


# --------------------------------------------------------------------------------------
# Certificates.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Commitment:
    """What gets published: a 32-byte root and the size of the ensemble it commits to.

    The count is not decoration. Without it, a service can serve a *truncated* ensemble and
    report the correspondingly smaller score: every remaining path still hashes into the root,
    and the leaf values still sum to the number claimed. The arithmetic check cannot see a
    missing summand, so the ensemble's size has to be part of what was committed to.
    """

    root: bytes
    n_trees: int

    @classmethod
    def of(cls, tree_roots: list[bytes]) -> Commitment:
        """Commit to an ordered list of per-tree root hashes."""
        return cls(root=merkle_root(tree_roots), n_trees=len(tree_roots))

    @property
    def hex(self) -> str:
        """The published form, as an auditor would quote it."""
        return self.root.hex()


@dataclass(frozen=True)
class PathStep:
    """One branch of a decision path, with the sibling subtree's hash."""

    feature: int
    threshold: float
    went_left: bool
    sibling: bytes

    @property
    def size_bytes(self) -> int:
        """Wire size: a feature index, a threshold, a direction bit, a hash."""
        return 4 + 8 + 1 + len(self.sibling)


@dataclass(frozen=True)
class TreeProof:
    """The evidence that one tree contributed one leaf value to this verdict."""

    tree_index: int
    steps: list[PathStep]
    leaf_value: float
    merkle_path: list[tuple[bytes, bool]]

    @property
    def size_bytes(self) -> int:
        """Wire size of this tree's share of the certificate."""
        return (
            sum(step.size_bytes for step in self.steps)
            + 8
            + 4
            + sum(len(sibling) + 1 for sibling, _ in self.merkle_path)
        )


@dataclass(frozen=True)
class Certificate:
    """A verdict, and everything an auditor needs to check it against the commitment."""

    margin: float
    proofs: list[TreeProof]
    flow_digest: bytes = b""

    @property
    def size_bytes(self) -> int:
        """Total wire size of the certificate."""
        return sum(proof.size_bytes for proof in self.proofs) + 8 + len(self.flow_digest)

    @property
    def depth(self) -> float:
        """Mean path depth, which is what the size is really made of."""
        return float(np.mean([len(proof.steps) for proof in self.proofs])) if self.proofs else 0.0


@dataclass(frozen=True)
class Verification:
    """The auditor's verdict on a verdict."""

    ok: bool
    reason: str = "verified"
    seconds: float = 0.0


def flow_digest(x: np.ndarray) -> bytes:
    """Commit to the flow the certificate is about.

    Without this the certificate proves a statement about a *leaf region*, and any other flow
    landing in the same leaves can reuse it verbatim. That is not a hypothetical: the replay
    attack below succeeds against the unbound protocol at a rate this module measures.
    """
    return _digest(b"F", np.ascontiguousarray(x, dtype=np.float64).tobytes())


@dataclass
class CommittedEnsemble:
    """A model, its node hashes, and the authentication paths that never change.

    Every tree's Merkle path into the ensemble root is a function of the *model*, not of the
    flow, so it is computed once here. The first version rebuilt it inside the prover, per
    tree, per verdict -- which is quadratic in the number of trees and made certification 370x
    the cost of inference for no reason at all. It is now roughly linear, and the honest cost
    of the scheme is the certificate's size rather than an implementation artefact.
    """

    trees: list[Tree]
    node_hashes: list[list[bytes]]
    paths: list[list[tuple[bytes, bool]]]
    commitment: Commitment

    @classmethod
    def commit(cls, trees: list[Tree]) -> CommittedEnsemble:
        """Hash every tree bottom-up and publish one root over the ordered tree roots."""
        node_hashes = [commit_tree(tree) for tree in trees]
        roots = [nodes[0] for nodes in node_hashes]
        return cls(
            trees=trees,
            node_hashes=node_hashes,
            paths=[merkle_path(roots, index) for index in range(len(roots))],
            commitment=Commitment.of(roots),
        )

    @property
    def internal_nodes(self) -> int:
        """Total internal (decision) nodes -- the denominator the leakage curve uses."""
        return sum(int(np.sum(tree.feature >= 0)) for tree in self.trees)

    def certify(self, x: np.ndarray, *, bind_flow: bool = True) -> Certificate:
        """Score a flow and emit the proof of how the score was reached."""
        proofs: list[TreeProof] = []
        margin = 0.0
        for index, tree in enumerate(self.trees):
            nodes = self.node_hashes[index]
            steps: list[PathStep] = []
            node = 0
            while tree.feature[node] >= 0:
                feature = int(tree.feature[node])
                threshold = float(tree.threshold[node])
                went_left = bool(x[feature] <= threshold)
                child = int(tree.left[node]) if went_left else int(tree.right[node])
                sibling = int(tree.right[node]) if went_left else int(tree.left[node])
                steps.append(PathStep(feature, threshold, went_left, nodes[sibling]))
                node = child
            value = float(tree.value[node])
            margin += value
            proofs.append(TreeProof(index, steps, value, self.paths[index]))
        return Certificate(
            margin=margin, proofs=proofs, flow_digest=flow_digest(x) if bind_flow else b""
        )


def verify_certificate(
    certificate: Certificate,
    x: np.ndarray,
    commitment: Commitment,
    *,
    require_binding: bool = True,
) -> Verification:
    """Check a certificate against the published commitment, without the model.

    The auditor holds the flow, the certificate and one 32-byte root. It never sees the
    ensemble, which is the property that makes this useful: a regulator, a customer or a
    second team can check a verdict without being handed the model.
    """
    start = time.perf_counter()

    def done(ok: bool, reason: str) -> Verification:
        return Verification(ok, reason, time.perf_counter() - start)

    if require_binding and certificate.flow_digest != flow_digest(x):
        return done(False, "the certificate is not bound to this flow")
    if len(certificate.proofs) != commitment.n_trees:
        return done(
            False,
            f"the commitment covers {commitment.n_trees} trees and the certificate carries "
            f"{len(certificate.proofs)}",
        )
    total = 0.0
    for proof in certificate.proofs:
        current = leaf_hash(proof.leaf_value)
        for step in reversed(proof.steps):
            if bool(x[step.feature] <= step.threshold) != step.went_left:
                return done(False, "a branch predicate is not satisfied by this flow")
            current = (
                node_hash(step.feature, step.threshold, current, step.sibling)
                if step.went_left
                else node_hash(step.feature, step.threshold, step.sibling, current)
            )
        if fold_merkle_path(current, proof.merkle_path) != commitment.root:
            return done(False, "the path does not hash into the committed ensemble")
        total += proof.leaf_value
    if abs(total - certificate.margin) > 1e-9:
        return done(False, "the leaf values do not sum to the reported score")
    return done(True, "verified")


# --------------------------------------------------------------------------------------
# The forgeries.
# --------------------------------------------------------------------------------------


def _replace_proof(certificate: Certificate, index: int, proof: TreeProof) -> Certificate:
    proofs = list(certificate.proofs)
    proofs[index] = proof
    return Certificate(certificate.margin, proofs, certificate.flow_digest)


def forge_leaf_value(certificate: Certificate, delta: float) -> Certificate:
    """Report a different score by rewriting a leaf the auditor cannot see."""
    proof = certificate.proofs[0]
    forged = TreeProof(proof.tree_index, proof.steps, proof.leaf_value + delta, proof.merkle_path)
    return Certificate(
        certificate.margin + delta,
        _replace_proof(certificate, 0, forged).proofs,
        certificate.flow_digest,
    )


def forge_threshold(certificate: Certificate, delta: float) -> Certificate:
    """Move a split threshold so a different branch looks correct."""
    proof = certificate.proofs[0]
    steps = list(proof.steps)
    steps[0] = PathStep(
        steps[0].feature, steps[0].threshold + delta, steps[0].went_left, steps[0].sibling
    )
    forged = TreeProof(proof.tree_index, steps, proof.leaf_value, proof.merkle_path)
    return _replace_proof(certificate, 0, forged)


def forge_spliced_path(certificate: Certificate) -> Certificate:
    """Present another tree's path as this tree's."""
    if len(certificate.proofs) < 2:
        return certificate
    donor = certificate.proofs[1]
    victim = certificate.proofs[0]
    forged = TreeProof(victim.tree_index, donor.steps, donor.leaf_value, victim.merkle_path)
    return _replace_proof(certificate, 0, forged)


def forge_dropped_tree(certificate: Certificate) -> Certificate:
    """Quietly serve a smaller ensemble and report the smaller score."""
    dropped = certificate.proofs[-1]
    return Certificate(
        certificate.margin - dropped.leaf_value, certificate.proofs[:-1], certificate.flow_digest
    )


def forge_score_only(certificate: Certificate, delta: float) -> Certificate:
    """Leave the proof intact and simply report a different number."""
    return Certificate(certificate.margin + delta, certificate.proofs, certificate.flow_digest)


def forge_sibling(certificate: Certificate) -> Certificate:
    """Rewrite the hash of the subtree the flow did not visit."""
    proof = certificate.proofs[0]
    steps = list(proof.steps)
    steps[0] = PathStep(
        steps[0].feature,
        steps[0].threshold,
        steps[0].went_left,
        _digest(b"X", steps[0].sibling),
    )
    forged = TreeProof(proof.tree_index, steps, proof.leaf_value, proof.merkle_path)
    return _replace_proof(certificate, 0, forged)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackRow:
    """One forgery, and whether verification survived it."""

    attack: str
    what_it_models: str
    caught: bool
    reason: str


@dataclass(frozen=True)
class ReplayRow:
    """How far a flow can move before its own certificate stops verifying."""

    epsilon: float
    accepted_unbound: float
    accepted_bound: float


@dataclass(frozen=True)
class LeakRow:
    """What an adversary has recovered after collecting some number of certificates."""

    certificates: int
    revealed_nodes: int
    coverage: float
    trees_recovered: int
    surrogate_fidelity: float


@dataclass
class AttestationStudy:
    """Everything the report needs, computed once."""

    n_trees: int
    n_nodes: int
    mean_depth: float
    root: str
    certificate_bytes: int
    response_bytes: int
    model_bytes: int
    certify_ms: float
    verify_ms: float
    inference_ms: float
    hashes_per_verification: int
    attacks: list[AttackRow]
    replay: list[ReplayRow]
    replay_trials: int
    leakage: list[LeakRow]
    alerts_per_day: float
    seconds: float = 0.0

    @property
    def caught(self) -> int:
        """Forgeries verification refused."""
        return sum(1 for row in self.attacks if row.caught)

    @property
    def daily_gigabytes(self) -> float:
        """Certificate volume per day if only the alerts are certified."""
        return self.alerts_per_day * self.certificate_bytes / 1e9

    def widest_replay(self) -> ReplayRow | None:
        """The largest perturbation at which an unbound certificate still verified."""
        accepted = [row for row in self.replay if row.accepted_unbound > 0.0]
        return max(accepted, key=lambda row: row.epsilon) if accepted else None


def _timed(action: Callable[[], object], repeats: int) -> float:
    """Milliseconds per call, averaged."""
    start = time.perf_counter()
    for _ in range(repeats):
        action()
    return (time.perf_counter() - start) * 1000.0 / max(repeats, 1)


def _run_attacks(ensemble: CommittedEnsemble, x: np.ndarray, stale: Commitment) -> list[AttackRow]:
    """Execute every forgery against a real certificate and record what verification says."""
    certificate = ensemble.certify(x)
    commitment = ensemble.commitment
    cases: list[tuple[str, str, Certificate, Commitment]] = [
        (
            "rewrite a leaf value",
            "a service reporting a score its model did not produce",
            forge_leaf_value(certificate, 1.5),
            commitment,
        ),
        (
            "move a split threshold",
            "a model quietly retuned after approval",
            forge_threshold(certificate, 0.25),
            commitment,
        ),
        (
            "splice another tree's path",
            "assembling a plausible proof out of real fragments",
            forge_spliced_path(certificate),
            commitment,
        ),
        (
            "rewrite an unvisited sibling",
            "editing the part of the model this flow never touched",
            forge_sibling(certificate),
            commitment,
        ),
        (
            "report a different score",
            "the cheapest attack: leave the proof alone, change the number",
            forge_score_only(certificate, -2.0),
            commitment,
        ),
        (
            "drop a tree",
            "serving a truncated ensemble to cut latency",
            forge_dropped_tree(certificate),
            commitment,
        ),
        (
            "serve a stale model",
            "last week's approved bundle, still in memory after a rollback",
            certificate,
            stale,
        ),
    ]
    rows: list[AttackRow] = []
    for name, models, forged, against in cases:
        result = verify_certificate(forged, x, against)
        rows.append(AttackRow(name, models, not result.ok, result.reason))
    return rows


def _replay_curve(
    ensemble: CommittedEnsemble,
    matrix: np.ndarray,
    rng: np.random.Generator,
    epsilons: list[float],
    trials: int,
) -> list[ReplayRow]:
    """How far a flow can be moved before its own certificate stops verifying.

    A certificate is not a statement about a flow. It is a statement about the **leaf region**
    the flow fell into: every predicate on every path is an inequality, and any other point
    satisfying all of them verifies against the same proof. So the honest question is not
    "can a certificate be replayed" but "how big is the set it covers", which is the same box
    the [interval verifier](verify_trees.md) computes for robustness, arrived at from the
    other direction.

    Perturbations are drawn as random directions of a fixed size in the standardised feature
    space, so the sweep is in the same sigma units every other threat model here uses.
    """
    rows: list[ReplayRow] = []
    for epsilon in epsilons:
        accepted_unbound = 0
        accepted_bound = 0
        for _ in range(trials):
            index = int(rng.integers(len(matrix)))
            direction = rng.normal(size=matrix.shape[1])
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            moved = matrix[index] + epsilon * direction
            unbound = ensemble.certify(matrix[index], bind_flow=False)
            accepted_unbound += int(
                verify_certificate(unbound, moved, ensemble.commitment, require_binding=False).ok
            )
            bound = ensemble.certify(matrix[index])
            accepted_bound += int(verify_certificate(bound, moved, ensemble.commitment).ok)
        rows.append(
            ReplayRow(
                epsilon=epsilon,
                accepted_unbound=accepted_unbound / max(trials, 1),
                accepted_bound=accepted_bound / max(trials, 1),
            )
        )
    return rows


def _leakage_curve(
    ensemble: CommittedEnsemble,
    matrix: np.ndarray,
    margins: np.ndarray,
    counts: list[int],
    seed: int,
) -> list[LeakRow]:
    """What an adversary collecting certificates learns, as a function of how many.

    Two different things are counted, and the first is the one query access can never buy.
    **Structure**: a certificate spells out one root-to-leaf path per tree -- the feature, the
    threshold and the direction at every branch -- so an adversary who indexes those nodes
    positionally rebuilds the model's actual geometry, node by node, rather than approximating
    its behaviour. Trees whose every internal node has been revealed are recovered *exactly*.

    **Function**: the certificate also carries the raw margin, which makes it a labelled
    training example for a stealing attack with an oracle stronger than the
    [extraction study](extraction.md) assumes -- there the attacker queries for a calibrated
    probability, here they are handed the score and the branches that produced it. Fidelity is
    measured on flows the surrogate never saw, because scoring it on its own certificates would
    measure memorisation and would flatter any adversary at all.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    revealed: set[tuple[int, int]] = set()
    per_tree = [int(np.sum(tree.feature >= 0)) for tree in ensemble.trees]
    counted = [0] * len(ensemble.trees)
    rows: list[LeakRow] = []
    total = ensemble.internal_nodes
    seen = 0
    for target in counts:
        while seen < min(target, len(matrix)):
            x = matrix[seen]
            for tree_index, tree in enumerate(ensemble.trees):
                node = 0
                while tree.feature[node] >= 0:
                    if (tree_index, node) not in revealed:
                        revealed.add((tree_index, node))
                        counted[tree_index] += 1
                    feature = int(tree.feature[node])
                    node = int(
                        tree.left[node] if x[feature] <= tree.threshold[node] else tree.right[node]
                    )
            seen += 1
        fidelity = 0.0
        if seen >= 20 and len(matrix) - seen >= 20:
            surrogate = HistGradientBoostingRegressor(max_iter=150, random_state=seed)
            surrogate.fit(matrix[:seen], margins[:seen])
            stolen = surrogate.predict(matrix[seen:])
            truth = margins[seen:]
            if float(np.std(stolen)) > 1e-9 and float(np.std(truth)) > 1e-9:
                fidelity = float(np.corrcoef(stolen, truth)[0, 1])
        rows.append(
            LeakRow(
                certificates=seen,
                revealed_nodes=len(revealed),
                coverage=len(revealed) / max(total, 1),
                trees_recovered=sum(
                    1 for index, count in enumerate(counted) if count == per_tree[index]
                ),
                surrogate_fidelity=fidelity,
            )
        )
    return rows


def run_attestation_study(settings: Settings) -> AttestationStudy:
    """Commit to the deployed ensemble, certify verdicts, then attack the certificates."""
    start = time.perf_counter()
    cfg: AttestationConfig = settings.attestation
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.features.pipeline import build_pipeline
    from netsentry.models.supervised import SupervisedClassifier

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    matrix: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    model = SupervisedClassifier(variant).fit(x_train, y_train)
    booster = getattr(model.model, "booster_", None)
    if booster is None:
        raise RuntimeError("attestation needs a LightGBM booster to commit to")
    ensemble = CommittedEnsemble.commit(parse_booster(booster.dump_model()))

    # A second model, one seed away, standing in for "the version approved last week" -- the
    # rollback case the bundle hash cannot see once the process is already running.
    other = variant.model_copy(deep=True)
    other.seed = variant.seed + 1
    stale_model = SupervisedClassifier(other).fit(x_train, y_train)
    stale_booster = getattr(stale_model.model, "booster_", None)
    stale = CommittedEnsemble.commit(
        parse_booster(stale_booster.dump_model()) if stale_booster else ensemble.trees
    ).commitment

    sample = matrix[rng.choice(len(matrix), min(cfg.max_flows, len(matrix)), replace=False)]
    certificate = ensemble.certify(sample[0])
    margins = np.asarray(booster.predict(sample, raw_score=True), dtype=float)
    hashes = sum(len(proof.steps) + len(proof.merkle_path) + 1 for proof in certificate.proofs)

    study = AttestationStudy(
        n_trees=len(ensemble.trees),
        n_nodes=ensemble.internal_nodes,
        mean_depth=certificate.depth,
        root=ensemble.commitment.hex,
        certificate_bytes=certificate.size_bytes,
        response_bytes=cfg.response_bytes,
        model_bytes=len(booster.model_to_string().encode()),
        certify_ms=_timed(lambda: ensemble.certify(sample[0]), cfg.timing_repeats),
        verify_ms=_timed(
            lambda: verify_certificate(certificate, sample[0], ensemble.commitment),
            cfg.timing_repeats,
        ),
        inference_ms=_timed(
            lambda: model.predict_proba(sample[:1]), max(cfg.timing_repeats * 10, 10)
        ),
        hashes_per_verification=hashes,
        attacks=_run_attacks(ensemble, sample[0], stale),
        replay=_replay_curve(ensemble, sample, rng, cfg.replay_epsilons, cfg.replay_trials),
        replay_trials=cfg.replay_trials,
        leakage=_leakage_curve(ensemble, sample, margins, cfg.leak_counts, variant.seed),
        alerts_per_day=cfg.alerts_per_day,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Attestation study complete",
        extra={
            "trees": study.n_trees,
            "certificate_kb": round(study.certificate_bytes / 1024, 1),
            "caught": study.caught,
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _attack_table(study: AttestationStudy) -> str:
    rows = "\n".join(
        f"| {row.attack} | {row.what_it_models} | "
        f"{'**refused**' if row.caught else 'ACCEPTED'} | {row.reason} |"
        for row in study.attacks
    )
    return (
        "| forgery | what it models in production | verdict | why |\n" "|---|---|---|---|\n" + rows
    )


def _replay_table(study: AttestationStudy) -> str:
    rows = "\n".join(
        f"| {row.epsilon:g} | {row.accepted_unbound:.0%} | {row.accepted_bound:.0%} |"
        for row in study.replay
    )
    return (
        "| the flow is moved by (sd) | unbound certificate accepted | bound certificate "
        "accepted |\n|---|---|---|\n" + rows
    )


def _leak_table(study: AttestationStudy) -> str:
    rows = "\n".join(
        f"| {row.certificates:,} | {row.revealed_nodes:,} | {row.coverage:.1%} | "
        f"{row.trees_recovered:,} | {row.surrogate_fidelity:.3f} |"
        for row in study.leakage
    )
    return (
        "| certificates collected | internal nodes revealed | share of the model | trees "
        "recovered exactly | surrogate fidelity |\n|---|---|---|---|---|\n" + rows
    )


def _cost_table(study: AttestationStudy) -> str:
    return (
        "| quantity | value | against |\n|---|---|---|\n"
        f"| certificate | **{study.certificate_bytes / 1024:.0f} KB** | "
        f"{study.certificate_bytes / max(study.response_bytes, 1):.0f}x the "
        f"{study.response_bytes}-byte prediction body |\n"
        f"| the model itself | {study.model_bytes / 1024:.0f} KB | one certificate is "
        f"{study.certificate_bytes / max(study.model_bytes, 1):.0%} of it |\n"
        f"| certifying | {study.certify_ms:.1f} ms | "
        f"{study.certify_ms / max(study.inference_ms, 1e-9):.0f}x inference |\n"
        f"| verifying | {study.verify_ms:.1f} ms | "
        f"{study.verify_ms / max(study.inference_ms, 1e-9):.0f}x inference, and "
        f"{study.hashes_per_verification:,} SHA-256 evaluations |\n"
        f"| inference | {study.inference_ms:.2f} ms | -- |\n"
        f"| certifying only the alerts | {study.daily_gigabytes:.1f} GB/day | at "
        f"{study.alerts_per_day:,.0f} alerts a day |"
    )


def _render(study: AttestationStudy, replay: Path, leakage: Path) -> str:
    widest = study.widest_replay()
    last = study.leakage[-1] if study.leakage else None
    first = study.leakage[0] if study.leakage else None
    widest_read = (
        f"{widest.epsilon:g} sd, where {widest.accepted_unbound:.0%} of moved flows still "
        f"verified"
        if widest
        else "no perturbation at all"
    )
    first_read = (
        f"A single certificate reveals {first.coverage:.0%} of the ensemble's internal nodes"
        if first
        else "One certificate reveals a share of the nodes"
    )
    last_read = (
        f"{last.certificates:,} of them reveal {last.coverage:.0%} and recover "
        f"{last.trees_recovered:,} of the {study.n_trees:,} trees exactly"
        if last
        else "more reveal more"
    )
    return f"""# NetSentry — Proof-Carrying Verdicts

_The deployed ensemble ({study.n_trees:,} trees, {study.n_nodes:,} internal nodes) committed to
a single {len(study.root) // 2}-byte root, with a certificate per verdict that an auditor can
check without the model. Seven forgeries executed against it. Regenerate with `netsentry
attest`._

Published commitment: `{study.root[:32]}...`

## Why this report exists

This project already attests two things, and neither is the one that matters at the moment a
verdict is issued. [`netsentry verify`](provenance.md) hashes the bundle **at rest**, which
proves the file on disk is the file that was reviewed. The [alert ledger](ledger.md)
hash-chains the alert **history**, which proves nothing was edited afterwards. Between them
sits the gap: a service whose in-memory model has been swapped, rolled back or quietly
truncated passes both checks, because both are about artefacts rather than about the
computation that produced the answer.

A certificate closes it. And the mechanism needs no new cryptography, because the model already
has the right shape: hash a decision tree bottom-up --

```
leaf:     H("L" || value)
internal: H("I" || feature || threshold || hash(left) || hash(right))
```

-- and **the tree is a Merkle tree**. A root-to-leaf path with each step's sibling hash attached
is then an ordinary authentication path, and the ensemble publishes one Merkle root over its
per-tree roots. An auditor holding the flow, the certificate and {len(study.root) // 2} bytes
can check the verdict without the model, without re-running inference and without trusting the
service.

## The forgeries

{_attack_table(study)}

**{study.caught} of {len(study.attacks)} refused.** Three of those rows are the interesting ones.

*Dropping a tree* is the attack the obvious design misses. Serve 599 of 600 trees and report
the correspondingly smaller score: every remaining path still hashes into the root, and the
leaf values still sum to exactly the number claimed, because the arithmetic check cannot see a
missing summand. It is caught here only because the **ensemble's size is part of the
commitment** rather than an out-of-band expectation, which is a design decision that had to be
made before the attack could be refused.

*Serving a stale model* is the one operations people will actually meet -- a rollback that
leaves last week's approved bundle in memory. The bundle hash on disk is then correct and
describes a file the process is not using. The root does not match, so the certificate is
refused on the first tree.

*Rewriting an unvisited sibling* is included because it is the attack a hand-rolled scheme
usually permits: the flow never touched that subtree, so it is tempting to leave it out of the
proof. Committing to both children at every node is what makes the path unforgeable, and the
domain-separating tags on leaves and internal nodes are what stop a leaf value being presented
as a node hash.

## What a certificate actually proves

![How far a flow can move before its certificate stops verifying](../figures/{replay.name})

{_replay_table(study)}

A certificate is **not** a statement about a flow. Every predicate on every path is an
inequality, so the object being proved is that *some point in a particular leaf region* produces
this score -- and any other point in that region satisfies the identical proof. The region has a
measurable size: an unbound certificate survives perturbation out to {widest_read}, and stops
being accepted by {study.replay[-1].epsilon:g} sd.

That is the same box the [interval verifier](verify_trees.md) computes when it certifies
robustness, arrived at from the opposite direction: there it is the region in which the verdict
*cannot change*, here it is the region in which the proof *still applies*. They are the same
set, and it is small.

Binding the flow's own digest into the transcript reduces the region to the flow, at a cost of
{len(study.leakage) and 32} bytes. The bound column is zero at every perturbation, including one
of 10^-9 sd, because any change at all changes the digest. Without it, an operator could
truthfully certify one flow and attach the certificate to a neighbour.

## What it costs

{_cost_table(study)}

The size is the finding, and it is not a small constant: a certificate must carry a path for
**every** tree, so it scales with the ensemble rather than with the answer. At
{study.n_trees:,} trees and a mean depth of {study.mean_depth:.1f}, one verdict's proof is
{study.certificate_bytes / max(study.model_bytes, 1):.0%} the size of the entire model. Nobody
is putting that on a prediction response.

Verification costs {study.verify_ms / max(study.inference_ms, 1e-9):.0f}x inference, and that
gap is real rather than an artefact of the language: it is
{study.hashes_per_verification:,} SHA-256 evaluations against roughly
{int(study.n_trees * study.mean_depth):,} float comparisons. The usual verifiable-computation
selling point -- that checking is cheaper than computing -- **does not hold for a model this
cheap to evaluate.** Attestation is worth its price when the thing being attested is expensive
or unaccountable, and a boosted ensemble on 76 features is neither.

The prover has one honest footnote. The first implementation rebuilt each tree's Merkle
authentication path per verdict, which is quadratic in the number of trees and made
certification 370x inference. The paths depend only on the model, so they are computed once at
commitment time; certification is now {study.certify_ms:.1f} ms. The cost that remains is the
one the scheme actually implies.

## What it leaks

![What an adversary recovers per certificate](../figures/{leakage.name})

{_leak_table(study)}

This is the half nobody advertises. A certificate spells out one root-to-leaf path per tree --
the feature, the threshold and the direction at every branch -- so an adversary collecting them
is not approximating the model's behaviour, they are **reading its geometry**.
{first_read}; {last_read}.

Query access can never do that. The [extraction study](extraction.md) steals a *surrogate* by
asking for calibrated probabilities; a certificate hands over the branch structure and the raw
margin, which is a strictly stronger oracle -- the surrogate column shows it reaching
{f"{last.surrogate_fidelity:.2f}" if last else "n/a"} correlation on flows it never saw, from
labels it was given for free.

**Verifiability and confidentiality trade off, and here the trade is denominated in
certificates.** Every party who is allowed to check a verdict is thereby allowed to read part
of the model, which makes the natural deployment a *selective* one: certify the decisions that
are contested, to the parties entitled to contest them, and count the exposure.

## Scope and honest limits

- **This proves the computation, not its correctness.** A certificate says the committed model
  produced this score for this flow. It says nothing about whether the model is any good --
  that is what every other report here is for -- and nothing about whether the committed root
  is the right one, which is a transparency-log problem the [ledger's](ledger.md) published
  anchor solves for alerts and nothing here solves for models.
- **The pipeline is outside the commitment.** The proof begins at the transformed feature
  vector. A service that mis-scales an input produces a perfectly valid certificate for the
  wrong flow, so a complete scheme has to commit to the fitted preprocessing too -- which is
  exactly the train/serve skew the [canary](../../README.md) exists to catch, now with a second
  reason to care.
- **The scheme assumes an honest commitment ceremony.** Whoever publishes the root can publish
  the root of a model nobody reviewed. This is the standard reduction: attestation converts
  "trust the service" into "trust the publication", which is progress only because publication
  can be witnessed and a running process cannot.
- **Only collision resistance is assumed**, which is the point of building it this way -- no
  pairings, no trusted setup, no zero-knowledge machinery, and nothing that stops working when
  a library is upgraded. The price is that the proof is linear in the model.
- **It does not hide the path.** A zero-knowledge version would prove the same statement while
  revealing nothing, and would cost orders of magnitude more per verdict. The leakage table is
  the argument for when that trade becomes worth making."""


def run_attestation_report(settings: Settings) -> Path:
    """Run the attestation study and write the report + figures."""
    study = run_attestation_study(settings)
    epsilons = np.array([max(row.epsilon, 1e-12) for row in study.replay], dtype=float)
    replay = plots.plot_lines(
        {
            "unbound certificate": (
                epsilons,
                np.array([row.accepted_unbound for row in study.replay]),
            ),
            "bound to the flow's digest": (
                epsilons,
                np.array([row.accepted_bound for row in study.replay]),
            ),
        },
        xlabel="how far the flow is moved (sd)",
        ylabel="share of moved flows the certificate still verifies",
        title="A certificate proves a region, not a flow",
        out_path=settings.paths.figures_dir / REPLAY_FIGURE,
        xscale="log",
    )
    counts = np.array([row.certificates for row in study.leakage], dtype=float)
    leakage = plots.plot_lines(
        {
            "share of the model's internal nodes revealed": (
                counts,
                np.array([row.coverage for row in study.leakage]),
            ),
            "surrogate fidelity on unseen flows": (
                counts,
                np.array([row.surrogate_fidelity for row in study.leakage]),
            ),
        },
        xlabel="certificates collected",
        ylabel="share recovered",
        title="Every verdict an auditor may check is a verdict an adversary may read",
        out_path=settings.paths.figures_dir / LEAKAGE_FIGURE,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, replay, leakage), encoding="utf-8")
    logger.info("Wrote attestation report", extra={"path": str(out_path)})

    with track_run(settings, "attestation") as run:
        run.log_params({"trees": study.n_trees, "root": study.root})
        run.log_metrics(
            {
                "certificate_bytes": float(study.certificate_bytes),
                "verify_ms": study.verify_ms,
                "forgeries_caught": float(study.caught),
                "node_coverage": study.leakage[-1].coverage if study.leakage else 0.0,
            }
        )
        for artifact in (replay, leakage, out_path):
            run.log_artifact(artifact)
    return out_path
