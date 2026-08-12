"""Build a real alert ledger from the deployed model, then try to tamper with it.

A tamper-evident design is worth exactly as much as the attacks it has been run against, so this
does not describe the property — it builds the ledger from the temporal split's actual alerts and
then executes each edit an attacker with write access would attempt, recording whether
verification caught it and where. The truncation row is the interesting one: it is *undetectable*
from inside the file and detected immediately once an anchor exists, which is the argument for
publishing one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.governance.ledger import (
    AlertLedger,
    Anchor,
    LedgerEntry,
    inclusion_proof,
    merkle_root,
    verify_entries,
    verify_inclusion,
)
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "ledger.md"


@dataclass
class TamperResult:
    """One attempted edit and what verification made of it."""

    attack: str
    description: str
    detected_without_anchor: bool
    detected_with_anchor: bool
    failure: str


def _build_alerts(settings: Settings, limit: int) -> list[dict[str, Any]]:
    """Score the temporal test split and turn the alerting flows into ledger payloads."""
    from netsentry.data.split import load_split

    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    model = SupervisedClassifier(variant).fit(
        x_train,
        train[BINARY_TARGET].to_numpy().astype(int),
        eval_set=(x_val, val[BINARY_TARGET].to_numpy().astype(int)),
    )
    s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
    s_test = attack_probability(np.asarray(model.predict_proba(x_test)), model.classes_, benign)
    threshold = threshold_at_fpr(
        val[BINARY_TARGET].to_numpy().astype(int), s_val, variant.thresholds.primary_fpr
    )
    flagged = np.flatnonzero(s_test >= threshold)[:limit]
    labels = test["Label"].to_numpy() if "Label" in test.columns else np.array(["?"] * len(test))
    return [
        {
            "flow_index": int(i),
            "attack_probability": round(float(s_test[i]), 6),
            "verdict": "attack",
            "ground_truth": str(labels[i]),
        }
        for i in flagged
    ]


def _tamper_cases(entries: list[LedgerEntry], anchor: Anchor) -> list[TamperResult]:
    """Every edit an attacker with write access to the ledger file would try."""
    mid = len(entries) // 2
    cases: list[tuple[str, str, list[LedgerEntry]]] = []

    edited = copy.deepcopy(entries)
    edited[mid].payload["verdict"] = "benign"
    cases.append(
        (
            "flip a verdict",
            "rewrite one alert's decision from attack to benign",
            edited,
        )
    )

    resealed = copy.deepcopy(entries)
    resealed[mid].payload["attack_probability"] = 0.01
    object.__setattr__(  # a careful attacker also fixes the payload digest
        resealed[mid],
        "payload_hash",
        __import__("hashlib")
        .sha256(
            __import__("json")
            .dumps(resealed[mid].payload, sort_keys=True, separators=(",", ":"), default=str)
            .encode()
        )
        .hexdigest(),
    )
    cases.append(
        (
            "flip a verdict and reseal it",
            "edit the payload *and* recompute its digest, as a careful attacker would",
            resealed,
        )
    )

    deleted = copy.deepcopy(entries)
    del deleted[mid]
    cases.append(("delete an alert", "remove one alert from the middle of the history", deleted))

    reordered = copy.deepcopy(entries)
    reordered[mid], reordered[mid + 1] = reordered[mid + 1], reordered[mid]
    cases.append(
        (
            "reorder two alerts",
            "swap two adjacent entries to change the apparent timeline",
            reordered,
        )
    )

    backdated = copy.deepcopy(entries)
    object.__setattr__(backdated[mid], "recorded_at", "2020-01-01T00:00:00+00:00")
    cases.append(
        ("backdate an alert", "restamp one entry so it appears to predate the incident", backdated)
    )

    truncated = copy.deepcopy(entries)[: len(entries) // 2]
    cases.append(
        (
            "truncate the tail",
            "delete every alert after a point, leaving a chain that is internally valid",
            truncated,
        )
    )

    results = []
    for name, description, mutated in cases:
        plain = verify_entries(mutated)
        anchored = verify_entries(mutated, anchor)
        results.append(
            TamperResult(
                attack=name,
                description=description,
                detected_without_anchor=not plain.ok,
                detected_with_anchor=not anchored.ok,
                failure=(anchored.failure or plain.failure or "—"),
            )
        )
    return results


@dataclass
class LedgerStudy:
    """The built ledger, the tamper matrix, and the inclusion-proof demonstration."""

    n_alerts: int
    head_hash: str
    merkle_root: str
    proof_length: int
    proof_verifies: bool
    forged_proof_rejected: bool
    ledger_path: Path
    anchor_path: Path
    results: list[TamperResult]


def run_ledger_study(settings: Settings) -> LedgerStudy:
    """Build the ledger from real alerts, publish an anchor, and run every tamper case."""
    cfg = settings.ledger
    ledger_path = Path(cfg.path)
    anchor_path = Path(cfg.anchor_path)
    if ledger_path.exists():
        ledger_path.unlink()  # the demonstration rebuilds from scratch, deterministically
    ledger = AlertLedger(ledger_path)
    payloads = _build_alerts(settings, cfg.demo_alerts)
    entries = [
        ledger.append(p, recorded_at=f"2026-07-0{1 + i // 500}T00:{i % 60:02d}:00+00:00")
        for i, p in enumerate(payloads)
    ]
    anchor = ledger.write_anchor(anchor_path)

    leaves = [e.entry_hash for e in entries]
    root = merkle_root(leaves)
    target = len(leaves) // 3
    proof = inclusion_proof(leaves, target)
    verifies = verify_inclusion(leaves[target], proof, root)
    # A forged leaf must not verify against the same proof — the property that makes the proof
    # worth anything at all.
    forged = "f" * 64
    forged_rejected = not verify_inclusion(forged, proof, root)

    return LedgerStudy(
        n_alerts=len(entries),
        head_hash=anchor.head_hash,
        merkle_root=root,
        proof_length=len(proof),
        proof_verifies=verifies,
        forged_proof_rejected=forged_rejected,
        ledger_path=ledger_path,
        anchor_path=anchor_path,
        results=_tamper_cases(entries, anchor),
    )


def run_ledger_report(settings: Settings) -> Path:
    """Run the tamper study and write the report."""
    study = run_ledger_study(settings)
    report = _render(study)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote ledger report", extra={"path": str(out_path)})

    with track_run(settings, "ledger") as run:
        run.log_metrics(
            {
                "alerts": float(study.n_alerts),
                "tamper_cases": float(len(study.results)),
                "detected_with_anchor": float(sum(r.detected_with_anchor for r in study.results)),
                "detected_without_anchor": float(
                    sum(r.detected_without_anchor for r in study.results)
                ),
            }
        )
        run.log_artifact(out_path)
    return out_path


def _tamper_table(study: LedgerStudy) -> str:
    rows = [
        "| attempted edit | what it does | caught by the chain alone | caught with an anchor "
        "| what verification reports |",
        "|---|---|---|---|---|",
    ]
    for r in study.results:
        plain = "yes" if r.detected_without_anchor else "**no**"
        anchored = "yes" if r.detected_with_anchor else "**no**"
        rows.append(f"| {r.attack} | {r.description} | {plain} | {anchored} | {r.failure} |")
    return "\n".join(rows)


def _read(study: LedgerStudy) -> str:
    chain_only = [r for r in study.results if not r.detected_without_anchor]
    all_caught = all(r.detected_with_anchor for r in study.results)
    parts = [
        "Every edit that touches the *body* of the history is caught by the chain alone, "
        "including the careful version where the attacker recomputes the payload digest after "
        "editing it — that repair fixes one hash and invalidates the entry hash that covers it, "
        "which in turn invalidates the link the next entry recorded."
    ]
    if chain_only:
        names = ", ".join(f"**{r.attack}**" for r in chain_only)
        parts.append(
            f"{names} is the exception, and it is a structural one rather than an oversight: "
            "deleting entries from the end of a hash chain leaves a chain that is perfectly "
            "valid, because nothing inside the file records how long the file was supposed to "
            "be. No amount of hashing fixes this. What fixes it is publishing the head — the "
            f"`(count, head_hash)` pair in `{study.anchor_path.as_posix()}` — somewhere the "
            "ledger's writer cannot reach, at which point the same truncation is detected "
            "immediately and reported as exactly what it is."
        )
    if all_caught:
        parts.append(
            f"With the anchor in place all {len(study.results)} attacks are detected, and each "
            "is localised to the sequence number where the history stops being consistent, "
            "which is where the investigation starts."
        )
    return " ".join(parts)


def _render(study: LedgerStudy) -> str:
    return f"""# NetSentry — Tamper-Evident Alert Ledger

_Synthetic stand-in. {study.n_alerts:,} real alerts from the temporal split at the deployed
operating point, sealed into `{study.ledger_path.as_posix()}`. Head hash
`{study.head_hash[:16]}...`, anchored in `{study.anchor_path.as_posix()}`._

## Why this report exists

A detector's output is evidence. It is read during incident review, quoted in post-mortems, and
occasionally relied on to establish what a system did and when. All of those uses assume the
record has not been altered since it was written, and a JSON-lines file on disk supports that
assumption not at all: anyone who can write the file can delete the alert that fired on the host
they compromised, or change a verdict from `attack` to `benign`, and leave nothing behind.

Hash-chaining the ledger makes that class of edit **detectable**. Each entry carries the digest
of the entry before it, so altering any past byte breaks the link its successor recorded, and a
single pass finds it. The claim is deliberately narrow: this proves **integrity**, not
**authenticity** — that the history is internally consistent with its published head, not that
any particular party wrote it.

## The attacks, executed

Each row below is run against the real ledger built above, not described.

{_tamper_table(study)}

{_read(study)}

## Proving one alert without disclosing the rest

Handing over the entire alert history to prove that one alert exists is impractical and, on real
traffic, a privacy problem. A Merkle tree over the entry hashes gives an inclusion proof instead:
for alert {study.n_alerts // 3} of {study.n_alerts:,}, the proof is **{study.proof_length}
sibling hashes** — logarithmic in the ledger size — and recomputing the root from the alert plus
those siblings reproduces `{study.merkle_root[:16]}...` exactly. Verification of the genuine
leaf: **{"passes" if study.proof_verifies else "FAILS"}**. Verification of a forged leaf against
the same proof: **{"correctly rejected" if study.forged_proof_rejected else "WRONGLY ACCEPTED"}**.
That is the whole property — a third party can confirm membership holding only the one record
they were given and the published root.

## Where this sits in the deployment

The spool watcher seals every alert it emits, so the ledger is written on the path that already
produces SIEM documents rather than as a separate bookkeeping step; `netsentry ledger verify`
walks the chain and exits non-zero on a break, which makes it usable as a cron check or a CI
gate. It complements the [provenance manifest](provenance.md), which attests the *model* that
produced the verdicts, and the [serving canary](metamorphic.md), which attests that the model
still behaves as it did when it was attested. Together they cover the three questions an auditor
asks: which model, behaving how, producing what.

## Scope

Integrity is not authenticity, and this is the honest boundary of the design: an attacker who
can rewrite the ledger *and* the anchor can rewrite history consistently. Closing that requires
the anchor to be signed with a key the writing host does not hold, or published to an append-only
external service — the same argument that leads real deployments to ship logs off-host within
seconds of writing them. Timestamps are recorded, not proven; a trusted timestamping authority
(RFC 3161) is the standard next rung and would turn "backdated" from *detected because the hash
covers the stamp* into *impossible to assert in the first place*. The Merkle construction
promotes an unpaired final node rather than duplicating it, which avoids the duplicate-leaf
ambiguity that has bitten more than one production tree."""
