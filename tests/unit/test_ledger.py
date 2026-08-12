"""The alert ledger: chain integrity, every tamper case, anchors, and Merkle inclusion.

A tamper-evident structure is only worth what it detects, so each test is an attack rather than
an assertion about shape: edit a payload, reseal it carefully, delete an entry, reorder two,
restamp one, truncate the tail. The truncation test pins the *limitation* — a hash chain cannot
see it — and then pins that an anchor can.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from netsentry.governance.ledger import (
    GENESIS_HASH,
    AlertLedger,
    canonical_json,
    inclusion_proof,
    merkle_root,
    verify_entries,
    verify_inclusion,
)


@pytest.fixture
def ledger(tmp_path: Path) -> AlertLedger:
    led = AlertLedger(tmp_path / "alerts.jsonl")
    led.extend([{"flow": i, "verdict": "attack", "score": 0.9} for i in range(8)])
    return led


def test_canonical_json_is_stable_under_key_order() -> None:
    # Two spellings of the same alert must hash identically, or verification would report
    # tampering on a file nobody touched.
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_a_fresh_ledger_starts_from_the_genesis_hash(tmp_path: Path) -> None:
    count, head = AlertLedger(tmp_path / "empty.jsonl").head()
    assert (count, head) == (0, GENESIS_HASH)


def test_appending_links_each_entry_to_its_predecessor(ledger: AlertLedger) -> None:
    entries = ledger.read()
    assert entries[0].prev_hash == GENESIS_HASH
    assert all(entries[i].prev_hash == entries[i - 1].entry_hash for i in range(1, len(entries)))
    assert [e.seq for e in entries] == list(range(len(entries)))


def test_an_untouched_ledger_verifies(ledger: AlertLedger) -> None:
    result = ledger.verify()
    assert result.ok
    assert result.n_entries == 8
    assert "intact" in result.summary


def test_editing_a_payload_is_caught_and_localised(ledger: AlertLedger) -> None:
    entries = ledger.read()
    entries[3].payload["verdict"] = "benign"
    result = verify_entries(entries)
    assert not result.ok
    assert result.first_bad_seq == 3
    assert "digest" in result.failure


def test_resealing_the_payload_digest_does_not_save_the_attacker(ledger: AlertLedger) -> None:
    # A careful attacker edits the payload *and* recomputes its digest. That repairs one hash
    # and invalidates the entry hash that covers it.
    import hashlib

    entries = ledger.read()
    entries[4].payload["verdict"] = "benign"
    object.__setattr__(
        entries[4],
        "payload_hash",
        hashlib.sha256(canonical_json(entries[4].payload).encode()).hexdigest(),
    )
    result = verify_entries(entries)
    assert not result.ok
    assert result.first_bad_seq == 4
    assert "entry hash" in result.failure


def test_deleting_an_entry_from_the_middle_is_caught(ledger: AlertLedger) -> None:
    entries = ledger.read()
    del entries[2]
    result = verify_entries(entries)
    assert not result.ok
    assert "sequence gap" in result.failure


def test_reordering_two_entries_is_caught(ledger: AlertLedger) -> None:
    entries = ledger.read()
    entries[5], entries[6] = entries[6], entries[5]
    assert not verify_entries(entries).ok


def test_backdating_an_entry_is_caught_because_the_hash_covers_the_stamp(
    ledger: AlertLedger,
) -> None:
    entries = ledger.read()
    object.__setattr__(entries[1], "recorded_at", "2000-01-01T00:00:00+00:00")
    result = verify_entries(entries)
    assert not result.ok
    assert result.first_bad_seq == 1


def test_truncating_the_tail_is_invisible_to_the_chain_alone(ledger: AlertLedger) -> None:
    # The structural limitation, pinned deliberately: nothing inside the file records how long
    # the file was supposed to be, so a prefix of a valid chain is a valid chain.
    assert verify_entries(ledger.read()[:4]).ok


def test_an_anchor_turns_truncation_into_a_detected_edit(
    ledger: AlertLedger, tmp_path: Path
) -> None:
    anchor = ledger.write_anchor(tmp_path / "anchor.json")
    result = verify_entries(ledger.read()[:4], anchor)
    assert not result.ok
    assert "truncated" in result.failure


def test_an_anchor_matches_an_untouched_ledger(ledger: AlertLedger, tmp_path: Path) -> None:
    anchor = ledger.write_anchor(tmp_path / "anchor.json")
    assert anchor.count == 8
    assert verify_entries(ledger.read(), anchor).ok


def test_a_rewritten_head_is_caught_against_the_anchor(ledger: AlertLedger, tmp_path: Path) -> None:
    anchor = ledger.write_anchor(tmp_path / "anchor.json")
    entries = copy.deepcopy(ledger.read())
    entries[-1].payload["verdict"] = "benign"
    assert not verify_entries(entries, anchor).ok


def test_the_ledger_file_is_one_json_object_per_line(ledger: AlertLedger) -> None:
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert all(json.loads(line)["seq"] == i for i, line in enumerate(lines))


def test_merkle_root_of_an_empty_tree_is_the_genesis_hash() -> None:
    assert merkle_root([]) == GENESIS_HASH


def test_merkle_root_of_a_single_leaf_is_that_leaf() -> None:
    assert merkle_root(["ab" * 32]) == "ab" * 32


def test_merkle_root_changes_when_any_leaf_changes() -> None:
    leaves = [f"{i:064x}" for i in range(7)]
    other = list(leaves)
    other[3] = f"{99:064x}"
    assert merkle_root(leaves) != merkle_root(other)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_every_leaf_has_a_verifying_inclusion_proof(size: int) -> None:
    # Odd sizes are the interesting ones: an unpaired node is promoted, so the proof for it
    # is shorter than for its siblings.
    leaves = [f"{i:064x}" for i in range(size)]
    root = merkle_root(leaves)
    for i in range(size):
        assert verify_inclusion(leaves[i], inclusion_proof(leaves, i), root)


def test_a_forged_leaf_is_rejected_by_a_genuine_proof() -> None:
    leaves = [f"{i:064x}" for i in range(9)]
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 4)
    assert not verify_inclusion("f" * 64, proof, root)


def test_inclusion_proof_is_logarithmic_in_the_ledger_size() -> None:
    leaves = [f"{i:064x}" for i in range(1024)]
    assert len(inclusion_proof(leaves, 500)) == 10  # log2(1024)


def test_inclusion_proof_rejects_an_index_outside_the_ledger() -> None:
    with pytest.raises(IndexError, match="outside the ledger"):
        inclusion_proof([f"{i:064x}" for i in range(4)], 9)
