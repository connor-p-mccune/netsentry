"""A tamper-evident alert ledger: prove the alert history was not edited after the fact.

An intrusion detector's output is evidence. It gets read during an incident review, quoted in a
post-mortem, and occasionally relied on to say what a system did and when. Every one of those
uses assumes the record has not been altered since it was written — and a JSON-lines file on
disk offers no support for that assumption at all. Anyone who can write the file can rewrite
history: delete the alert that fired on the host they compromised, backdate one that makes a
response look faster than it was, or change a verdict from `attack` to `benign` and leave
nothing behind.

This makes that class of edit **detectable**. Each ledger entry carries the hash of the entry
before it, so the file is a hash chain: changing any byte of any past entry changes its hash,
which breaks the link the *next* entry recorded, and the break is visible from a single pass.
Verification names the first sequence number where the chain fails, which is where the edit
happened.

Two properties get treated carefully, because they are where naive implementations quietly fail:

- **Truncation.** Deleting entries from the *end* of a hash chain leaves a perfectly valid
  chain. Nothing internal to the file can detect it. The fix is an **anchor**: the head hash and
  entry count, written somewhere the ledger's writer does not control (another host, a log
  shipper, a signed commit). Verification against an anchor turns tail-deletion from
  undetectable into obvious, and the report demonstrates exactly that gap.
- **Selective disclosure.** Handing a regulator or a customer the whole alert history to prove
  one alert is in it is both impractical and a privacy problem. A **Merkle tree** over the entry
  hashes gives an inclusion proof of size `O(log n)`: the single alert, a handful of sibling
  hashes, and the published root are enough to verify membership without revealing any other
  entry.

The design deliberately stops short of claiming more than it delivers. A hash chain proves
**integrity**, not **authenticity**: it shows the history is internally consistent with its
anchor, not that the entries were written by anyone in particular. Signing the anchor with a key
the writer cannot reach is what adds that, and is named in the report as the next rung rather
than implied by it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from netsentry.log import get_logger

logger = get_logger(__name__)

GENESIS_HASH = "0" * 64
LEDGER_VERSION = 1


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def canonical_json(payload: dict[str, Any]) -> str:
    """A byte-stable serialisation of a payload, so the same alert always hashes the same.

    Key order and whitespace must be fixed or two serialisations of one alert would produce two
    different hashes, and the chain would report tampering on a file nobody touched.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class LedgerEntry:
    """One sealed record: the payload's digest, the previous link, and this entry's hash."""

    seq: int
    recorded_at: str
    prev_hash: str
    payload_hash: str
    entry_hash: str
    payload: dict[str, Any]

    def recompute_hash(self) -> str:
        """Re-derive this entry's hash from its own contents (the verification primitive)."""
        return entry_digest(self.seq, self.recorded_at, self.prev_hash, self.payload_hash)

    def to_json(self) -> str:
        return canonical_json(asdict(self))


def entry_digest(seq: int, recorded_at: str, prev_hash: str, payload_hash: str) -> str:
    """Hash the linking fields together — sequence, time, predecessor, and payload digest.

    The sequence number and timestamp are inside the digest on purpose: without them an entry
    could be moved to a different position, or restamped, without breaking anything.
    """
    return _sha256(f"{LEDGER_VERSION}|{seq}|{recorded_at}|{prev_hash}|{payload_hash}")


class AlertLedger:
    """Append-only, hash-chained ledger of alert records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def head(self) -> tuple[int, str]:
        """The current `(count, head_hash)` — the pair an anchor pins."""
        entries = self.read()
        if not entries:
            return 0, GENESIS_HASH
        return len(entries), entries[-1].entry_hash

    def append(self, payload: dict[str, Any], recorded_at: str | None = None) -> LedgerEntry:
        """Seal one payload onto the end of the chain and flush it to disk."""
        count, prev = self.head()
        stamp = recorded_at or datetime.now(UTC).isoformat()
        payload_hash = _sha256(canonical_json(payload))
        entry = LedgerEntry(
            seq=count,
            recorded_at=stamp,
            prev_hash=prev,
            payload_hash=payload_hash,
            entry_hash=entry_digest(count, stamp, prev, payload_hash),
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")
        return entry

    def extend(self, payloads: list[dict[str, Any]]) -> list[LedgerEntry]:
        """Append many payloads, keeping the chain intact across the batch."""
        return [self.append(payload) for payload in payloads]

    def read(self) -> list[LedgerEntry]:
        """Load every entry in file order (no verification — that is `verify`'s job)."""
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            entries.append(LedgerEntry(**raw))
        return entries

    def verify(self, anchor: Anchor | None = None) -> VerificationResult:
        """Walk the chain and report the first break, if any."""
        return verify_entries(self.read(), anchor)

    def write_anchor(self, path: Path) -> Anchor:
        """Publish the head as an anchor — the only defence against tail deletion."""
        count, head = self.head()
        anchor = Anchor(count=count, head_hash=head, published_at=datetime.now(UTC).isoformat())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(asdict(anchor)), encoding="utf-8")
        return anchor


@dataclass(frozen=True)
class Anchor:
    """A published `(count, head_hash)` pair — what makes truncation detectable."""

    count: int
    head_hash: str
    published_at: str

    @staticmethod
    def load(path: Path) -> Anchor:
        return Anchor(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class VerificationResult:
    """The outcome of a full chain walk."""

    ok: bool
    n_entries: int
    head_hash: str
    failure: str | None = None
    first_bad_seq: int | None = None

    @property
    def summary(self) -> str:
        if self.ok:
            return f"intact ({self.n_entries:,} entries)"
        where = "" if self.first_bad_seq is None else f" at seq {self.first_bad_seq}"
        return f"BROKEN{where}: {self.failure}"


def verify_entries(entries: list[LedgerEntry], anchor: Anchor | None = None) -> VerificationResult:
    """Check sequence order, per-entry hashes, chain links, and (if given) the anchor.

    The checks run in that order because it is the order that localises a fault: a sequence gap
    says an entry was removed, a bad entry hash says its contents were edited, and a bad link
    says something was inserted or reordered between two otherwise-valid entries.
    """
    prev = GENESIS_HASH
    for i, entry in enumerate(entries):
        if entry.seq != i:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure=f"sequence gap: expected {i}, found {entry.seq}",
                first_bad_seq=entry.seq,
            )
        if _sha256(canonical_json(entry.payload)) != entry.payload_hash:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure="payload does not match its recorded digest",
                first_bad_seq=entry.seq,
            )
        if entry.recompute_hash() != entry.entry_hash:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure="entry hash does not match its own contents",
                first_bad_seq=entry.seq,
            )
        if entry.prev_hash != prev:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure="broken link: entry does not follow its predecessor",
                first_bad_seq=entry.seq,
            )
        prev = entry.entry_hash

    if anchor is not None:
        if len(entries) < anchor.count:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure=(
                    f"truncated: anchor pins {anchor.count:,} entries, file holds "
                    f"{len(entries):,}"
                ),
                first_bad_seq=len(entries),
            )
        if len(entries) == anchor.count and prev != anchor.head_hash:
            return VerificationResult(
                ok=False,
                n_entries=len(entries),
                head_hash=prev,
                failure="head hash does not match the published anchor",
                first_bad_seq=len(entries) - 1,
            )
    return VerificationResult(ok=True, n_entries=len(entries), head_hash=prev)


# --------------------------------------------------------------------------------------
# Merkle inclusion: prove one alert was in the ledger without handing over the rest.
# --------------------------------------------------------------------------------------


def _pair(left: str, right: str) -> str:
    return _sha256(f"{left}{right}")


def merkle_root(leaves: list[str]) -> str:
    """Root of a binary Merkle tree over entry hashes (odd levels promote the last node)."""
    if not leaves:
        return GENESIS_HASH
    level = list(leaves)
    while len(level) > 1:
        nxt = [_pair(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def inclusion_proof(leaves: list[str], index: int) -> list[tuple[str, str]]:
    """Sibling hashes proving `leaves[index]` is under the root, as `(side, hash)` pairs."""
    if not 0 <= index < len(leaves):
        raise IndexError("index outside the ledger")
    proof: list[tuple[str, str]] = []
    level = list(leaves)
    idx = index
    while len(level) > 1:
        if idx % 2 == 0 and idx + 1 < len(level):
            proof.append(("right", level[idx + 1]))
        elif idx % 2 == 1:
            proof.append(("left", level[idx - 1]))
        # An odd last node is promoted unchanged, so it needs no sibling at this level.
        nxt = [_pair(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
        idx //= 2
    return proof


def verify_inclusion(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    """Recompute the root from one leaf and its siblings — the whole point of the tree."""
    current = leaf
    for side, sibling in proof:
        current = _pair(sibling, current) if side == "left" else _pair(current, sibling)
    return current == root
