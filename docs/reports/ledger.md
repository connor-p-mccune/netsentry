# NetSentry — Tamper-Evident Alert Ledger

_Synthetic stand-in. 500 real alerts from the temporal split at the deployed
operating point, sealed into `data/ledger/alerts.jsonl`. Head hash
`ed326a66a547eb9d...`, anchored in `data/ledger/anchor.json`._

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

| attempted edit | what it does | caught by the chain alone | caught with an anchor | what verification reports |
|---|---|---|---|---|
| flip a verdict | rewrite one alert's decision from attack to benign | yes | yes | payload does not match its recorded digest |
| flip a verdict and reseal it | edit the payload *and* recompute its digest, as a careful attacker would | yes | yes | entry hash does not match its own contents |
| delete an alert | remove one alert from the middle of the history | yes | yes | sequence gap: expected 250, found 251 |
| reorder two alerts | swap two adjacent entries to change the apparent timeline | yes | yes | sequence gap: expected 250, found 251 |
| backdate an alert | restamp one entry so it appears to predate the incident | yes | yes | entry hash does not match its own contents |
| truncate the tail | delete every alert after a point, leaving a chain that is internally valid | **no** | yes | truncated: anchor pins 500 entries, file holds 250 |

Every edit that touches the *body* of the history is caught by the chain alone, including the careful version where the attacker recomputes the payload digest after editing it — that repair fixes one hash and invalidates the entry hash that covers it, which in turn invalidates the link the next entry recorded. **truncate the tail** is the exception, and it is a structural one rather than an oversight: deleting entries from the end of a hash chain leaves a chain that is perfectly valid, because nothing inside the file records how long the file was supposed to be. No amount of hashing fixes this. What fixes it is publishing the head — the `(count, head_hash)` pair in `data/ledger/anchor.json` — somewhere the ledger's writer cannot reach, at which point the same truncation is detected immediately and reported as exactly what it is. With the anchor in place all 6 attacks are detected, and each is localised to the sequence number where the history stops being consistent, which is where the investigation starts.

## Proving one alert without disclosing the rest

Handing over the entire alert history to prove that one alert exists is impractical and, on real
traffic, a privacy problem. A Merkle tree over the entry hashes gives an inclusion proof instead:
for alert 166 of 500, the proof is **9
sibling hashes** — logarithmic in the ledger size — and recomputing the root from the alert plus
those siblings reproduces `9984f9b737c6e2e8...` exactly. Verification of the genuine
leaf: **passes**. Verification of a forged leaf against
the same proof: **correctly rejected**.
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
ambiguity that has bitten more than one production tree.