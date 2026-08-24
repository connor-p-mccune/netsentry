# NetSentry — Is the Seed Enough?

_One thing changed at a time against a reference fit, with the serialised model, its raw
margins, its verdicts at the 24,957-flow operating point and its PR-AUC all compared
exactly. Regenerate with `netsentry determinism`._

## Why this report exists

`.claude/rules/ml.md` states the invariant plainly: *a run must be re-creatable from its logged
config and seed*. Three mechanisms take that literally and hash the result -- the
[integrity manifest](provenance.md) and its `netsentry verify` gate, and the
[attestation root](attestation.md) that a proof-carrying verdict has to fold into. Each is a
claim that the same inputs produce the same bytes, and nobody had checked it.

**The seed is not enough, and the thing it fails to pin does not change the model.**

Of the 7 things changed one at a time, 3 produce a different file and **0 produce a different function**. The row order does not matter; a round trip through disk does not matter; the batch size predictions are made in does not matter. The **thread count** does --  and only to the bytes. The entire difference is `-[num_threads: 14]` and `+[num_threads: 1]`.

Every model here scores identically. The raw margins are bit-for-bit equal, the PR-AUC agrees to four decimal places, and not one of the 1,411 alerts on 24,957 flows changes. What moves is one line of the serialised model recording the machine's core count, because `n_jobs: -1` is not a configuration value -- it is a *lookup*, resolved from the host.

That is the whole finding, and it is worth stating carefully because it cuts both ways. **Byte reproducibility and behavioural reproducibility are different properties.** This project's integrity manifest hashes the bytes, so a bundle rebuilt on a machine with a different core count fails `netsentry verify` while being the same detector. And the [attestation root](attestation.md) -- built two studies ago for an unrelated reason -- commits to the ensemble's *trees* rather than to its parameter block, so it is **stable exactly where the file hash is not**.

## What survives, and what does not

| what changed | same bytes | same function | same scores, bit for bit | verdicts changed | PR-AUC delta |
|---|---|---|---|---|---|
| nothing (a second identical fit) | yes | yes | yes | 0 | +0.0000 |
| the order of the training rows | yes | yes | yes | 0 | +0.0000 |
| the thread count (`n_jobs = 1`) | **no** | yes | yes | 0 | +0.0000 |
| the thread count (`n_jobs = 2`) | **no** | yes | yes | 0 | +0.0000 |
| the thread count (`n_jobs = 4`) | **no** | yes | yes | 0 | +0.0000 |
| a round trip through disk | yes | yes | yes | 0 | +0.0000 |
| the batch size predictions are made in | yes | yes | yes | 0 | +0.0000 |

The row-order line is worth its own sentence, because it is the one that could have been much
worse. The order of rows in a parquet file is not a configuration value -- it is an artefact of
how the file was written -- so a model that depended on it would be reproducible only by
accident. It does not: the same rows shuffled produce byte-identical output.

The last two lines cover the serving path, where a float reduction could plausibly reorder:
saving a model and reading it back, and scoring one row at a time instead of a whole matrix.
Both come back bit-identical, which is a small result that the [batching study](batching.md)
had been assuming.

## Which guarantees depend on which property

| mechanism | what it hashes | across a thread-count change | consequence |
|---|---|---|---|
| the integrity manifest (`netsentry provenance`) | the bundle file's SHA-256 | **breaks** | a rebuild on a machine with a different core count fails `netsentry verify` |
| proof-carrying verdicts (`netsentry attest`) | a Merkle root over the ensemble's *trees* | holds | unaffected: the parameter block is not part of the commitment |
| the behavioural digest introduced here | the serialised model minus environment-recorded parameters | holds | answers 'is this the same function' rather than 'is this the same file' |
| the release gate and promotion checks | metrics, not bytes | holds | unaffected while the verdicts are unchanged |

This is the table the study exists for. The mechanisms are not interchangeable, and the one
that breaks is the one nearest the word "provenance".

The integrity manifest is *right* to hash the file: its question is "are these the bytes that
were reviewed", and for detecting a swapped artifact at rest, nothing else will do. But it is
not the same question as "is this the model that was reviewed", and on a rebuild those two
answers diverge for a reason that has nothing to do with the model.

The attestation root does not have this problem, and not by foresight: it commits to the
ensemble's decision trees, and the parameter block simply is not part of what it hashes. A
commitment to the *computation* turns out to be more portable than a commitment to the
*artifact*.

## The fix, and what it costs

The narrow fix is to stop letting a configuration value resolve from the host -- pin
`n_jobs` to a number, and the bytes become a function of the config again:

| setting | fit time | against the default |
|---|---|---|
| `n_jobs = -1` | 3.3 s | 1.00x |
| `n_jobs = 1` | 5.1 s | 1.58x |
| `n_jobs = 2` | 3.2 s | 0.98x |
| `n_jobs = 4` | 2.2 s | 0.67x |

At 5.1 s against 3.3 s, pinning to a single thread costs
1.6x the fit time, which is a real price for a training loop
this project runs dozens of times per wave.

The better fix is to stop asking one digest two questions. `netsentry provenance` now records a
**behavioural digest** beside the file hash: the serialised model with environment-recorded
parameters removed. The file hash still answers "are these the bytes"; the behavioural digest
answers "is this the same function", and the two disagreeing is itself informative -- it means
the model was rebuilt somewhere else and is otherwise unchanged, which is exactly the situation
a single hash reports as tampering.

## Scope and honest limits

- **One machine, one platform, one library version.** Everything here varies the thread count
  *within* a machine. A genuinely different host -- another CPU, another BLAS, another LightGBM
  build -- could move the trees themselves, and this study cannot see that. What it establishes
  is that the thread count alone is enough to break a byte-level claim, which was the cheapest
  possible way for it to break.
- **The behavioural digest is a stripped serialisation, not a semantic equivalence.** Two models
  that compute the same function through different trees would still disagree, correctly for a
  provenance record and unhelpfully for anyone hoping it means "equivalent".
- **The finding is a near-miss, not a disaster.** No verdict changes anywhere in this study. It
  is in the reports because a mechanism that fires on a difference that does not exist is a
  mechanism people learn to override, and an integrity gate that has been overridden once is not
  an integrity gate.
- **`deterministic=True` is doing something, and this does not isolate what.** LightGBM's flag
  guards run-to-run variation for a fixed configuration; the reference fits agree with it on and
  off, so its value shows up in configurations this study did not construct.