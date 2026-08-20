# NetSentry — The Serving Lifecycle, as a State Machine

_A model of what the API is allowed to do, driven against the real application for
200 random operations, then re-run against 5 deliberately
broken versions of the same service. Regenerate with `netsentry statemachine`._

## Why this report exists

Every part of the serving lifecycle has a test: the bundle loads, the canary replays, the guard
rejects a missing key, a reload with a bad path is refused. What none of them covers is the
part that breaks in production -- the **sequences**. A reload that half-succeeds, an
authentication check that stops applying after a swap, a health endpoint still reporting the
version it used to serve: each is a two-step bug, and a suite of single-step tests is
structurally unable to see any of them.

So the contract is written down as a model -- a state machine holding only what an observer can
check -- and the real service is driven through random sequences, with model and service
compared after every single step.

## The properties

1. **A refused reload changes nothing.** Not the version, not the health, not the next
   prediction. A 409 that leaves a half-swapped engine looks identical to a clean refusal until
   the next request arrives.
2. **Only a successful reload may change the served version.** Every response that names a
   version is checked against the version the model believes is live.
3. **A refusal is a refusal.** A canary mismatch, a missing bundle and a path outside the
   models directory must each produce their own error status and never a success.
4. **Health never claims `ok` while its own canary is failing**, since that is the signal every
   orchestrator uses to decide whether to send traffic.
5. **The guard applies to the guarded routes in every state**, including after a swap.

## The run

Starting from model version `0.2.0`, the machine performed
200 operations: no disagreement between the model and the service.

| operation | times exercised |
|---|---|
| predict | 21 |
| predict batch | 24 |
| predict without the API key | 19 |
| health | 28 |
| metrics | 19 |
| reload a good bundle | 4 |
| reload a bundle whose canaries do not reproduce | 4 |
| reload a bundle that is not there | 29 |
| reload a path outside the models dir | 34 |
| predict with a malformed flow | 18 |

The mix is not realistic and is not meant to be. A production trace is 99% predictions, which is
the sequence single-request tests already cover; the interesting transitions are the rare ones,
so the schedule allocates them deliberately.

That allocation replaced a weighted random draw, and the reason is worth recording. Two of the
ten operations construct an entire inference engine and take seconds rather than milliseconds,
so they have to be drawn less often -- and under a weighted draw the headline run came back with
**zero** successful reloads. The most important positive transition in the lifecycle went
unexercised, and the report would have said nothing about it while looking complete. The
schedule now allocates the expensive operations explicitly, fills the remainder with cheap ones
and shuffles: the order stays random, the coverage stops being a matter of luck.

## Does the machine catch anything?

A conformance machine that has never failed is indistinguishable from one that cannot fail, so
each mutant below is a specific regression injected into the service's responses -- the
observable symptom of a real bug -- and the identical walk is re-run against it.

**5 of 5 caught.**

| injected regression | what it imitates | steps that disagreed | first disagreement |
|---|---|---|---|
| swap before the canary gate | the candidate is installed and *then* validated, so a bundle this runtime cannot reproduce ends up serving traffic | 4 | step 38: reload a bundle whose canaries do not reproduce: expected HTTP 409 or 422, got 200 |
| stale version after a swap | the engine is replaced but /health keeps reporting the version it used to serve, so an operator cannot tell which model answered | 28 | step 4: health: served version '0.0.0-stale' but the model is on '0.2.0' |
| the guard stops applying | authentication passes on a route that must require it -- the hole a refactor leaves and no single-request test notices | 19 | step 5: predict without the API key: expected HTTP 401, got 200 |
| a refusal reported as success | a missing or out-of-tree bundle returns 200, which is what a swallowed exception looks like from outside | 63 | step 0: reload a bundle that is not there: expected HTTP 404, got 200 |
| health that lies about its canary | status ok while the embedded canary did not reproduce: the readiness probe every orchestrator trusts | 28 | step 4: health: reported ok while its canary was failing |

The mutants are injected at the HTTP boundary rather than inside the application. That buys
determinism and costs realism: it proves the model notices the *symptom* of each regression,
not that the code path producing it is reachable. The symptom is what a monitoring system sees,
which is the same reason the properties are stated in terms of an observer.

## Scope and honest limits

- **The walk is random, not exhaustive.** 200 operations over
  10 verbs cover the pairs and most triples that matter, and prove nothing about
  a rare interleaving nobody drew. The Hypothesis-driven version of this machine lives in the
  test suite, where a failure shrinks to a minimal sequence instead of a long one.
- **The model is deliberately weaker than the implementation.** It holds a version, a canary
  state and a key -- not thresholds, not the engine, not the shadow model. A model that
  mirrored the implementation could not disagree with it, and a model that cannot disagree
  cannot find anything.
- **Concurrency is not tested here.** Every operation is sequential. The claim that an in-flight
  request finishes on the model it started with rests on the swap being a single reference
  reassignment, which this checks by inspection rather than by racing it.
- **A canary-failing bundle is manufactured by perturbing the stored expectations**, which
  reproduces the symptom of an environment mismatch without needing a second training run to
  create one. The gate cannot tell the two apart, and that is the point of the gate.