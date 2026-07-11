# A ten-minute tour for reviewers

You are probably here to answer one question: *is this real engineering, or a
notebook with a good README?* This page is the fastest route to your own verdict.
Every stop pairs a claim with the code that implements it, the test that enforces
it, and the artifact it produced — so nothing has to be taken on faith.

## Stop 1 — The headline number is deliberately lower than everyone else's

Most public CIC-IDS2017 projects report ~99.9% accuracy; that number is almost
always leakage plus a shuffled split. NetSentry's headline is **PR-AUC 0.529 on a
temporal split**, reported next to the optimistic shuffled number (0.786) so the
**+0.257 over-optimism gap is the finding**, with a bootstrap CI and p-value.

- Claim & numbers: [`README.md`](../README.md#headline-results), full report
  [`reports/evaluation.md`](reports/evaluation.md)
- The gap is model-agnostic — and it flips the podium: every family from naive
  Bayes to LightGBM pays it, and the honest split crowns a *different winner*
  than the optimistic one ([`reports/leaderboard.md`](reports/leaderboard.md))
- Split machinery: `netsentry/data/split.py` (temporal / stratified /
  leave-one-attack-out, content-hashed persistence)
- Enforced by: `tests/unit/test_split.py` (disjointness, temporal ordering)

## Stop 2 — Leakage is prevented structurally, tested, and re-checked at release

The feature pipeline is a `ColumnTransformer` with `remainder="drop"`: only
explicitly-listed behaviour columns can ever reach a model. Identifiers, IPs,
timestamps, and (deliberately) `Destination Port` are dropped; every transformer is
fit on the training split only.

- Firewall: `netsentry/features/pipeline.py`; column contract in
  `netsentry/data/schema.py`
- Enforced by: `tests/unit/test_features.py` (identifier columns injected and
  asserted gone; imputer statistics proven to come from train only)
- Re-checked on the artifact that ships: `netsentry gate` re-runs the leak check on
  the *fitted* feature space and **fails a PR-AUC above 0.999** as suspected
  leakage — [`reports/gate.md`](reports/gate.md)

## Stop 3 — The metrics themselves are tested

A wrong metric implementation silently invalidates every number downstream, so
PR-AUC, TPR-at-fixed-FPR, and the per-class report are unit-tested against
hand-computed confusion matrices (`tests/unit/test_metrics.py` over
`netsentry/evaluation/metrics.py`), and the headline numbers carry
percentile-bootstrap CIs (`netsentry/evaluation/confidence.py`). The operating
points are then stress-read at deployment prevalences — Axelsson's base-rate
fallacy, computed rather than cited: below a 0.64% prevalence the queue is
majority-false, and a 90%-precision queue at 1-in-10⁵ would need an FPR ~5,800×
tighter than measured ([`reports/base_rate.md`](reports/base_rate.md)).

## Stop 4 — The adversary is measured, and so is the fix

- Evasion: full feature-space mimicry collapses detection ~83% → ~0%
  ([`reports/robustness.md`](reports/robustness.md)) — measured, not hand-waved.
- The fix, re-measured: adversarial training recovers full-mimicry detection at a
  stated clean-performance cost, and the report says what it does *not* defend
  ([`reports/hardening.md`](reports/hardening.md)).
- The training-time adversary: label flips barely move PR-AUC while the operating
  point collapses 21% → 1.8% ([`reports/poisoning.md`](reports/poisoning.md)) —
  the ranking-vs-operating-point thesis in the security dimension.

## Stop 5 — The lifecycle layer is machinery, not slideware

Every stage between training and production is an exit-coded command
([README section](../README.md#model-lifecycle-what-happens-after-the-metrics-table)):

- `netsentry seeds` — same-seed refits are bit-identical (asserted), cross-seed
  noise is measured ([`reports/seed_variance.md`](reports/seed_variance.md)) and
  *used*: it calibrates the promotion margins.
- `netsentry promote` — paired-bootstrap champion/challenger with a SHA-256-pinned
  registry. Read [`reports/promotion.md`](reports/promotion.md): the first real
  decision was a **HOLD** because a PR-AUC-equivalent retrain shipped 1.5pp less
  detection at the operating point. The policy logic is pure and unit-tested
  (`tests/unit/test_promotion.py`).
- `netsentry canary` — `verify` attests the artifact's bytes; canaries attest its
  *behavior*: bundles embed validation flows + build-time scores that the serving
  runtime must reproduce (`netsentry/serving/canary.py`, surfaced on `/health`).
- `netsentry retrainpolicy` — drift-triggered retraining priced against calendar
  retraining; the trigger **under-delivers** on this stream and the report keeps
  that finding ([`reports/retrain_policy.md`](reports/retrain_policy.md)).
- `netsentry refresh` — the label-cheap lever (re-choose only the threshold)
  priced against retraining: it buys **~1% of the recovery** here and does not
  even win budget compliance on this stable stream — a kept double negative
  ([`reports/refresh.md`](reports/refresh.md)). Its counterpart for the
  *guarantee* layer: adaptive conformal steers alpha online and restores the
  attack coverage the temporal shift broke (64% → 89.7%), priced in review load
  ([`reports/adaptive_conformal.md`](reports/adaptive_conformal.md)).

## Stop 6 — Serving is a product surface, not an afterthought

`netsentry/serving/app.py` + `tests/integration/test_serving.py`: pydantic-validated
contract (422s tested), operator-selectable threshold profiles including
`per_service` (a fairness-audit finding shipped as a feature), conformal
`recommended_action` per prediction, SHAP top-features as part of the contract,
API-key auth + rate limiting, Prometheus metrics with bounded label cardinality,
an optional shadow challenger whose disagreement metrics are integration-tested
to be *provably zero* against an identical copy, and opt-in case-based evidence:
`?exemplars=true` returns the nearest known training flows per prediction, from
an index that was **audited before it shipped**
([`reports/exemplars.md`](reports/exemplars.md)). The input side goes all the way
to the wire: `netsentry pcap --demo` parses a raw packet capture (classic pcap or
pcapng, both pure-stdlib), assembles the exact 78 training columns
(`netsentry/capture/`), and scores them through the same engine — no
re-implemented preprocessing to skew — and `netsentry incident` folds the scored
flows into an analyst-ready incident report with ATT&CK context
([`reports/incident_demo.md`](reports/incident_demo.md)).

## Stop 7 — Where the bodies are buried, on purpose

[`NOTES.md`](../NOTES.md) is a running log of self-audits: the gate failing its own
first ECE bar, a report render that assumed a result the numbers contradicted, the
signature ruleset honestly beating the model at one operating point, the anomaly
detector's modest zero-day recall. If you review ML projects for a living, this
file is probably the fastest signal in the repo.

## Run it yourself

```bash
make install
netsentry download && netsentry prep   # synthetic stand-in, out of the box
make lifecycle                         # seeds → gate → promote → retrainpolicy → canary
netsentry analyze                      # regenerate all 34 reports + the index
netsentry pcap --demo                  # raw packets → CIC flows → verdicts
```

Skeptical pokes that should behave exactly as documented:

- `netsentry seeds` twice → identical reports (determinism is asserted, not hoped).
- Flip one byte of `models/serving_bundle.joblib` → `netsentry verify` exits 1.
- `NETSENTRY_GATE__MIN_PR_AUC_LIFT=100 netsentry gate` → exit 1, report says which
  bar failed and by how much.

_Every number in this tour comes from the schema-faithful synthetic stand-in (the
real CIC-IDS2017 requires registration); the [README](../README.md) states this
prominently, and the commands are identical on the real data._
