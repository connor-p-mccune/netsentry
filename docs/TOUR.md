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
- And the training-time fix, re-measured too: `netsentry sanitize` audits and drops
  the poisoned labels and recovers detection 2.2% → 18.4% at a 50% flip, *through
  the threshold channel* the poisoning study identified — with the clean-data tax
  kept as a measured row ([`reports/poisoning_defense.md`](reports/poisoning_defense.md)).
  Measure → fix → re-measure now closed for both adversaries.

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
([`reports/incident_demo.md`](reports/incident_demo.md)). The output side reaches
the operator's tools too: `netsentry watch` turns a spool of rotated flow files
into **ECS** JSON-lines alerts a SIEM ingests directly (exactly-once via a
size/mtime state file), and `POST /admin/reload` swaps the served model in place
**only if the candidate reproduces its own behavioral canaries in the live
runtime** — the deploy-time twin of the load-time canary, integration-tested for
the swap, the 409 rejection, and models-dir path safety.

## Stop 7 — The queue is simulated, because a fraction lies about time

The alert-queue study says budget K catches this fraction of attacks; `netsentry
socsim` asks the question a fraction cannot answer — *which* attacks get reviewed
before the shift ends. A seeded, event-driven **M/G/c queue with abandonment**
(`netsentry/evaluation/socsim.py`, core hand-checked in `tests/unit/test_socsim.py`)
works the model's real alerts under FIFO vs score-priority, and the payoff is a
number capacity-planning hides: risk-ordering the queue is worth up to **18 points
of attack-SLA** once the offered load crosses 1 and the backlog forms — with the
backlog column kept beside it to show the tail-starvation trade, not hide it
([`reports/socsim.md`](reports/socsim.md)).

## Stop 8 — The claims themselves are audited, and four of them failed

The newest wave went after the sentences the rest of the repo takes on trust, and kept the
results that came back badly.

- *"The threshold is chosen at a 0.1% FP budget"* describes a procedure, not a promise: its
  true rate exceeds that budget **51%** of the time. [`neyman_pearson.md`](reports/neyman_pearson.md)
  replaces it with a finite-sample guarantee — and finds the sample-size floor below which no
  threshold can certify the budget at all.
- *"We can tell when the model is out of its depth."* Not reliably.
  [`uncertainty.md`](reports/uncertainty.md) deletes an attack class from training: the detector
  scores it at chance, and its epistemic uncertainty is also at chance. It is blind and does not
  know it.
- *"Worst-case training helps."* [`dro.md`](reports/dro.md) gives the adversary eight rounds; it
  selects the round where it did nothing, and every round in which it acted made the worst group
  worse.
- *"More sophisticated tail estimation is better."* [`evt.md`](reports/evt.md) shows it wins by an
  order of magnitude on unbounded tails and provably nothing on bounded ones — which is the regime
  this detector is in.

Alongside them: [`verify_trees.md`](reports/verify_trees.md) proves per-flow robustness radii by
interval arithmetic rather than sampling (and refuses to report unless the flattened trees
reproduce LightGBM exactly), [`ope.md`](reports/ope.md) shows 77% of the "what would a lower
threshold have caught?" counterfactual is unanswerable from a deterministic policy's logs, and
[`survival.md`](reports/survival.md) puts the never-detected campaigns back into the latency
metric, moving it by 8x.

## Stop 9 — Two things stop being measured and start being proved

Most robustness and interpretability results in this field are measurements: *this* model, on
*this* data, scored *that*. Retrain and the number moves. The newest wave replaces two of them
with properties that hold by construction, and validates both against something that could
have contradicted them.

- **An entire evasion family is made impossible, not merely hard.**
  [`monotonic.md`](reports/monotonic.md) constrains the model non-decreasing in all 39
  attacker-inflatable features, so padding a flow can never lower its attack score. Not usually
  — never. The property is confirmed three independent ways: an interval-arithmetic proof over
  an *unbounded* inflation box (100% of alerts provably robust, against 0% for the deployed
  model), a greedy padding search that destroys 44.4% of the deployed model's alerts and none
  of the constrained one's, and a random probe that finds 375 score-lowering additions against
  the deployed model and zero against the constrained one. It costs −0.001 PR-AUC and **gains**
  3.6% detection, because "more bytes is never less suspicious" is true of network traffic and
  the unconstrained model had only three capture days in which to learn it.
- **"The best small tree we found" becomes the best small tree that exists.**
  [`optimal_tree.md`](reports/optimal_tree.md) runs branch and bound with two sound prunes and
  reports, per setting, whether the search space was *exhausted*. Greedy CART is provably
  suboptimal at all five penalties, by up to **69%**, and the optimal tree reaches three times
  greedy's held-out detection with half the leaves. The search itself is checked against
  exhaustive enumeration of every tree of every shape on fifteen small problems.

The same wave also turns a piece of framing into a measurement.
[`earliness.md`](reports/earliness.md) points out that flow exporters emit one record per
*finished* flow, so the deployed detector is structurally a post-mortem one — then finds that
an in-flight model using half the features **beats** it, 0.574 against 0.529 PR-AUC, on a
frontier where waiting never pays at any horizon. And three results in the wave came back
negative and were kept: deferral to a human loses against its own control (with the reason
given as a ratio), both causal-invariance methods reject genuine structure because 42% of
features point in opposite directions on different days, and the sketch study finds its own
high-precision configuration costs more memory than exact counting.

## Stop 10 — The ranking metric is not the deliverable

The newest wave is about what happens *after* training, and three of its five studies produced a
headline that looked like a win next to a deployment consequence that was not. Reading only the
first number would have shipped all three.

- **A from-scratch streaming learner beats the deployed model and still cannot be deployed.**
  [`online.md`](reports/online.md) implements a Hoeffding tree (VFDT) and ADWIN from scratch and
  runs them prequentially — test then train — against the frozen model and periodic retraining.
  The tree wins on PR-AUC (0.581 against 0.529) using 0.11 MB of sufficient statistics instead of
  28 MB of retained history. Then the operating column: **2.7% detection at the 0.1%
  false-positive budget against the frozen model's 10.3%**, because thirty leaves emit thirty
  distinct scores and a threshold can only sit between two of them. A SOC deploys a threshold,
  not an average precision.
- **Folding in a new attack family costs the old ones, and the cheap way of doing it costs more
  than it saves.** [`continual.md`](reports/continual.md) measures the full retention matrix
  across four update policies. Warm-start fine-tuning loses **61%** of the first family's
  detection three days later; full retraining still loses some, which is interference rather than
  forgetting; and the compute argument for incremental training does not survive measurement —
  a third of the rows for 19% less time, and a 4x larger ensemble that costs 6.3x more per
  thousand flows at inference.
- **Closing the loop on alert volume hands the attacker a lever.**
  [`control.md`](reports/control.md) turns the threshold into a PI control loop that holds the
  queue at the analyst budget the open-loop threshold misses by 100%. Then it floods the loop
  with ten batches of cheap decoys: the operating point moves from 2.01% of flows to 0.143% and
  detection of the genuine attacks arriving behind them falls from 6.0% to 1.6%. **The attacker
  buys 4.4 points of invisibility by generating alerts.** The static threshold is immune because
  it is not listening — adaptivity is the attack surface, and the mitigation is measured too.

The same wave adds the joint drift test the deployed per-feature monitors are blind to *by
construction* ([`mmd.md`](reports/mmd.md) — the KS statistics under a dependence-only fault come
back bit-identical to the unfaulted run), and checks the reason this project uses boosted trees
at all against an FT-Transformer under one shared protocol
([`deep_tabular.md`](reports/deep_tabular.md)).

## Stop 11 — The second metric was the report

The newest wave is about the parts of a detection system that are not the model, and it kept
finding the same shape: the metric a study leads with decides what it can see, and the
interesting number was always the second one.

- **A private data release is indistinguishable from real data on PR-AUC and detects a twentieth
  as much where it matters.** [`dp_synth.md`](reports/dp_synth.md) publishes a
  differentially-private synthetic capture and trains on it. Every privacy budget lands within
  0.129 PR-AUC of every other, against a 0.121 range across repeated draws of the *same*
  configuration — noise. Detection at the 0.1% false-positive budget, with the threshold chosen
  the way a recipient must choose it, moves 0.1% → 0.3% → 4.1% → 8.0% as epsilon goes 0.5 → 16.
  Noise destroys the tails of each marginal long before it disturbs the ordering, and an
  operating point lives entirely in the tail.
- **The sampling design that catches the most attacks is the one that cannot tell you what it
  missed.** [`sampling.md`](reports/sampling.md) scores 1% of the stream four different ways.
  Greedy top-k wins detection (3.9% against 2.0%) and admits **no unbiased estimator of the
  total at any budget**, because a flow below its cut has inclusion probability exactly zero and
  nothing observed can speak for it. Its lead is not even permanent — by a 25% budget the
  randomised design overtakes it, since greedy spends everything inside the region its
  pre-filter already believes.
- **A guarantee that holds in expectation is violated by half the deployments that hold it.**
  [`risk_control.md`](reports/risk_control.md) bounds the *miss rate* rather than the
  false-positive rate. Conformal risk control keeps its theorem and is exceeded on **39–46%** of
  200 simulated calibrate-and-deploy cycles; Learn-then-Test buys `P(miss > alpha) <= delta` and
  measures 4–12% against its 10% promise. Asking for a miss-rate *and* an alert-volume clause
  together returns an empty valid set — a certificate of infeasibility, produced before the
  contract is signed.
- **Federating hides the flows and not the incident.** [`secagg.md`](reports/secagg.md) shows a
  coordinator naming which attack family each site is holding **81% of the time** from the
  update alone, implements Bonawitz et al.'s masking protocol from scratch to remove that
  channel, and then measures what it costs: every Byzantine defence in this repository is a
  function of the individual updates the protocol exists to hide.

Two of the wave's studies are about the search rather than the model, and both are built around
their own controls. [`slice_discovery.md`](reports/slice_discovery.md) hunts ~19,000 feature
regions for the failures nobody predicted — and reports first that the identical search on
*permuted* losses finds 2,249 significant regions and zero after correction, then that the
weakest surviving slices lose half their effect on held-out rows while the strongest lose 5%.
[`pareto.md`](reports/pareto.md) evolves a Pareto front with NSGA-II, makes it beat random
search on exact hypervolume before believing it, and ends on a proof rather than a measurement:
**5 of the 12 front members are optimal under no weighting of the objectives whatsoever**, so
every scalar tuning procedure in this repository is structurally unable to return them.

And one is pure systems work: [`batching.md`](reports/batching.md) measures **10.03 ms of fixed
cost per scoring call against 0.0149 ms per flow**, moves the capacity ceiling from 101 to
63,479 requests a second by batching what the queue already holds, and replaces its own queueing
model after the first one missed by 25x — a batching server is self-regulating, because its
service capacity grows with its own backlog.

## Stop 12 — The thing that was assumed, not the thing that was measured

The newest wave takes four assumptions this project had been making silently and measures each
one. Three of them turned out to be false.

- **Sharing indicators as hashes is not private, and salting does not fix it.**
  [`psi.md`](reports/psi.md) implements Diffie-Hellman private set intersection over RFC 3526
  group 14 so one organisation can ask another about an indicator without naming it — 40 of 40
  shared indicators recovered exactly, the responder learning nothing. Then it runs the complete
  dictionary attack against the practice it replaces: an IPv4 address is a 32-bit preimage, so
  the **whole space falls in 7.9 hours** on one laptop core, and the 2^16 port space is exhausted
  in 0.41 s with every preimage recovered. Salting is the standard answer and it fails
  structurally — a sharing group must use the *same* salt or no two hashes would match, so the
  salt is group knowledge. Then the protocol itself gets attacked: it assumes truthful inputs, so
  a party submitting 1,600 guesses instead of its 40 real indicators gets **100% of the reachable
  overlap with no signal to the peer**.
- **The expensive features were never earning their cost.**
  [`acquisition.md`](reports/acquisition.md) prices six behavioural feature families the way an
  exporter pays for them and finds **four features beating all seventy-six** — 17.3% detection at
  a cost of 6.0 against 8.4% at 24.5. The adaptive policy that motivated the study, escalating
  only the flows whose verdict is in doubt, **loses to a random-gating control spending the same
  budget** — twice, once with a symmetric uncertainty band and again after the gate was rebuilt
  asymmetrically. The diagnostic says why in one table: the cheap tier forwarding 30% of flows
  retains 27.2% of the detections, where forwarding at random retains 30%. There was no signal
  for either policy to use, and the report keeps the negative rather than tuning until it wins.
- **The operating point does not need the stream stored.**
  [`quantiles.md`](reports/quantiles.md) builds four streaming quantile estimators from scratch
  and grades them not in quantile error but in **alert volume, the unit a SOC lead actually
  notices**. Nine of ten approximations deliver an *identical* alert volume to sorting all
  200,000 scores, because a threshold anywhere inside the gap between two adjacent benign scores
  is the same decision. P-squared holds the 0.1% operating point in **160 bytes**. The t-digest,
  the most sophisticated structure in the table, is beaten on both memory and update cost by a
  fixed-bin histogram, for a stated reason: a model score is bounded by construction, so the
  cheap option's assumption is free. All four then fail the same way under real
  validation-to-test drift, because none of them forgets.
- **A conformance claim is worth exactly as much as the check behind it.**
  [`compliance.md`](reports/compliance.md) maps the repository onto NIST AI RMF 1.0 and the EU AI
  Act's high-risk articles across 26 controls — and **verifies every control's evidence against
  the tree at generation time, so a renamed or deleted study downgrades its own control to
  unmet**. That mechanism, not the 93%/73% coverage, is the deliverable; the load-bearing unit
  test deletes an artifact and asserts the downgrade. Article 17 and Article 43 are recorded as
  permanently unmeetable by code, because a repository cannot perform a conformity assessment on
  itself.

## Stop 13 — The checks that had to be able to fail

The newest wave points four studies at things this project had been taking on trust. What they
have in common is a problem worth naming: **a check that is working prints exactly what a broken
one prints.** A linter on a clean codebase reports zero. A conformance machine against a correct
service reports no violations. So three of the four carry deliberately broken inputs, and in
every case those found a defect in the checker before they found anything about the subject.

- **The leakage rules, enforced by a parser.** [`mlint.md`](reports/mlint.md) turns six prose
  invariants into AST rules and grades them by **injecting twelve violations into real source
  plus ten pieces of correct code that resemble them — 12 caught, 0 false alarms**. The same
  rules over a textbook CIC-IDS2017 pipeline trip 11 violations across all six in twenty-six
  lines. The rules shipped with the bug they exist to catch (`val` inside `values` reads
  `values.mean()` as a validation-split leak), and **five hits led to real code changes**. Three
  violations stand: the feature store's as-of join keys, the one place in the model path where an
  identifier legitimately enters — left visible, with the CI budget set at exactly three. It
  then failed the build over a **leak in a study written the same week**: the bandit below was
  standardising its context by the stream's own mean, which hands an online learner a statistic of
  flows it has not seen.
- **A detector that never sees the training data beats four that do.**
  [`density.md`](reports/density.md) asks whether the anomaly score is a density estimate or a
  size measure. The squared norm of the standardised feature vector — no fitting, no parameters —
  detects **6.0%**, beating Isolation Forest, Mahalanobis, a KDE and PCA reconstruction, and the
  autoencoder's entire margin over it is 0.4 points. Regress the size proxy out and the lift over
  chance almost vanishes: **the best arm keeps 13%, the autoencoder 3%, and two arms rank worse
  than a coin.**
- **The serving contract, driven as a state machine.**
  [`state_machine.md`](reports/state_machine.md) states five properties an observer can check — a
  refused reload changes nothing, only a successful reload moves the version, health never claims
  `ok` while its canary fails — and drives the real application through 200 random operations.
  Clean, and **all five injected regressions caught**. The first version's weighted draw had
  produced a run with *zero* successful reloads: the most important transition unexercised while
  the report looked complete.
- **A bandit that hits the theory and still loses.** [`bandit.md`](reports/bandit.md) learns the
  triage policy online under partial feedback. LinUCB's regret grows as **`T^0.41`** against the
  `sqrt(T)` the analysis promises, and it **never overtakes a threshold chosen once on
  validation** — which itself lands within $1,075 of the best threshold obtainable with hindsight.
  What exploration spends is not detection but **the alert budget**: 5.35% of benign traffic
  against the deployed 0.88%, catching more attacks and losing money doing it. A reward function
  is not a constraint, which is why every operating point in this repository is a rate.

## Stop 15 — The premises nothing had checked

Four of the seven studies in the newest wave audit an assumption the project (or the field) had
been running on without measuring it, and three of them found the assumption false.

- **Interpretability is not what costs accuracy — capacity is.** A from-scratch additive model
  (GAM) makes capacity a *dial* rather than an architecture: sweeping bins per shape function
  2 → 64 takes training PR-AUC 0.474 → 0.859 while the later days rise, turn and fall. Validation,
  carved from the training days, catches the turn but stops one rung early and **overstates the
  achievable score by 0.231**. And the most readable model in the comparison — logistic
  regression — wins the honest split outright ([`reports/gam.md`](reports/gam.md)).
- **Hyperparameter search rests on two premises and both fail here.** No cheap rung of the
  fidelity ladder ranks configurations like the full run (−0.07 to +0.26, changing sign) while the
  cheap rungs correlate **+0.71 with the learning rate**; and validation ranking predicts the
  later days at **+0.23, p = 0.277**. Consequence, measured: all four searches finish *below* the
  configuration nobody searched for ([`reports/multifidelity.md`](reports/multifidelity.md)).
- **The mimicry attack this repo already shipped aims at the worst target available.** Optimal
  transport gives evasion a distance with units and a plan — and shows that only a *coupling*
  can be distributionally invisible. Centroid mimicry ends up **further** from benign traffic
  than the undisguised attack was, at a worst-feature PSI of 5.63 the deployed drift monitor
  catches without being told the attack exists ([`reports/transport.md`](reports/transport.md)).
- **`top_features` answers a question nobody wrote down.** TreeExplainer's default is one of
  three estimands; graded against a brute-force coalition sum (5×10⁻⁹) and separated from the
  other two by a duplicate-feature experiment whose answers are provable in advance
  ([`reports/shap_estimand.md`](reports/shap_estimand.md)).

A seventh is an attack that is devastating and infeasible at once: a **universal perturbation**
fitted on 400 flows takes detection on 800 unseen ones from 21.9% to **1.4%**, needs no queries
at attack time, and transfers from a *different model family* — and then the recipe says it works
by asking the attacker to *send less*. Restricted to additions, which is what padding actually
does, it takes 2.7 points; against a monotone-constrained model it takes **exactly zero**, which
is a property of the hypothesis class rather than an empirical bound
([`reports/universal.md`](reports/universal.md)).

The last two extend the trust boundary rather than auditing it: **proof-carrying verdicts**
(the model committed as a Merkle tree, seven forgeries refused, 392 KB and 95.5% node leakage
priced — [`reports/attestation.md`](reports/attestation.md)) and **private inference** (38 KB and
one round, plus the malicious-client attack that reads the model out in 1,217 queries —
[`reports/private_inference.md`](reports/private_inference.md)).

## Stop 15b — The instrument, not the model

Six studies that barely touch the classifier. Each one audits a piece of the **measuring
apparatus** instead — and that is where the defects turned out to be.

- **The API leaks its verdict in the length of the reply.** Every adversary defence here is
  query-side and assumes the attacker learns the answer by being told it. The verdict is
  recoverable from the response size at **AUC 1.000** on every endpoint configuration, free and
  passive. Five of the six rungs on the fix ladder do nothing, because the leak is the response's
  *shape*; padding is the only one that closes it. There is no timing channel for an instructive
  reason: SHAP runs unconditionally and hides the conditional work underneath it
  ([`reports/side_channel.md`](reports/side_channel.md)).
- **The seed is not enough, and what it fails to pin does not change the model.** Row order,
  disk round-trip and batch size are all irrelevant; the **thread count** changes the bytes and
  nothing else, because `n_jobs: -1` is a lookup resolved from the host. A bundle rebuilt
  elsewhere fails `netsentry verify` while being the same detector — so `provenance` now records
  a behavioural digest beside the file hash
  ([`reports/determinism.md`](reports/determinism.md)).
- **Every deployed operating point is dominated, and almost none of the gain is real.** All four
  budgets sit below the ROC convex hull; three of the four gains evaporate on the later days.
  The one that survives is worth +1.23 points and **the project declines it**, because a per-flow
  coin changes 0.67% of verdicts between runs and two other components exist to catch exactly
  that ([`reports/hull.md`](reports/hull.md)).
- **A holdout is burned by being asked to choose, not by being read.** The sealed split is read
  103 times from 98 modules. Selecting among *indistinguishable* candidates costs +0.0093 PR-AUC
  and leaves you with a detector worse than the one you replaced; selecting among *genuinely
  different* ones costs nothing and finds six points. Thresholdout fails twice — it cannot debias
  an argmax, and a temporal split is not exchangeable ([`reports/reuse.md`](reports/reuse.md)).
- **The README quoted twelve numbers no report contains.** Now a CI gate, with numeric rather
  than textual matching (a quote of `2.31` asserts the report *rounds* to 2.31) because the first
  version cried wolf at roundings. Its injection harness corrected the gate, then corrected
  itself: it had reported 100% detection by asking whether a token still verified *after being
  replaced* ([`reports/claims.md`](reports/claims.md)).
- **The false-positive budget is decided by nine flows.** At the tightest budget nine benign flows
  clear the threshold, so the realised rate is known to ±90% of itself, and detection there needs
  **1.0 points** to be resolvable. Pairing a comparison narrows its interval **2.6×** — and
  several of this project's own published differences sit at or below the bar
  ([`reports/power.md`](reports/power.md)).

And one that runs them all at once. A **2⁴ factorial** finds the coverage promise already broken
before any stressor is applied with no monitor watching; evasion costing 31% of detection while
every monitor stays silent; a prevalence change tripping an alarm with nothing wrong; and the
false-positive budget never breached because every failure *lowers* the scores. No compound
break — kept as a negative — but **75% of the monitor interactions are negative**: responses
saturate rather than stack, so concurrent failures are less visible than separate ones
([`reports/composition.md`](reports/composition.md)).

## Stop 16 — Where the bodies are buried, on purpose

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
netsentry analyze                      # regenerate every report + the index
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
