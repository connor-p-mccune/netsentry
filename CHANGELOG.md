# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once released.

## [Unreleased]

### Added
- Anchor explanations: high-precision IF-THEN rules with a guarantee (`netsentry anchors`,
  `netsentry/explain/anchors.py`): the explainability suite answers many questions but not the one
  a SOC analyst asks out loud — "give me a *rule* I can trust." SHAP attributes a verdict across
  features, the counterfactual finds the smallest clearing change, exemplars point at similar
  cases, but none states a **sufficient condition**. An anchor (Ribeiro, Singh & Guestrin, AAAI
  2018, from the authors of LIME) does: a short conjunction of feature predicates such that,
  whenever they hold, the model returns this verdict with high **precision** (>= tau), and, among
  such rules, high **coverage**. Each candidate feature is discretised into quantile bins and a
  greedy search pins the flagged flow to its own bins — adding at each step the predicate that most
  raises precision, estimated on a background of real flows satisfying the rule (so the rule
  respects the feature correlations a synthetic perturbation would break), with a
  lower-confidence-bound stopping rule in place of the paper's KL-LUCB bandit. Every reported
  anchor's precision is re-measured on a **held-out background** it was not grown against, so the
  guarantee is validated, not just fit. On the stand-in it produces crisp, actionable rules — e.g.
  `Flow Packets/s >= 299 AND Flow Bytes/s >= 3.15e3 AND Total Backward Packets >= 25.6 -> attack`
  at 97% precision (95% LCB, 99% held-out) for DDoS, and a 99% rule for DoS Hulk — while flows the
  model flagged that are actually benign honestly get lower-precision rules. Runs on the
  exchangeable stratified/binary split at the model's natural decision boundary. The greedy search
  (perfect-separator recovery, no-signal ceiling, support refusal) and the precision lower bound
  are unit-tested; e2e slow test; in the analysis suite. `anchors.*` config.
- The H-measure, a coherent alternative to ROC-AUC (`netsentry hmeasure`,
  `netsentry/evaluation/hmeasure.py`): the suite reports ROC-AUC with the imbalance caveat, but
  Hand (2009) identified a subtler flaw — averaging over thresholds, AUC implicitly weights
  false-positive against false-negative cost by a distribution that **depends on the classifier's
  own score distribution**, so two models are compared under two different, incomparable cost
  assumptions and an AUC win can encode a cost stance no one would hold. The H-measure removes the
  incoherence by fixing an **explicit, shared** Beta prior on the misclassification-cost parameter
  for every classifier and reporting the normalised expected minimum loss (0 = the best trivial
  classifier, 1 = perfect separation), built from the ROC convex hull and integrated against the
  prior. It is reported next to ROC-AUC and Gini, and under a second **cost-skewed** prior that
  encodes the SOC's real stance (a missed attack costs more than a false alarm) — a knob AUC
  structurally cannot expose. On the stand-in's honest temporal split it lands an honest finding:
  logistic regression (AUC 0.711, H 0.213) edges the deployed gradient-boosted model (AUC 0.668,
  H 0.180), with AUC and the H-measure agreeing on the ranking, and the random control at H 0.000.
  The value is comparison hygiene for the leaderboard and the promotion gate. Metric anchors
  (perfect = 1, trivial = 0, range, monotone-invariance) unit-tested; e2e slow test; in the
  analysis suite. `hmeasure.*` config.
- Anytime-valid drift detection via a conformal test martingale (`netsentry exchangeability`,
  `netsentry/monitoring/exchangeability.py`): the drift suite already has PSI, per-feature KS with
  Benjamini-Hochberg FDR, and online Page-Hinkley / DDM — but every one of them needs a reference
  window or spends its false-alarm budget at a declared moment. A monitor that runs forever needs a
  stronger contract: it may alarm at **any** stopping time and still control the false-alarm
  probability over the whole unbounded run. A conformal test martingale (Vovk, Nouretdinov &
  Gammerman, ICML 2003) provides it: each flow yields an online conformal p-value — the smoothed
  rank of its nonconformity (the deployed attack score) among all flows seen so far — which is
  Uniform(0, 1) exactly under the null that the stream is **exchangeable**. Those p-values feed a
  parameter-free mixture-of-power-martingales betting process that stays a fair game under the null
  and grows without bound under drift, so by **Ville's inequality** alarming at `M_t >= 1/alpha` has
  false-alarm probability at most `alpha` at any stopping time — no window, no multiple-testing
  correction, no fixed horizon. On the stand-in the exchangeable (shuffled) stream's martingale
  peaks at only 1.7 (no alarm), while a stream that turns attack-heavy at flow 1,000 is detected
  ~140 flows later (median), and across 50 independent exchangeable streams **0%** ever crossed the
  `1/alpha = 100` line — at or under the 1% budget Ville promises. Complements rather than replaces
  the feature-wise PSI/KS reports: those localise *which* feature moved, this answers *whether and
  when* the stream stopped being the distribution the model was validated on. The martingale/Ville
  property, growth under drift, and the conformal p-values' uniformity are unit-tested; e2e slow
  test; in the analysis suite. `exchangeability.*` config.
- Prediction-powered inference for attack prevalence (`netsentry ppi`,
  `netsentry/evaluation/ppi.py`): the whole evaluation suite assumes a fully-labelled test
  set; a SOC never has one. It scores every flow and labels a tiny audit sample, and still
  owes a defensible answer to "what fraction of today's traffic is malicious?" with an honest
  interval. Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic,
  *Science* 2023) is the estimator that gets it right: start from the model's average over all
  the unlabelled flows, then subtract the model's **measured bias on the labelled audit** — the
  rectifier `mean(f - y)` — so the estimate is unbiased *whether or not the model is
  calibrated*, and, because a useful model's residual is lower-variance than the raw label, the
  interval is tighter than the label-only classical one at the same coverage. The study sweeps
  the audit budget and, at each, measures every interval's half-width and its **empirical
  coverage** of the true test prevalence over hundreds of random audit draws — validity shown,
  not asserted. On the stand-in (true prevalence 0.221), PPI runs **~23% narrower** than
  classical at 1,000 labels (worth ~1.8x the labels) while both hold ~90% coverage, and the
  naive "let the model label everything" baseline is priced as the cautionary column: its point
  is the model's own biased mean (+0.053), and its interval is far too narrow (it never looks at
  a label), so it **misses** the truth — tight but invalid. Runs on the exchangeable
  stratified/binary split, because PPI's guarantee needs the audit to be a random sample of the
  scored population, exactly the exchangeability the temporal split is built to break. The
  estimator algebra (unbiasedness, the constant-score fallback to classical, the perfect-model
  width collapse, the label-savings ratio) is unit-tested; e2e slow test; in the analysis suite.
  `ppi.*` config.

## [0.9.0] — 2026-07-14

The adversarial-completeness & attribution wave: the model-stealing attack that
completes the classic adversarial-ML quadrilogy (evasion + poisoning + privacy +
**extraction**), the two attribution studies that go under the model (training-data
valuation, feature interactions), and the *certified* robustness guarantee that is to
the evasion study what differential privacy is to the membership audit.

### Added
- Model-extraction (model-stealing) attack (`netsentry extraction`,
  `netsentry/robustness/extraction.py`): the fourth classic attack on an ML model after
  evasion (inference-time), poisoning (training-time), and membership inference
  (privacy) — the one about the **confidentiality of the model itself** (Tramer et al.
  2016; Papernot et al. 2017). A surrogate is trained purely on the victim's returned
  scores over the attacker's own same-distribution traffic (no ground-truth labels), and
  its **fidelity** (agreement with the victim) and stolen detection (PR-AUC) are swept
  over the query budget. On the stand-in, ~4,000 free queries reach **95.5% fidelity**
  and **98%** of the victim's PR-AUC. The classic Tramer defense — return less (rounded
  probabilities, then top-1 label only) — is measured and lands the literature's finding:
  it barely dents fidelity, because a hard label still reveals which side of the boundary
  each query lands on. The security payoff is priced directly: an evasion search run
  offline against the *stolen* surrogate transfers to the victim, recovering **95%** of a
  fully white-box attack's effect (victim detection 43% -> 17%) without a single evasion
  query to the victim, and clearly beating a random-perturbation control — extraction as
  the enabler behind black-box transfer evasion. Runs on the stratified/binary split.
  Query-response defenses, surrogate fit, fidelity, and the transfer search are
  unit-tested; e2e slow test; in the analysis suite. `extraction.*` config.
- Training-data valuation via exact KNN-Shapley (`netsentry datavalue`,
  `netsentry/evaluation/data_value.py`): every other study values the *model*; this
  values the **data**. The KNN-Shapley value (Jia et al., VLDB 2019) is the exact,
  game-theoretic contribution of each training flow to a nearest-neighbour classifier's
  accuracy on held-out traffic, computed in O(N log N) per query via the closed-form
  recursion (checked in the tests against a brute-force exact-Shapley enumeration). The
  value is signed, and the sign is the point: a negative flow sits among the opposite
  class and hurts. Two uses follow — a **self-validated mislabel detector** (planted
  label flips concentrate in the negative-value tail: flip-detector AUC **0.83** on the
  stand-in, reaching the confident-learning label audit's finding from an independent
  geometric first principle) and a **value-guided pruning** knob whose transfer to the
  deployed tree model is measured, not assumed (reported honestly as weak on a
  near-duplicate-heavy stand-in). A per-class value table is included with the
  KNN-Shapley-under-imbalance caveat stated plainly. Runs on the stratified/binary split.
  Recursion, recovery reads, and edge cases unit-tested; e2e slow test; in the analysis
  suite. `data_value.*` config; `plots.plot_hist_overlay` added.
- Feature interactions via Friedman's H-statistic (`netsentry interactions`,
  `netsentry/explain/interactions.py`): the one interpretability view the suite was
  missing — how features *combine*. The partial-dependence report warns that a PDP
  assumes feature independence and hides interaction; this measures it. Friedman's H
  (Friedman & Popescu, 2008) is the share of a feature pair's joint-partial-dependence
  variance that is *not* explained by summing the marginals — 0 (additive) to 1 (fully
  entangled) — estimated on the honest temporal model through the fitted pipeline, so it
  reads against the PDP. On the stand-in the strongest interaction is **Flow Duration x
  Flow IAT Mean (H = 0.41)**, a physically sensible coupling (duration ~ packets x
  inter-arrival time). The H math (additive -> 0, multiplicative -> 1) is unit-tested on
  constructed and end-to-end synthetic responses; e2e slow test; in the analysis suite.
  `interactions.*` config; `plots.plot_heatmap` added.
- Certified robustness via randomized smoothing (`netsentry certify`,
  `netsentry/robustness/certify.py`): the formal-guarantee counterpart to the empirical
  evasion study — the same role differential privacy plays for the membership audit. The
  smoothed classifier (majority vote under Gaussian noise) comes with a **provable** L2
  radius `R = sigma * Phi^-1(p_A)` (Cohen, Rosenfeld & Kolter, 2019), where `p_A` is a
  Clopper-Pearson lower bound on the majority-vote probability over Monte-Carlo draws;
  inside that radius no perturbation can change the verdict, found or not. The
  certified-accuracy-vs-radius curve is swept across noise levels, exposing the
  accuracy/robustness frontier (sigma 0.25 -> 1.0: clean accuracy 70% -> 68% for a median
  certified radius 0.50 -> 0.77 on the stand-in). Reported with both conservatisms named:
  the certificate is against *any* L2 perturbation (the evasion attacker only moves the
  controllable subset), and an undefended tree certifies conservatively (the measure ->
  noise-augmented-training fix is named, as with hardening and DP). Radii share the
  evasion study's standardised-feature units. The certification math (Clopper-Pearson,
  Cohen's radius, the accounting) is unit-tested; e2e slow test; in the analysis suite.
  `certify.*` config.

## [0.8.0] — 2026-07-13

The privacy & explainable-anomaly wave: the membership audit's named next step
finally taken — differentially-private training with a from-scratch privacy
accountant, priced on a utility–leakage frontier — and the "detect the unknown"
component made explainable, both offline (a per-feature attribution study with a
faithfulness check) and live (an opt-in API field that says *why* a flow was
flagged).

### Added
- Differential-privacy training + frontier (`netsentry dp`,
  `netsentry/robustness/dp.py`): the mitigation the [membership audit](docs/reports/membership.md)
  names but does not exercise. Two reusable primitives — a **pure-stdlib
  (math-only, no scipy) Rényi-DP accountant** for the subsampled Gaussian mechanism
  (Abadi et al. 2016; Mironov 2017) at integer orders (a sound upper bound on
  epsilon; log-space composition; the sharpened Canonne–Kamath–Steinke RDP→DP
  conversion), and a **DP-SGD logistic classifier** (per-example gradient clipping +
  Gaussian noise; the spent epsilon is a function of the noise multiplier, sampling
  rate, and step count only, so it is certified for any dataset). The study trains a
  non-private reference and DP models across a noise sweep on the exchangeable
  stratified/binary split and prices each on one axis: epsilon spent, detection kept
  (PR-AUC + TPR@FPR), and membership leak closed (the same Yeom attack, reusing the
  membership module). Stand-in finding, reported as it fell: detection is remarkably
  robust to the guarantee (PR-AUC 0.690 → 0.683 down to a strong epsilon ≈ 1.7, then
  0.666 at epsilon ≈ 0.8), while the empirical Yeom leak barely moves because a
  regularised **linear** model memorises little to begin with — so the report leads
  with DP's real value: the *formal* (epsilon, delta) certificate holds against every
  attacker, not just the one measured. Accountant closed-forms + monotonicities and
  DP mechanics unit-tested; e2e slow test; in the analysis suite. `dp.*` config.
- Anomaly-flag attribution (`netsentry anomexplain`,
  `netsentry/explain/anomaly_explain.py`): the unsupervised mirror of SHAP. The
  supervised model returns its top features on every prediction; the anomaly
  detector emitted only a score. This names the behaviours behind a flag by
  **model-agnostic benign occlusion** (reset each feature to its benign reference,
  re-score, read the drop), so it explains whichever detector ships (the autoencoder
  locally, Isolation Forest in torch-less CI). Because occlusion can be a just-so
  story, the report **validates** it the XAI way — a deletion/faithfulness check that
  occluding the top-attributed features must move the score far more than random
  ones. Stand-in: strongly faithful (top-5 occlusion drops the score 13.4× more than
  random-5) and cleanly interpretable — DDoS flags driven by Flow Packets/s + Flow
  Bytes/s (volumetric), PortScan by SYN Flag Count (the scan signature). Pure
  attribution + faithfulness math unit-tested against a known-rule stub; e2e slow
  test; in the analysis suite. `anomaly_explain.*` config.
- Live anomaly explanations in serving (`?anomaly_explain=true` on `/predict` +
  `/predict/batch`): turns the attribution study into an API contract. The serving
  bundle now embeds a per-feature benign median (the occlusion reference), and the
  engine returns `anomaly_features` — the top benign-occlusion contributions behind a
  flag, with analyst-readable feature names — for flagged flows only. Mirrors the
  `?exemplars=true` posture: opt-in (no latency tax on the standard path),
  evidence-only (verdict fields byte-identical), and best-effort (a bundle without
  the reference, or any failure, returns null, never an error). Integration-tested.

## [0.7.0] — 2026-07-13

The adversarial-privacy & host-graph wave: NetSentry completes the classic
adversarial-ML attack triad by adding the third axis — privacy — alongside the
existing evasion and poisoning studies; adds the cross-flow *topology* analytic the
identity-blind per-flow model is structurally blind to (scan fan-out + lateral
movement, the topology mirror of beaconing's timing); and makes the project's founding
thesis executable with a leakage-attribution study that reproduces the field's inflated
~99% and prices each leakage source.

### Added
- Membership-inference privacy audit (`netsentry privacy`,
  `netsentry/robustness/membership.py`): the third classic attack on an ML model after
  evasion (inference-time) and poisoning (training-time) — the one about privacy. With
  only query access, can an attacker tell whether a flow was in the training set (Shokri
  et al. 2017; Yeom et al. 2018)? Runs on the exchangeable stratified split (the
  assumption MI needs, the same reason active learning runs there) with a Yeom
  confidence-threshold attack and a Shokri shadow-model attack (eight shadows teach an
  attack classifier), plus a deliberately-overfit reference of the same architecture that
  keeps the project's measure → re-measure arc. Worst-case leakage is reported as TPR at a
  low false-accusation budget (Carlini et al. 2022), not attack accuracy. On the
  stand-in the deployed model leaks above chance (threshold AUC 0.68, shadow 0.70) but
  the low-FPR worst case is thin (~2% of members), while the overfit reference's
  advantage nearly doubles (0.27 → 0.54) even though its accuracy gap barely moves —
  leakage is driven by memorisation, so the deployed model's regularisation and early
  stopping are its privacy control; the report names DP training as the next study. Pure
  attack math unit-tested; e2e slow test; in the analysis suite. `membership.*` config.
- Host-graph analytics (`netsentry graph`, `netsentry/intel/graph.py`): the cross-flow,
  topology-aware complement to the per-flow classifier — the topology mirror of how
  `netsentry beacon` is its timing mirror. Reconstructs the host communication graph
  from the `Src IP`/`Dst IP`/`Dst Port` metadata (the fields the model never sees) and
  surfaces two attacks no single flow can show: scan fan-out (ATT&CK Discovery, T1046 —
  horizontal by distinct hosts, vertical by distinct ports; the signal the temporal
  model misses on PortScan) and lateral-movement chains (ATT&CK Lateral Movement, T1021
  — a reached host pivoting deeper, recovered whole via a depth-bounded DFS along
  internal→internal hops so ordinary egress can't form a chain). `--demo` plants a
  horizontal sweep, a vertical sweep, and a four-hop pivot among benign egress talkers
  and recovers all three (`docs/reports/graph_demo.md`). Reads identity as metadata only;
  a hunt-lead generator, not a verdict, like beaconing. Internal/external is strict
  RFC1918 (not `ip_address.is_private`, which matches documentation ranges). Fan-out,
  chain recovery, order invariance, and the demo unit-tested. `graph.*` config.
- Leakage-attribution study (`netsentry leakage`, `netsentry/evaluation/leakage.py`): the
  executable form of the project's thesis. Starting from the honest temporal model, three
  leakage sources are added back one at a time — a shuffled split, `Destination Port`, and
  a synthetic per-(day, class) session identifier standing in for Flow ID / Source IP —
  and each rung's raw-score PR-AUC gain is priced. On the stand-in: honest 0.529 →
  shuffled 0.783 (+0.254) → +port 0.958 (+0.176) → +identifier 1.000 (+0.042), reproducing
  and decomposing the field's ~99%. The identifier leak only works on the shuffled split
  (later-day campaigns carry ids the model never saw), so it is a consequence of the split
  leak, not an independent term. The injected identifier is a controlled demonstration of
  the anti-pattern the `remainder="drop"` firewall stops, never adopted by the pipeline;
  ties back to `netsentry gate`'s > 0.999 leakage ceiling. Injected-id stability, dense
  coercion, and ladder deltas unit-tested; e2e slow test; in the analysis suite.
  `leakage.*` config.

## [0.6.0] — 2026-07-12

The explainability-depth and parser-hardening wave: the response-curve shape of the
model's top features (partial dependence + ICE, the one interpretability view the
suite was missing), and a Hypothesis fuzz harness that pins the never-crash contract
of the untrusted-input capture parser.

### Added
- Capture-parser fuzz harness (`tests/unit/test_capture_fuzz.py`): a Hypothesis
  fuzzer asserting the never-crash / count-and-skip contract of the untrusted-input
  pcap/pcapng reader — a classic memory-safety and DoS surface. Three regimes
  (arbitrary bytes, a valid container magic followed by garbage, and byte-level
  mutations of a real capture) drive the reader, which must only ever return a
  `(records, stats)` pair or raise the one typed `PcapReadError` — never an uncaught
  `struct.error`, an unbounded allocation, or a hang. The parser passes as-is (it was
  already defensively written); the harness makes that a regression guard. Slow-marked
  (file I/O + binary parsing), so it runs in `make test` / CI, not the fast dev loop.
- Partial dependence + ICE (`netsentry pdp`,
  `netsentry/explain/partial_dependence.py`): the response-curve shape the rest of
  the explainability suite doesn't show. Partial dependence (Friedman) for the top
  model features with individual conditional expectation (ICE) curves layered under
  each, computed in raw feature space — a feature is swept across its own data
  quantiles while the others stay put, and every perturbed frame is scored through
  the fitted pipeline + model (the API's transform), so the axis is interpretable
  and there is no train/serve skew. On the stand-in the steepest curves (Total Fwd
  Packets, flow rates, Flow Duration) are the attacker-controllable features the
  evasion/recourse studies exploit. The report states the PDP independence caveat
  plainly and points at the ICE spread as the signal of correlation-driven
  extrapolation, framing it as a marginal-response diagnostic, not a causal claim
  (that is the ablation's job). Adds `plots.plot_pdp_grid` (small-multiples PDP+ICE
  panels, shared y-axis) and a `partial_dependence.*` config block; in the analysis
  suite. Grid trimming, the sweep math, ICE heterogeneity, and direction/effect
  unit-tested; an end-to-end slow test writes the report and figure.

## [0.5.0] — 2026-07-12

The SOC-native integrations wave: NetSentry stops speaking only its own dialect and
starts speaking the languages a detection/intel team already deploys — the signature
baseline exported as Sigma rules, detections as STIX 2.1 threat-intel bundles — adds
the cross-flow C2 detection the per-flow model is structurally blind to (beaconing),
and ships production Kubernetes manifests (Helm + Kustomize) for the inference API.

### Added
- Kubernetes deployment (`deploy/`): a production Helm chart
  (`deploy/helm/netsentry`) and equivalent raw Kustomize manifests (`deploy/k8s`)
  for the inference API, both rendering the same hardened deployment — a
  health-gated rollout (liveness/readiness on `/health` + a `startupProbe` for the
  first-boot bundle bootstrap), autoscaling (CPU-target HPA + PodDisruptionBudget),
  a Prometheus Operator ServiceMonitor scraping `/metrics`, and a hardened runtime
  (non-root uid 1000, readOnlyRootFilesystem, all capabilities dropped,
  RuntimeDefault seccomp, no mounted service-account token). Optional `X-API-Key`
  auth is injected from a Kubernetes Secret, never baked into a manifest. The model
  volume defaults to an emptyDir (the image's synthetic-bundle bootstrap) and takes
  a PVC for a real bundle. `make helm-lint` / `helm-template` / `k8s-render` /
  `k8s-apply` targets and a `deploy/README.md` guide.
- Beaconing / C2 periodicity detection (`netsentry beacon`,
  `netsentry/intel/beacon.py`): the cross-flow, identity-aware complement to the
  per-flow classifier, which drops every identifier and so is structurally blind to
  a host calling home on a fixed cadence (ATT&CK Command and Control, T1071). Groups
  connections by talker pair (`Src IP` -> `Dst IP`, optionally per destination port)
  and scores each pair's inter-arrival-time regularity with a robust dispersion (MAD
  over the median interval), 0.0 (bursty/human) to 1.0 (perfectly periodic), skipping
  pairs below `beacon.min_events`. `--demo` runs a deterministic synthetic capture
  that plants one 60 s beacon among jittery benign talkers; the detector ranks it
  first (regularity 0.975, CV 0.04) above every benign pair (<=0.44), committed to
  `docs/reports/beacon_demo.md`. Reads the timestamp/identity columns as metadata
  only — the fields the model never sees — and the report states the scope plainly:
  a hunt-lead generator, not a verdict (legitimate periodic services also flag), and
  it adds no detection to the per-flow verdicts. `beacon.*` config. Regularity math,
  ranking, min-events skipping, timestamp-order recovery, and the demo unit-tested.
- STIX 2.1 threat-intel bundle export (`netsentry stix`,
  `netsentry/intel/stix.py`): scored-flow incidents (reusing the incident
  grouping) folded into a standards-conformant STIX 2.1 bundle a TAXII server or
  intel platform (MISP, OpenCTI) ingests directly. Emits an identity SDO, one
  attack-pattern per observed ATT&CK technique (`external_references` into
  `mitre-attack`, shared with the `mitre` field), an indicator per incident with a
  real STIX pattern over the attacking hosts (`ipv4-addr:value`) or targeted
  service (`network-traffic:dst_port`), observed-data + the SCOs it references
  (`ipv4-addr`, `network-traffic`) when capture identity is present, a sighting
  (count/first/last-seen) and an `indicator indicates attack-pattern`
  relationship, all under a TLP marking-definition (default AMBER). Object ids are
  deterministic UUIDv5s over stable content, so re-export is byte-identical
  (idempotent TAXII push); the bundle id is content-addressed over its objects.
  `stix.*` config for the identity name and TLP level. Bundle structure,
  pattern selection, SCO emission, dedup, TLP marking, and determinism
  unit-tested; an end-to-end slow test scores a real bundle.
- Sigma detection-rule export (`netsentry sigma`, `netsentry/intel/sigma.py`):
  the hand-written signature baseline (`rules.definitions`, the incumbent
  `netsentry rules` benchmarks the model against) emitted as portable
  [Sigma](https://sigmahq.io) rules a detection-engineering team compiles to any
  SIEM backend via pySigma. Each rule carries the Sigma comparison modifiers
  (`|gte`/`|lte`), an indicative ATT&CK tag shared with the `mitre` prediction
  field via the one mapping (so they cannot drift), and a deterministic UUIDv5
  `id` (byte-stable regeneration, no version-control churn). Field names stay
  CICFlowMeter/NetSentry flow-feature names with the field-mapping caveat written
  into the generated `README.md`. Colliding clauses on one field split into
  separate `selection_*` groups; NaN never matches, mirroring the `RuleEngine`
  semantics. The committed pack lives in `docs/reports/sigma/`; wired into the
  analysis suite. Modifier mapping, ATT&CK tags, deterministic ids, YAML
  validity, and the export unit-tested.

## [0.4.0] — 2026-07-11

The defense-and-operations wave: the training-time adversary's defense
re-measured (audit-and-drop sanitization, completing the measure→fix→re-measure
arc), the cross-schema operating-point advice finally priced (threshold
transfer), a discrete-event SOC queue simulation that shows why triage order
matters once the queue saturates, and two production surfaces — canary-gated hot
model reload and an ECS spool watcher that streams SIEM-ready alerts from a
directory of dropped flow files.

### Added
- SOC queue simulation (`netsentry socsim`, `netsentry/evaluation/socsim.py`): a
  non-preemptive M/G/c queue with abandonment at the shift boundary, seeded and
  event-driven, that lays the deployed model's real alerts onto a shift (benign
  false positives uniform, attacks clustered into campaigns) and works them under
  FIFO vs score-priority. The headline is attack-SLA attainment — the share of
  true-attack alerts an analyst starts within the SLA window — which decomposes
  the alert-queue study's "detected" into "detected AND triaged in time." On the
  stand-in, score-priority is worth up to 18 points of attack-SLA, appearing once
  the offered load crosses 1 and the backlog forms — the queueing knee a static
  fraction cannot express. The event-driven core is a pure, deterministic function
  hand-checked against known shifts; the arrival timeline is documented as a model
  (CIC-IDS2017 has no per-flow clock); in the analysis suite.
- Canary-gated hot model reload (`POST /admin/reload`, `netsentry/serving/app.py`):
  config-gated (`serving.reload_enabled`, off by default, API-key guarded) swap of
  the live bundle without a restart. The candidate is loaded into a fresh engine
  that replays its own embedded behavioral canaries in this runtime; the swap
  happens only if they reproduce within tolerance (mismatch → 409, keeping the old
  model; escaping path → 400; missing bundle → 404). The engine lives behind a
  mutable holder so the swap is a single atomic reassignment and in-flight requests
  finish on the model they started with; every attempt increments
  `netsentry_model_reloads_total{outcome}`. The deploy-time analogue of the
  load-time canary — `verify` attests the bytes, this attests the behaviour at the
  moment of the swap. Integration-tested (swap, canary gate, path safety).
- Spool watcher (`netsentry watch`, `netsentry/serving/watch.py`): scores each new
  flow file dropped into a directory (Zeek rotation, a CICFlowMeter cron, or
  `pcap --flows-out`) through the same engine the API serves and appends the attack
  verdicts as Elastic Common Schema (ECS) JSON lines — `event.*` envelope,
  `rule.name`, `threat.*` for the MITRE mapping, and `source`/`destination`/
  `network` enriched from capture-identity columns that ride along (never model
  features). A JSON state file keyed on each file's size and mtime makes processing
  exactly-once across restarts; a malformed file is skipped, never fatal; `--once`
  drains the backlog and exits. ECS mapping and state logic are pure and
  unit-tested; an end-to-end slow test drives a real bundle over a spooled file.
- Poisoning defense (`netsentry sanitize`, `netsentry/robustness/sanitize.py`):
  the third step of the measure→fix→re-measure arc for the training-time
  adversary, mirroring `netsentry harden` for evasion. Flips are planted across
  the operator's whole labeled pool (train + validation, since threshold
  selection is poisoned too); the confident-learning audit (shared knob with
  `netsentry labelaudit`) flags suspects in both directions; every flag is
  dropped (an operator cannot know which way labels rot); the model is refit and
  the decay curve is re-measured undefended vs sanitized on the clean temporal
  test split. On the stand-in, detection at the operating point recovers 2.2% →
  18.4% at a 50% flip rate despite catching only ~45% of flips — the healing
  runs through the poisoned-threshold channel, not perfect cleaning. The
  zero-poison point is kept as the defense's measured tax, and rows are dropped
  rather than relabeled so the auditing model's own errors are not bootstrapped
  back in. Limits stated: random flips only; contamination untouched; the tax
  recurs every retrain. Flip/drop/outcome arithmetic unit-tested; e2e slow test;
  in the analysis suite.
- Threshold transfer (`netsentry transfer`, `netsentry/evaluation/transfer.py`):
  prices the cross-dataset study's closing instruction, "re-choose thresholds on
  labeled local traffic." Four policies meet the foreign NetFlow-schema set at
  the primary FPR budget — the transplanted source threshold (231× over budget
  on the stand-in), an unsupervised quantile on the *unlabeled* target scores
  (measured at both the as-is and a production-like mix, exposing that it
  under-alerts when traffic is hostile), a threshold bought with k local labels
  (redrawn 30× so small-sample quantile noise is spread not hidden), and the
  all-label oracle. Budget compliance counts both sides, since an over-strict
  cut silently spends detection. Quantile/compliance/trial arithmetic
  unit-tested; e2e slow test. `plot_lines` gains a `yscale` passthrough for the
  log-log realized-FPR figure.
- Zeek conn.log ingestion (`netsentry zeek`, `netsentry/integrations/zeek.py`):
  score the logs a network team already collects. Parses classic TSV logs
  (`#separator`/`#fields`/`#unset_field` respected) and JSON-lines output,
  maps connection totals onto the CIC columns they can honestly speak for
  (duration, per-direction packets/bytes, derived rates/means, history-derived
  flag counts as documented lower bounds), and leaves the intra-flow detail
  conn.log cannot express missing for the fitted pipeline to impute — the
  cross-dataset study's regime, with its expectation stated (ranking transfers;
  re-choose thresholds on labeled local traffic before trusting a budget).
  Zero-duration rates map to NaN per the cleaning policy; the Zeek UID rides as
  pivot metadata; the scored output feeds `netsentry incident` unchanged.
  Parsing + mapping unit-tested against a hand-built log; e2e slow test.

## [0.3.0] — 2026-07-10

The adaptive-operations wave: the oldest result in the IDS literature computed
against the deployed operating points (the base-rate fallacy), the two
self-repairing layers a deployment runs under drift (adaptive conformal for the
coverage guarantee; threshold refresh for the operating point — with the honest
finding that the cheap lever buys almost nothing on this stream), case-based
explanations audited and then shipped in the API, native pcapng ingestion
closing the capture stack's stated limitation, and analyst-ready incident
reports as the last mile from per-flow verdicts to a response artifact.

### Added
- Incident reports (`netsentry incident`, `netsentry/intel/incident.py`): scored
  flows folded into the artifact an analyst actually reads. Consecutive
  same-class alerts (small benign gaps bridged, `incident.gap_tolerance`) become
  incidents rendered with flow count/span, peak and mean calibrated probability,
  the ATT&CK tactic/technique link, services via `Destination Port` routing
  metadata, source/target talkers when capture metadata is present, the conformal
  action mix, and the most-cited SHAP feature. The committed demo artifact
  (`docs/reports/incident_demo.md`) runs the synthetic capture end-to-end into
  two PortScan incidents and a DoS Hulk incident with T1046/T1499 links. The
  grouping is a stated contiguity heuristic (the campaigns study's correlation
  assumption) and creates no detection — every number is a re-reading of the
  same engine verdicts the API serves. Grouping pure + unit-tested; end-to-end
  slow test on a fresh bundle.
- Native pcapng ingestion (`netsentry/capture/pcap.py`): the capture stack's
  "convert with tshark first" limitation, closed. A pure-stdlib pcapng reader
  parses Section Header / Interface Description / Enhanced Packet / Simple
  Packet blocks in either byte order, converts per-interface `if_tsresol`
  resolutions (10^-v and 2^-v encodings) to the microsecond timeline the flow
  assembler expects, and supports concatenated sections with section-scoped
  interface numbering. The skip-don't-die posture extends to the block level:
  unknown block types skip by declared length, packets on unsupported-linktype
  interfaces are counted per interface, SPBs parse with a "no timestamps" note.
  `read_capture()` sniffs the container by magic; `netsentry pcap` accepts
  either format transparently. Tests build every block field-by-field with
  struct, checking the parser against known on-wire values.
- Exemplar (case-based) explanations (`netsentry exemplars`,
  `netsentry/explain/exemplars.py` + `similar_flows` in the API): the nearest
  known training flows per prediction — label, capture day, and distance in the
  fitted pipeline's standardized space — audited before being served. The audit:
  exemplar-supported alerts are 89% precise vs 82% unsupported on the stand-in
  (bucket sizes 1,428 vs 44 reported alongside, so the gap reads as
  triage-ordering evidence, not a calibrated re-ranker), and NN distance does
  not separate caught from missed attacks (the novelty study's geometry,
  restated per flow — said plainly). The index is a class-balanced, seeded
  float32 subsample (rare classes represented, not drowned), exact brute-force
  k-NN. Serving embeds it in the bundle and `?exemplars=true` opts in on both
  prediction endpoints — evidence-only (decision fields untouched), best-effort
  (can never break a build or a prediction), and integration-tested for both.
  In the analysis suite.
- Threshold-refresh study (`netsentry refresh`, `netsentry/monitoring/refresh.py`):
  the label-cheap adaptation lever priced against full retraining. Static /
  refresh / retrain / retrain+refresh policies ride the prequential later-day
  stream at one FPR budget, decomposing drift's cost into operating-point drift
  (fixable by re-estimating one quantile on a trailing labeled window) and
  ranking drift (fixable only by retraining). Refreshed cuts are chosen on the
  prequentially *emitted* scores, so no model picks its threshold on flows it
  trained on. The stand-in verdict is a kept double negative: ~1% of the
  retraining recovery, and no budget-compliance win on a stream whose benign
  scores barely move — the compliance prose is sign-aware and the lever's value
  case (a material score shift blowing the frozen cut's budget) is constructed
  and asserted in unit tests. In the analysis suite.
- Adaptive conformal inference (`netsentry adaptiveconformal`,
  `netsentry/evaluation/adaptive_conformal.py`): the conformal report's broken
  temporal guarantee, repaired online. The Gibbs-Candes update treats alpha as a
  control variable steered per class by realized coverage errors — a long-run
  coverage guarantee that needs no distributional assumption, priced honestly:
  on the stand-in stream attack coverage recovers 64.4% → 89.7% (90% target)
  while the human-review share rises 35% → 69%, because ACI widens the sets
  exactly where the model is blind rather than improving the detector. The
  quantile lookup reproduces the static module's finite-sample arithmetic (the
  adaptive run starts from the static thresholds), alpha is deliberately
  unclamped (the wide-open excursion is what makes the guarantee
  assumption-free), and a `label_delay` knob models triage lag. Updater and
  coverage repair unit-tested on constructed streams; in the analysis suite.
- Base-rate stress test (`netsentry baserate`, `netsentry/evaluation/baserate.py`):
  Axelsson's base-rate fallacy (1999) measured against the deployed operating
  points rather than cited. Thresholds are chosen on validation at each FPR
  budget, conditional TPR/FPR are measured on the honest temporal test split, and
  Bayes' rule sweeps the production attack prevalence across orders of magnitude:
  per-prior queue composition (alerts/day, false share, precision, attacks
  caught), the break-even prevalence below which most alerts are false (0.64% at
  the tight budget on the stand-in), and the inverted question — the FPR a
  90%-precision queue would need at a 1e-5 base rate (~5,800x tighter than the
  measured point). The report ties the fallacy to the layers that already answer
  it (score ranking, campaign aggregation, explicit costs). Bayes arithmetic pure
  + unit-tested; in the analysis suite.

## [0.2.0] — 2026-07-09

Everything from packets to policy, shipped since `v0.1.0`: raw-capture ingestion,
the honest-protocol studies (model leaderboard, self-training, campaign-level
detection) alongside the earlier post-release wave, a property-based invariant
layer that immediately caught a drift-monitor blind spot, a measured serving fast
path, and the model-lifecycle machinery (noise floor → release gate → promotion →
behavioral canaries → shadow challenger → retrain policy).

### Added
- Packet-capture ingestion (`netsentry pcap`, `netsentry/capture/`): raw PCAP →
  CIC flows → verdicts, with no capture-library dependency. A pure-stdlib
  classic-libpcap reader (both byte orders, µs/ns timestamps, Ethernet/VLAN/raw-IP
  link layers; malformed and non-IP frames counted and skipped, never fatal) feeds
  a bidirectional flow assembler that reimplements the CICFlowMeter aggregation
  over the canonical schema module — the output is exactly the 78 training
  columns, and scoring runs through the same `InferenceEngine` as the API, so
  there is no re-implemented preprocessing to skew. Departures from CICFlowMeter
  are deliberate and documented (bulk features 0, zero-duration rates NaN →
  imputed by the fitted pipeline, flows end on idle timeout or TCP close).
  `--demo` writes a deterministic synthetic capture (benign web/DNS sessions, a
  SYN sweep, a flood) from struct-packed frames — no binary fixture in the repo —
  which doubles as the parser's ground-truth test harness and the CI smoke.

- Model-family leaderboard (`netsentry leaderboard`,
  `netsentry/evaluation/leaderboard.py`): majority prior, Gaussian naive Bayes,
  logistic regression, random forest, and the deployed LightGBM through the
  identical honest harness on both splits — same persisted splits, same
  leakage-safe pipeline, per-model validation-chosen thresholds, raw-score PR-AUC.
  Stand-in findings: the split gap replicates across every family and exceeds the
  whole between-family spread on the honest split, and the two splits crown
  different winners (simple models lead temporally, flexible ones optimistically;
  the gap grows with capacity) — model selection on the shuffled split ships the
  wrong model. The render states the ranking inversion only when the data shows
  it. Family builder + evaluator unit-tested; in the analysis suite.
- Self-training study (`netsentry selftrain`, `netsentry/training/selftrain.py`):
  the pseudo-label shortcut priced against the labeled ceiling. The temporal test
  stream splits in time order into an unlabeled adaptation window and an untouched
  evaluation window; static vs self-trained (confident raw scores folded in under
  their pseudo-labels) vs oracle retrain (true labels) meet the future at their own
  validation-chosen thresholds, and the study audits the pseudo-labels against the
  truth the models were blinded to. Stand-in: −0.003 of a +0.190 labeled headroom
  recovered; 12.9% of the window's attacks confidently absorbed as benign while
  per-side pseudo-label precision reads ~92% — the confirmation-bias loop measured,
  not asserted. Selection/audit helpers pure + unit-tested; in the analysis suite.

- Verdict-only fast path on the prediction endpoints (`?explain=false`): SHAP is
  the measured majority of request latency (p50 48 → 13 ms, ~22 → ~75 req/s on
  the stand-in), so throughput-bound callers can skip it per request —
  `top_features` returns empty, every decision field is byte-identical, and the
  response model is unchanged. Explanations remain the default (they are the
  contract). `netsentry benchmark --no-explain` drives and reproduces the
  comparison; integration-tested that only the explanation is skipped.
- Campaign-level detection (`netsentry campaigns`,
  `netsentry/evaluation/campaigns.py`): the (day, attack-class) operation as the
  SOC's unit of account. Per campaign and FPR budget: flow-level rate, alerted at
  k=1 / confirmed at k, and first-alert latency in the campaign's own hostile
  flows (stream order; benign interleavings do not advance the counter — tested).
  Stand-in: 21% flow-level at the 1% budget is 5/5 campaigns alerted, but
  first-alert latency spans flow 1 (DDoS) to flow 687 (PortScan) — "detected"
  vs "detected in time". The report states what the framing does not buy: alert
  volume is still priced per flow, alert correlation is assumed (k_confirm is
  the conservative column), and small campaigns get few draws.
- Property-based invariant suite (`tests/unit/test_properties.py`, hypothesis):
  the contracts the results stand on, asserted for arbitrary inputs — the FPR
  budget is never exceeded on the selecting set, detection is monotone in the
  budget, confusion rates stay coherent, `attack_probability` is a probability
  in every shape, PSI is a nonnegative divergence (zero on identity, major on
  total migration), and cleaning's guarantees survive generated adversarial
  frames. `hypothesis` joins the dev extras.

### Fixed
- PSI is no longer blind on degenerate reference features: a constant or
  two-valued reference (always-zero bulk columns, near-binary flag counts) had
  its histogram collapsed to one open bin, so a total migration off the
  reference support read PSI = 0 — silently exempting those features from the
  serving gauges, the Prometheus drift alert, and the drift-triggered retrain
  trigger. Degenerate references now keep each observed value as its own bin.
  Found by the property suite's total-migration invariant; regression-tested in
  both directions. Committed drift reports are unaffected (their degenerate
  features are identical in reference and current, which reads 0 either way).
- Explanations return analyst-readable feature names: the API `top_features` and
  batch `top_feature` no longer leak the fitted ColumnTransformer's `numeric__`
  branch prefix. One shared `display_feature_name` helper now serves the
  explainer, the distilled rules, and the evasion tables (surfaced by the capture
  path, whose scored output an analyst reads directly).

### Added
- Surrogate distillation (`netsentry distill`, `netsentry/explain/distill.py`): the
  inverse of the rules baseline — how much of the *learned* model survives
  translation into an auditable form? A depth-limited decision tree imitates the
  teacher's attack ranking (raw score, so the teacher row matches the headline
  evaluation; the monotone calibrator applies identically on top), swept across
  depths and judged on fidelity (Spearman + decision agreement at matched alert
  volume) and on its own detection. Stand-in: 49 rules reproduce 97.5% of
  volume-matched decisions but only 0.61 of the fine ranking (PR-AUC 0.529 → 0.451)
  — coarse behavior compresses, the ranking does not. Quantization (K leaves = K
  scores) and the behavior-not-mechanism scope are stated in the report; the chosen
  tree is rendered in full. Fidelity math pure + unit-tested; in the analysis suite.
- Shadow-challenger scoring in serving (`serving.shadow_artifact_path`): a second
  bundle scores every request through the identical path and never touches the
  response — the champion answers, the shadow is measured. Prometheus gains the
  paired evidence a promotion wants from live traffic: a |champion − shadow|
  calibrated-probability delta histogram and a decision-disagreement counter, each
  model judged at its own threshold for the active profile. `/health` reports the
  shadow's version. Failure isolation is explicit: a shadow that fails to load is
  skipped, one that fails mid-request disables itself. Integration-tested with an
  identical-copy shadow (delta histogram fills; disagreements provably zero).
- Behavioral canaries (`netsentry canary`, `netsentry/serving/canary.py`): every
  persisted bundle (serving *and* training — the artifact promotion snapshots)
  embeds a class-mixed handful of raw validation flows with the exact calibrated
  scores it produced at build; the serving runtime replays them at load and must
  reproduce them. `verify` checks the artifact's bytes, the canary checks its
  behavior — env skew (library/BLAS changes) moves scores without moving a byte.
  Surfaced on `/health` (status flips to "degraded"), exit-coded as a deploy gate
  (1 = behavioral drift, 2 = no canary present — a missing check is not a passing
  check), and `serving.canary_strict` refuses to serve on mismatch. NaN feature
  values round-trip as None. Unit-tested against a stub bundle; end-to-end in the
  serving integration suite.
- Retrain-trigger policy study (`netsentry retrainpolicy`,
  `netsentry/monitoring/retrain_policy.py`): the streaming study shows retraining
  recovers what drift costs; this prices *when*. Never / periodic / drift-triggered
  (the deployed model's own score-PSI vs the same `psi_major` line the Prometheus
  alert fires on, with cooldown) / every-batch policies ride the prequential stream,
  each with its own reference that resets on redeploy. The stand-in result is kept
  as a finding: the PSI trigger fires early, goes quiet after the redeploy, and
  captures ~none of the headroom (0.413 vs 0.534 mean batch PR-AUC) while the
  calendar cadence lands at 0.474 — a score distribution can settle while labeled
  quality is still being bought, so an unsupervised trigger is a cost-saver, not a
  substitute for labels. Trigger logic pure + unit-tested; in the analysis suite.
- Champion/challenger promotion (`netsentry promote`,
  `netsentry/models/promotion.py` + `confidence.paired_diff`): the decision layer
  between training and serving. Challenger and champion are scored on the same
  frozen temporal test rows; deltas are paired-bootstrap (one resample scores both
  models, cancelling shared sampling noise), detection is compared at each bundle's
  own validation-chosen threshold, and the non-inferiority margins sit just above
  the seed audit's measured noise floor. Two policies (`non_inferiority` for routine
  retrains under drift; `superiority` for risky swaps); on promotion the challenger
  is snapshotted to a SHA-256-pinned champion and every decision appends to a JSONL
  history; non-zero exit on HOLD for pipeline branching. First real decision: HOLD —
  a seed-43 retrain was PR-AUC-equivalent (+0.0001) but credibly worse at the
  0.1%-FPR operating point (−1.5pp, CI excludes zero), the ranking-vs-operating-
  point thesis at the deployment layer.
- Release quality gate (`netsentry gate`, `netsentry/evaluation/gate.py`): the
  definition of done as an exit code. Structural honesty checks on the artifact
  that would ship — the leakage firewall re-verified on the fitted feature space,
  calibrator attached, every configured FPR profile present, an end-to-end scoring
  smoke — plus configurable floors (PR-AUC as a multiple of prevalence, TPR at the
  primary FP budget, ECE of the calibrated score) and one deliberate ceiling:
  PR-AUC > 0.999 fails as suspected leakage. The gate failed its own first run
  (ECE bar set from taste at 0.10 vs the measured temporal-shift 0.1057) and the
  bar was reset with the reasoning documented. In the analysis suite.
- Seed-sensitivity audit (`netsentry seeds`, `netsentry/evaluation/seed_variance.py`):
  the error bar bootstrap CIs cannot see. The honest temporal model is refit at
  consecutive seeds with thresholds re-chosen per run (the full deployment
  pipeline); reproducibility (same seed ⇒ bit-identical, asserted every run) is
  separated from stability (PR-AUC sd 0.0017, TPR@0.1%FPR sd 0.0063 on the
  stand-in). Data noise dominates training noise here (bootstrap half-width 0.0116)
  — stated, and used: the promotion margins are calibrated against this floor. In
  the analysis suite.
- Feature-importance stability audit (`netsentry importance`,
  `netsentry/explain/importance_stability.py`): the honesty check behind treating
  explainability as a contract. The model is refit on bootstrap resamples of the
  temporal training split, global feature importance is recomputed each time, and the
  ranking's movement is summarised by the mean pairwise Spearman rank correlation and
  the top-k Jaccard overlap. On the stand-in the full ranking is noisy (Spearman ~0.40)
  while the top-10 leaders are comparatively stable (Jaccard ~0.59) — the honest
  "trust the head, not the tail" read, and the reason the API returns only the top few
  features. Pure metric computation is unit-tested against known matrices; companion to
  the SHAP global summary (explains one model) and the ablation (family causal value).
  In the analysis suite.
- Adversarial hardening (`netsentry harden`, `netsentry/robustness/hardening.py`):
  adversarial training against the feature-space mimicry the evasion study measures.
  It augments the honest temporal/binary training set with mimicry-perturbed copies of
  the attack flows (the attacker's own move toward the benign centroid, still labeled
  attack), refits, and runs the **same** evasion study against baseline and hardened
  models — closing the loop the robustness report only pointed at. Calibration and FPR
  thresholds are fit on the clean validation split for both, so operating points
  compare like-for-like. On the stand-in, full-mimicry detection recovers 0% → ~100%
  at a small clean cost (temporal PR-AUC 0.529 → 0.519); the report leads with the
  trade-off and states that adversarial training only defends the perturbation it
  trains on. Augmentation mechanics unit-tested; end-to-end determinism checked. In
  the analysis suite.
- Statistical & online drift detectors (`netsentry driftscan`,
  `netsentry/monitoring/detectors.py`): the significance and timing PSI omits. A
  per-feature two-sample Kolmogorov-Smirnov test with a Benjamini-Hochberg FDR
  procedure across features (5/76 certified as genuinely shifted on the stand-in, vs
  PSI's magnitude ranking), plus two classic online detectors on the deployed model's
  streams — Page-Hinkley on the score stream and DDM (Gama et al., 2004) on the error
  stream — that report the change-point index. Against a planted reference→current
  boundary both alarm within the later-day segment. Detectors are pure and validated
  against planted shifts (the DDM zero-baseline warmup degeneracy is guarded). In the
  analysis suite.
- MITRE ATT&CK Navigator layer export (`netsentry navigator`,
  `netsentry/intel/navigator.py`): NetSentry's detection coverage written as a valid
  ATT&CK Navigator layer JSON, colored by support-weighted per-class recall at the
  operating FPR (stratified reference split, where every class is evaluable). A file a
  detection-engineering team loads directly into the ATT&CK Navigator to see its
  coverage in the framework its threat model is written in — floods green, stealthy
  classes as red gaps. Tactic shortnames live beside the existing ATT&CK mapping so the
  layer and the `mitre` prediction field cannot drift; layer builder + aggregation
  unit-tested incl. JSON validity. In the analysis suite.
- Alert-queue capacity planning (`netsentry alertqueue`,
  `netsentry/evaluation/alert_queue.py`): the detection a fixed analyst budget buys.
  A budget of K alerts/day maps to the ROC operating point whose alert volume equals K
  at a realistic 1% production base rate, so recall, queue precision, and the **lift
  over random triage** are read off the score ranking. On the stand-in the ranking is
  worth ~50-60× random triage (~12 analysts catch 2.5% of attacks at ~83% precision,
  8.2% at 2,500/day), with detection flattening as staffing climbs. Complements the
  cost report (which picks the economically optimal threshold). Pure simulator
  unit-tested; in the analysis suite.
- Per-service threshold profile in serving (`?profile=per_service`): the parity
  audit's finding shipped as a product feature. The bundle builder computes, on the
  same calibrated validation scores as every other profile, a decision threshold per
  service at the primary FPR target (`netsentry/serving/bundle.py`), and inference
  judges each flow at its service's threshold. The flow's `Destination Port` rides
  in the request as routing metadata — it is never a model feature — and selects the
  threshold via a shared port→service map (`netsentry/data/services.py`, extracted
  from the subgroups audit so both layers agree by construction). Absent ports and
  thin/one-class services fall back to the profile's global threshold, so the
  profile degrades to the global cut rather than misrouting.
- Label-noise audit (`netsentry labelaudit`, `netsentry/evaluation/label_audit.py`):
  confident-learning-style detection of likely label errors on the temporal training
  split — out-of-fold k-fold scores (no row judged by a model that trained on it)
  with class-conditional mean thresholds, so a benign-labeled row scoring like a
  typical attack is flagged. Validates itself by planting a known fraction of label
  flips: on the stand-in it recovers 58.8% of planted flips at 19.8% precision
  against a 1.2% base rate (a 16x triage concentration), and the intrinsic flags on
  the clean-by-construction labels are reported as the method's ambiguity floor, not
  as errors. Complements the poisoning study (which prices corruption; this finds
  it) and connects to the documented CIC-IDS2017 label corrections (Engelen et al.,
  WTMC 2021). In the analysis suite.
- Leave-one-day-out temporal sensitivity (`netsentry lodo`,
  `netsentry/evaluation/lodo.py`): every capture day takes a turn as the held-out
  "future" (train on the other four, validation carved from train, threshold chosen
  there), reusing the temporal-split machinery per fold. Because each CIC-IDS2017
  attack class lives on exactly one day, every fold is zero-shot class detection; and
  benign-only Monday becomes a pure quiet-day false-alarm audit (0.94% FPR ≈ 9.4k
  alerts/day at the assumed volume). On the stand-in, novel-family detection ranges
  1.5% (Web/Infiltration) to 25.4% (the DoS family, which generalises from DDoS and
  vice versa), mean 13% — the headline temporal conclusion holds under every
  rotation, with the spread reported as a per-family difficulty profile. In the
  analysis suite.
- Novelty-distance study (`netsentry novelty`, `netsentry/evaluation/novelty.py`):
  for every test attack, the Euclidean distance to its nearest **training** attack in
  the pipeline's standardized space, profiled for both split strategies on shared
  quantile bins with detection at the operating threshold. Decomposes the headline
  temporal-vs-stratified gap into a **composition** part (nearness/near-twins — the
  shuffled split's leakage proper) and an **at-distance** part (the later days are
  harder at matched novelty) by reweighting stratified per-bin detection to the
  temporal distance mix. On the synthetic stand-in the mixes nearly coincide (no
  burst near-twins by construction — stated, with the real-data expectation) and the
  gap is ~all at-distance (-1.1 vs +27.8 pts); detection *rises* with distance in
  both splits, honestly reported: extremes are easy, the near-benign-manifold attacks
  are the hard ones — the same geometry the evasion study exploits. In the analysis
  suite.
- Per-service detection-parity audit (`netsentry subgroups`,
  `netsentry/evaluation/subgroups.py`): the temporal test flows grouped by the
  service implied by `Destination Port` — a field the model never sees, used only to
  *slice*, never to predict — with per-service detection and false-positive rates at
  the single global operating threshold, each carrying a Wilson 95% interval so
  binomial noise is not mistaken for disparity. An equalized-odds-style audit in
  security clothing: on the synthetic stand-in the per-service FPR spread
  (0.57–1.11% around the 1% budget) straddles its intervals (reported as such), while
  the detection gap (HTTP 42% vs ephemeral-port traffic 0.3%, i.e. PortScan/
  Infiltration) is far outside them — showing one global cut guarantees only the
  aggregate budget, and which service queue fills with false positives first. In the
  analysis suite.
- Prequential streaming simulation (`netsentry streaming`,
  `netsentry/monitoring/streaming.py`): replay the later-day (temporal test) flows as
  a time-ordered stream and compare a static model (frozen at deploy) against one
  retrained on each labeled batch, scored prequentially (test-then-train) at one fixed
  operating threshold. Overlays per-batch model-score PSI so the batches where the
  static model decays line up with the drift signal — closing the loop from measuring
  drift (`netsentry drift`) to acting on it. On the synthetic stand-in retraining
  lifts mean batch PR-AUC 0.43 → 0.54 (reaching ~0.90 on late-stream batches). In the
  analysis suite.
- Feature-group ablation study (`netsentry ablation`,
  `netsentry/evaluation/ablation.py` + behavioural-family groupings in
  `features/feature_sets.py`): leave-one-family-out on the honest temporal split,
  refitting with each behavioural family (timing/IAT, flow rates, packet size, TCP
  flags, volume/counts, header/window) removed to measure its marginal detection
  value — the causal complement to SHAP's attribution. On the synthetic stand-in
  removing flow rates collapses PR-AUC (0.529 → 0.224) while removing volume/counts
  *improves* it — the fingerprint of overfitting to the temporal shift, reported as
  such (with an explicit warning against selecting features on the test split). In the
  analysis suite.
- Active-learning label-efficiency study (`netsentry activelearning`,
  `netsentry/evaluation/active_learning.py`): from a small labeled seed, compare
  uncertainty sampling (query flows nearest the decision boundary) against random
  labeling, refitting and scoring test after each round — the analyst-labeling-budget
  question the rest of the project's "analyst time is the constraint" framing implies.
  Runs on the stratified split (where the pool and test are exchangeable, the
  assumption active learning needs — the training-time mirror of conformal selective
  prediction). On the synthetic stand-in uncertainty sampling reaches random's
  full-budget PR-AUC with ~22% fewer labels. In the analysis suite.
- Provenance & supply chain (`netsentry provenance` / `netsentry verify`,
  `netsentry/governance/provenance.py`): a CycloneDX 1.5 SBOM of the project's
  declared dependencies resolved to installed versions (with Package URLs a CVE
  scanner keys on), and a model-integrity manifest — the bundle SHA-256, a digest of
  the resolved training config, the git commit, the runtime, and a summary of the
  bundle's contents. `netsentry verify` recomputes the hash and exits non-zero on a
  mismatch — the deploy/CI integrity gate against a swapped or corrupted artifact.
  The SBOM is hand-emitted to the schema (not via a churning library API) so it
  stays a stable, spec-valid artifact. In the analysis suite.
- Training-set poisoning study (`netsentry poisoning`,
  `netsentry/robustness/poisoning.py`): the training-time counterpart to the evasion
  study. Label-flip poisoning (attack rows relabeled benign) against the supervised
  model and benign-pool contamination (attack rows injected into the benign-only
  pool) against the anomaly detector, with degradation always measured on the clean
  test split while train/val carry the poison. The headline finding on the synthetic
  stand-in: PR-AUC (a ranking metric) is robust to label flips while detection at the
  operator's poisoned-validation threshold collapses (21% → 1.8% at a 50% flip) — the
  project's operating-point-vs-ranking thesis, in the security dimension. In the
  analysis suite.
- Rules-vs-model baseline (`netsentry rules`, `netsentry/models/rules.py` +
  `netsentry/evaluation/rules.py`): a config-driven signature ruleset (six
  Suricata-style, port-scoped threshold rules) benchmarked against the classifier
  on the same temporal test split at a **matched false-positive budget**, plus the
  hybrid (rules OR model) and a per-class breakdown. On the synthetic stand-in the
  signatures win the single operating point (the test mix is dominated by the two
  patterns they encode, and PortScan is novel to the Mon–Wed model) while having
  exactly 0% recall on every class without a rule — the complements-not-rivals
  case, stated with the numbers either way. In the analysis suite.
- Per-attack-class detection slices (`netsentry slices`,
  `netsentry/evaluation/slices.py`): detection rate per attack class on the temporal
  split, exposing *which* later-day (largely novel) attacks are caught. On the
  synthetic stand-in DDoS transfers (~53%, behaviourally like the trained DoS family)
  while PortScan/Bot/Web Attack/Infiltration are mostly missed — the concrete
  known-vs-novel breakdown the aggregate PR-AUC hides, and the case for the anomaly
  detector. In the analysis suite.
- Auto-generated model card (`netsentry modelcard`, `netsentry/evaluation/
  model_card.py`): a factual spec sheet derived straight from the deployed bundle
  (backend, classes, calibration, threshold profiles, attached components, ATT&CK
  coverage, provenance) so it can't drift from what ships — complementing the
  hand-written narrative card. Governance automation.
- Counterfactual recourse explanations (`netsentry recourse`,
  `netsentry/explain/counterfactual.py`): for a flagged flow, the minimal set of
  moves to attacker-controllable features that would clear it — the analyst's what-if
  that complements SHAP's why. The features it surfaces line up with the robustness
  study's most-exploitable ones (both are the controllable subspace). In the analysis
  suite; on the synthetic stand-in the top hits clear within 1-2 changes.
- Learning-curve / data-efficiency study (`netsentry learningcurve`,
  `netsentry/evaluation/learning_curve.py`): PR-AUC vs training size for both splits,
  with the bias/variance read. On the synthetic stand-in the temporal curve is flat
  (+0.003 from 2.8k→28k examples) — saturated, so more data of the same kind won't
  move the honest number — while the temporal-vs-stratified gap persists at every
  size, confirming it is a validation-protocol effect, not a sample-size one.
- Data-quality gates (`netsentry validate`, `netsentry/data/validation.py`): validate
  a dataset against the schema contract — required feature columns, label vocabulary,
  numeric dtypes (structural failures) plus missingness, duplicates, and degenerate
  class balance (warnings) — writing a report and exiting non-zero on failure so CI
  can gate on it. Wired into the CI smoke after `prep`.
- Offline batch scoring (`netsentry score`, `netsentry/serving/batch.py`): score a
  CSV/Parquet of flows to a predictions file with the same InferenceEngine the API
  uses (class, probability, decision, anomaly, recommended action, ATT&CK technique,
  top feature) — the model is usable without standing up the service.
- MITRE ATT&CK enrichment (`netsentry/intel`, `netsentry intel`): each attack class
  is mapped to an ATT&CK tactic + technique, returned in the `mitre` field of every
  attack prediction and summarised in a coverage report (12 classes → 6 tactics, 8
  techniques). One source of truth shared by serving and the report; mappings are
  documented as indicative of the CIC-IDS2017 scenarios.
- Bootstrap confidence intervals + significance tests (`netsentry/evaluation/
  confidence.py`): the evaluation report now gives PR-AUC and TPR@FPR percentile-
  bootstrap CIs, and the temporal-vs-stratified over-optimism gap comes with a CI and
  a bootstrap p-value — so the project's headline finding is backed by statistics,
  not a point estimate (on the synthetic stand-in: gap +0.257, 95% CI
  [+0.239, +0.276], p < 0.001).
- One-command analysis suite (`netsentry analyze`, `make analysis`,
  `netsentry/evaluation/analyze.py`): regenerates every model-analysis report
  (operational eval + calibration, cost, conformal, robustness, drift) and writes a
  linked `docs/reports/INDEX.md` with per-report status, each run defensively so one
  failure does not abort the rest.
- Observability stack (`docker/prometheus`, `docker/grafana`, compose `monitoring`
  profile): Prometheus scraping the API, a pre-provisioned Grafana dashboard
  (request/error/latency, scored-flows-by-decision, anomaly rate, feature-drift PSI
  gauges, attack-probability distribution), and alert rules for drift, attack-rate
  spikes, error rate, and a p99 latency SLO. New serving metrics expose model
  behaviour (`netsentry_predictions_total`, `netsentry_anomalies_total`,
  `netsentry_attack_probability`). One command: `make docker-monitor`.
- Hyperparameter optimization (`netsentry/training/tune.py`, `netsentry train tune`):
  an Optuna (TPE) search over the gradient-boosted classifier, leakage-safe by
  construction (pipeline fit on train, every trial scored by validation PR-AUC, test
  never touched), with a seeded random-search fallback when Optuna is absent. Wires
  the previously-unused `supervised.tune` / `tune_trials` config and writes the best
  params to a YAML override (`configs/tuned.yaml`) for a reproducible retrain.
- Conformal prediction & selective alerting (`netsentry/evaluation/conformal.py`,
  `netsentry conformal`): class-conditional split-conformal prediction sets with a
  finite-sample, distribution-free coverage guarantee, mapped to SOC actions
  (auto-clear / auto-alert / route-to-human for ambiguous or novel-empty sets). The
  report contrasts the exchangeable stratified split (guarantee met) with the
  temporal split (attack-class coverage falls short) — surfacing that the conformal
  shortfall is itself a distribution-shift signal, complementing the PSI monitor.
- Cost-sensitive threshold selection (`netsentry/evaluation/cost.py`, `netsentry
  cost`): a decision-theoretic operating point that minimises expected cost
  (analyst time per alert vs expected loss per missed attack), the closed-form
  Bayes threshold for a calibrated probability, a production-base-rate daily-cost
  extrapolation, and a comparison against the fixed-FPR profiles. Builds directly
  on the calibrated score; surfaces the val→test temporal drift in threshold choice.
- Adversarial-evasion robustness study (`netsentry/robustness`, `netsentry
  robustness`): two feature-space attacks against the deployed model — a mimicry
  attack (shape attacker-controllable volume/timing features toward benign) and an
  adaptive L2-bounded query search — with robustness curves, a most-exploitable-
  feature ranking, and a report. Converts the model card's "not adversarially
  robust" caveat from an assertion into a measured curve (full mimicry takes
  supervised detection from ~83% to ~0% on the synthetic stand-in), motivating the
  pairing with the benign-only anomaly detector.
- Probability calibration (`netsentry/models/calibration.py` +
  `netsentry/evaluation/calibration.py`): a monotonic isotonic/Platt calibrator
  fit on the validation split, applied to both the served `attack_probability` and
  the FPR decision thresholds, plus reliability-diagram / Brier / ECE / MCE
  diagnostics in the evaluation report. Closes the `ml.md` §4 requirement that
  threshold claims be backed by calibrated probabilities (the `thresholds.calibrate`
  config flag is now wired). Because the map is monotonic it preserves ranking, so
  PR-AUC/TPR@FPR are unaffected — only the meaning of the probability improves.
- Drift monitoring (`netsentry/monitoring`): a Population Stability Index (PSI)
  implementation, a `netsentry drift` report contrasting a current dataset with a
  reference (feature drift + model-score drift; default temporal test-vs-train),
  and an in-serving rolling-window monitor exporting `netsentry_feature_drift_psi_max`
  / `_mean` Prometheus gauges. The drift reference travels inside the serving bundle.
- Cross-dataset generalization study (`netsentry crosseval`): a synthetic
  NetFlow-schema foreign dataset, an adapter mapping it into the CIC feature space,
  and an honest in-domain-vs-cross report (PR-AUC + TPR@FPR + the gap, with
  sign-aware framing). Point the adapter at UNSW-NB15 / NF-*-v2 for real numbers.
- vulnpipe integration (`netsentry triage`): re-rank vulnerability findings by a
  fused risk score (severity/CVSS + model attack probability + anomaly flag), so a
  CVE on a host with attack-like traffic outranks the same severity on a quiet host.
- Streamlit demo dashboard (`netsentry demo`): pick/edit a flow and see the live
  verdict, anomaly score, and SHAP explanation; verified headless via Streamlit
  AppTest. Optional `demo` extra.
- ONNX export + quantized inference (`netsentry onnx`): export the classifier to
  ONNX, verify it matches sklearn (~1e-7), and benchmark ONNX Runtime (~1.4x the
  Python path) against dynamic quantization (a documented no-op for tree ensembles).
  Optional `onnx` extra.

### Added
- API security hardening (`netsentry/serving/app.py`): optional API-key auth
  (`X-API-Key`) and a per-client fixed-window rate limit on the prediction endpoints,
  both config-gated (`serving.api_key`, `serving.rate_limit_per_minute`) and enforced
  in middleware so `/health` and `/metrics` stay open for probes. 401/429 on violation.

### Changed
- CI now exercises the model-lifecycle layer on every push (seeds → gate → promote →
  retrainpolicy on the tiny synthetic workspace) and attests the promoted champion
  both ways: `verify` for the bytes, `canary` for the behavior. `configs/ci.yaml`
  trims the new studies to CI scale; `make lifecycle` mirrors the sequence locally.
- `netsentry train supervised` now embeds behavioral canaries in the bundle it
  persists (the artifact promotion snapshots as champion), so `netsentry canary`
  can attest any deployable bundle — not only the serving one. Embedding lives in
  the persist path; the analysis suite's refits pay nothing.
- CI now runs the rules and ablation reports in the analysis-suite smoke and adds a
  model-integrity gate (`netsentry provenance` then `netsentry verify`) so a corrupted
  or swapped bundle fails the build; a `make verify` target mirrors it locally.
- The serving API now returns the conformal `prediction_set` and a
  `recommended_action` (`auto_alert` / `auto_clear` / `review`) on every prediction,
  and exposes a decision-theoretic `cost_optimal` threshold profile alongside the
  fixed-FPR ones — so the calibration, cost, and conformal work is live in the
  product surface, not only in the offline reports. Both are computed on the
  exchangeable stratified validation split when the serving bundle is built.
- Serving request metrics are now labelled by the matched route template instead of
  the raw URL path, bounding Prometheus label cardinality (unauthenticated callers
  could otherwise mint unbounded time series via arbitrary paths).

## [0.1.0] — 2026-06-25

First end-to-end release: the pipeline trains, evaluates, and serves, with honest
temporal-split metrics and a synthetic data path so it runs out-of-the-box.

### Added
- Project scaffolding: installable `netsentry` package (PEP 621), typed
  `pydantic-settings` configuration with YAML loaders, structured logging, global
  seeding, and a Typer CLI (`download`/`prep`/`train`/`eval`/`serve`/`benchmark`).
- Tooling: ruff, black, mypy, pytest configuration; pre-commit hooks; a Makefile;
  and a GitHub Actions CI workflow (lint, typecheck, test on Python 3.11/3.12).
- Data ingestion: a single-source-of-truth `schema.py` (feature columns,
  identifier/leaky columns, label vocabulary, per-day attack layout); an
  idempotent, checksum-verifying `download` command; and a schema-faithful
  synthetic data generator (with the dataset's defects and imbalance) for
  development and CI. Data Card filled in.
- Cleaning pipeline (`clean.py`): whitespace-stripped headers, Inf→NaN, exact
  duplicate removal, label normalization/consolidation, binary + multiclass
  targets, and configurable negative-sentinel handling — each step logged with
  before/after counts. `netsentry prep` writes `data/processed/clean.parquet`.
- EDA notebook (`notebooks/01_eda.ipynb`) and `docs/EDA_SUMMARY.md` covering
  imbalance, missingness, feature signal, and the `Destination Port` leakage trap.
- Honest splitting (`split.py`): temporal/by-day (headline), stratified
  (reference), and leave-one-attack-out (anomaly) strategies, with validation
  carved from train only and content-hashed, persisted partitions.
- Leakage-safe feature pipeline (`features/pipeline.py`): a single
  `ColumnTransformer` (train-fit median impute → scale → optional port encoding)
  with `remainder="drop"` as a firewall. `netsentry prep` now persists both
  split strategies. Added the no-leakage, fit-on-train-only, and split-integrity
  test battery.
- Supervised models (`models/`): a common `BaseModel` interface, majority +
  logistic-regression baselines, and a gradient-boosted classifier (LightGBM,
  scikit-learn `HistGradientBoosting` fallback) with balanced sample weights,
  early stopping, and deterministic seeding. A deployable `ModelBundle`
  (pipeline + model + metadata) is the single serving artifact.
- Training (`training/`): `netsentry train supervised` fits on the temporal
  split, trains baselines, evaluates honestly, and logs params/metrics/artifacts/
  environment to MLflow (with a local-file fallback). Determinism test included.
- Evaluation (`evaluation/`): operational metrics (PR-AUC, ROC-AUC, per-class
  P/R/F1, TPR@fixed-FPR with val-chosen thresholds, alerts/day); PR/ROC/threshold/
  confusion figures; and a `netsentry eval` report contrasting the honest temporal
  split with the optimistic stratified split. Metrics are unit-tested on
  hand-computed cases.
- Anomaly detection (`models/anomaly.py`, `training/train_anomaly.py`): benign-only
  Isolation Forest and a PyTorch autoencoder with FPR-calibrated thresholds.
  `netsentry train anomaly` reports leave-one-attack-out detection per held-out
  class and an ensemble comparison (supervised + anomaly) on the temporal split.
- Explainability (`explain/shap_explainer.py`): SHAP `TreeExplainer` with a
  feature-importance fallback; a global-importance figure/section in the eval
  report and top-k per-prediction contributions for the API.
- Serving (`serving/`): a FastAPI service loading one pipeline+model+anomaly
  bundle at startup — `/health`, `/predict`, `/predict/batch`, `/metrics`
  (Prometheus) — with request validation (422 on bad input), operator-selectable
  threshold profiles, SHAP explanations in every response, and latency middleware.
  `netsentry serve` runs it; `netsentry benchmark` reports p50/p95/p99 + throughput.
- Containerization & CI: multi-stage, non-root `serve`/`train` Docker images, a
  `docker-compose.yml` (with an optional MLflow service), and a CI workflow that
  runs lint/typecheck/test, a synthetic train-smoke + slow tests, a non-blocking
  `pip-audit`, and a serving-image build. Makefile docker/smoke targets added.
- Documentation: README with honest headline results and the methodology story,
  a completed model card and data card, an architecture overview, an MIT license,
  and `NOTES.md` capturing decisions and self-audits.
