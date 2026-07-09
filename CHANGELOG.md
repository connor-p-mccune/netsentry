# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once released.

## [Unreleased]

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
