# Model Card — NetSentry

> The limitations section is a feature, not a weakness. Numbers here are on the
> **synthetic** CIC-IDS2017 stand-in (see the Data Card) and illustrate the
> methodology, not real-world performance.

## Model details
- **Name / version:** NetSentry v0.2.0
- **Type:** Supervised gradient-boosted classifier (LightGBM, with a scikit-learn
  `HistGradientBoosting` fallback) + an unsupervised anomaly stack (Isolation
  Forest and a benign-only PyTorch autoencoder).
- **Input:** CICFlowMeter network-flow features (CIC-IDS2017 schema, identifier
  columns removed) — pre-computed, or extracted from a classic-pcap capture by
  `netsentry pcap`.
- **Output:** attack class + attack probability, an is-attack decision at a
  selectable false-positive-budget threshold, an anomaly score, and SHAP top
  contributing features.

## Intended use
- **Intended:** education, research, and demonstration of ML-based intrusion
  detection on flow data; a reference for honest evaluation methodology.
- **Out of scope:** a drop-in production NIDS; live/streaming capture and
  IPv6 (offline pcap/pcapng ingestion is supported); adversarial-evasion-robust
  deployment; any safety-critical sole control.

## Training data
- CIC-IDS2017 (see [`DATA_CARD.md`](DATA_CARD.md)), or the bundled synthetic
  stand-in. Known issues handled: leakage columns dropped, Inf/NaN imputed
  (train-fit), duplicates removed, labels normalised/consolidated, imbalance
  addressed with class weights.

## Evaluation methodology
- **Headline split:** temporal / by-day (train Mon–Wed, test Thu–Fri) — honest,
  because near-duplicate flows from an attack burst no longer straddle the split
  and later-day attacks are largely novel. The headline task is **binary**
  (attack vs benign) because attack *classes* are disjoint across the day
  boundary, so multiclass "naming" is degenerate there.
- **Reference split:** stratified random — optimistic; reported only to expose
  the over-optimism gap.
- **Multiclass naming** is evaluated on the stratified split (all classes appear
  in training); the served model is this multiclass model.
- **Novelty:** the anomaly detector is trained on benign traffic only and
  evaluated leave-one-attack-out.
- **Metrics:** PR-AUC (primary), per-class precision/recall/F1, macro/weighted
  F1, confusion matrix, ROC-AUC, and **TPR at fixed FPR (0.1% and 1%)** with the
  threshold chosen on validation. Accuracy is deliberately not a headline metric.

## Results (synthetic stand-in; see `docs/reports/evaluation.md`)
- PR-AUC attack-vs-benign — temporal (honest): **0.529** (majority baseline
  0.250); stratified (optimistic): **0.786**; over-optimism gap: **+0.257**.
- Detection rate at a 0.1% / 1% false-positive budget (temporal): **9.1% / 21.0%**
  (≈441 / 6,571 false alerts per day at an assumed 1M flows/day).
- Multiclass naming (stratified): macro-F1 0.31, weighted-F1 0.84 — the macro
  average is held down by genuinely hard rare classes (e.g. Bot), reported
  honestly rather than hidden.
- Anomaly detector, average detection of held-out attack classes at a 1% FP
  budget: autoencoder **8.5%**, Isolation Forest 4.3%. The supervised+anomaly
  ensemble reaches **0.537** PR-AUC on the temporal test, beating either component
  alone (supervised 0.529, anomaly 0.433).
- Serving latency: p50 ≈ 47 ms / p95 ≈ 56 ms per single-flow request (SHAP per
  request dominates), ~21 req/s single-process.

## Limitations & ethical considerations
- **Dataset age & bias.** Trained on 2017 traffic from a specific testbed (here a
  synthetic stand-in); concept drift and a non-representative attack mix mean
  real-world performance will differ. Do not assume these numbers transfer.
- **False positives cause alert fatigue.** The binding constraint in a SOC is
  analyst time. Thresholds are tunable and the operating point (and its
  alerts/day implication) is reported explicitly rather than hidden behind an AUC.
- **Evadable by an adaptive attacker.** This *is* now evaluated (see
  [`docs/reports/robustness.md`](reports/robustness.md)): an attacker who shapes a
  flow's controllable volume/timing features toward benign drops the supervised
  detection rate sharply (on the synthetic stand-in, full mimicry takes it from
  ~83% to ~0% at the 1%-FPR operating point). This is expected for a tabular tree
  model and is the explicit case for pairing it with the benign-only anomaly
  detector and not relying on it as a sole control.
  **One family of that attack is removable.** A variant constrained non-decreasing
  in the 39 attacker-inflatable features cannot be evaded by padding at all — the
  property is structural rather than learned, holds for every input in the domain,
  and costs nothing measurable (see
  [`docs/reports/monotonic.md`](reports/monotonic.md)). An attacker who can *remove*
  bytes or packets is outside that guarantee; the shipped model is unconstrained,
  so the numbers above still describe it.
- **Flow features, not packets.** The model consumes flow statistics, so it
  inherits any bias or error in the flow-extraction step — including NetSentry's
  own capture stack, whose documented departures from CICFlowMeter (bulk
  features, zero-duration rates, close semantics) apply to `netsentry pcap` input.
- **It is a post-mortem detector, structurally.** Exporters emit one record per
  *finished* flow, and most of the features are undefined until then, so the
  earliest instant a verdict can exist is the instant the flow closes — which for a
  flow that stops without a teardown is the exporter's idle timeout away. Detection
  latency is therefore a property of the traffic rather than of the model, and it is
  measured rather than assumed in
  [`docs/reports/earliness.md`](reports/earliness.md), which also finds that the
  features requiring the wait are the ones that transfer worst across the day
  boundary.
- **Explanations are local approximations.** SHAP values explain the model, not
  ground-truth causation; treat top features as investigative leads.
