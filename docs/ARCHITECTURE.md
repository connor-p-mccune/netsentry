# NetSentry — Architecture

## Purpose

A machine-learning network intrusion detection system that classifies network
flows as benign or one of several attack types, and independently flags
anomalous traffic that matches no known attack. Built to demonstrate end-to-end
ML engineering: honest data handling, dual detection, explainability, a served
API, and reproducible MLOps.

## System overview

```
                ┌─────────────────────────────────────────────────────────┐
                │                     TRAINING PIPELINE                     │
                │                                                           │
  CIC-IDS2017   │  download ─► clean ─► split ─► feature pipeline ─► train  │
   (raw CSVs)   │   (data/)   (data/)  (temporal/   (fit on TRAIN    │      │
                │                       stratified/   only)          ▼      │
                │                       leave-1-out)            ┌──────────┐ │
                │                                               │ Supervised│ │
                │                                               │ LightGBM  │ │
                │                                               ├──────────┤ │
                │                                               │ Anomaly   │ │
                │                                               │ IForest + │ │
                │                                               │ Autoencoder│ │
                │                                               └────┬─────┘ │
                │                       evaluate ◄───── SHAP ◄───────┤       │
                │                  (PR-AUC, per-class, TPR@FPR)      │       │
                │                          │                         ▼       │
                │                       MLflow                  pipeline+model│
                │                  (params/metrics/             artifact      │
                │                   artifacts/env)              (versioned)   │
                └────────────────────────────────────────────────┬──────────┘
                                                                   │ load once
                ┌──────────────────────────────────────────────────▼─────────┐
                │                     SERVING (FastAPI)                        │
                │  POST /predict ─► pipeline.transform ─► model.predict_proba  │
                │                   + anomaly score + SHAP top features        │
                │  GET /health   GET /metrics (Prometheus)   POST /predict/batch│
                └──────────────────────────────────────────────────────────────┘
```

## Components

**Data (`netsentry/data`)** — ingestion, a single-source-of-truth schema (columns,
leaky columns, labels), cleaning (whitespace/Inf/dupes/sentinels/label
consolidation → binary + multiclass targets), and the three split strategies.

**Capture (`netsentry/capture`)** — packet-to-verdict ingestion: a pure-stdlib
classic-libpcap reader (both byte orders, µs/ns timestamps, Ethernet/VLAN/raw-IP)
and a bidirectional flow assembler that reimplements the CICFlowMeter aggregation
against the canonical schema module, so a raw capture yields the exact training
columns and is scored by the same engine the API uses (no re-implemented
preprocessing, no skew). A deterministic synthetic-capture writer powers the demo,
the CI smoke, and the parser tests.

**Features (`netsentry/features`)** — one fitted `sklearn` Pipeline /
ColumnTransformer that drops identifiers, imputes (train-fit median), scales, and
optionally encodes `Destination Port`. This is the leakage firewall: fit on train
only, applied identically at train and serve time.

**Models (`netsentry/models`)** — a common interface; a supervised LightGBM
classifier (with baselines) for known attacks; an unsupervised stack (Isolation
Forest + benign-only PyTorch autoencoder) for novel attacks; optional ensemble
risk score.

**Training (`netsentry/training`)** — entrypoints that wire data → features →
model, handle imbalance, seed determinism, and log everything to MLflow; save the
**pipeline+model bundle** as the deployable artifact.

**Evaluation (`netsentry/evaluation`)** — operational metrics (PR-AUC, per-class
P/R/F1, TPR@fixed-FPR, alerts/day) with bootstrap CIs, plots, and a report
contrasting the honest temporal split with the optimistic stratified one; plus the
one-command analysis suite (`netsentry analyze`): cost-optimal thresholds, conformal
selective alerting, per-class and per-service detection slices, the novelty-distance
gap decomposition, rules-vs-model at matched FPR, feature-group ablation, active
learning, learning curves, cross-dataset transfer, and an auto-generated model card.

**Robustness (`netsentry/robustness`)** — the adversarial studies: evasion (mimicry
+ adaptive query search) at inference time and label-flip / benign-pool-contamination
poisoning at training time, each reported as detection-vs-attacker-effort curves.

**Monitoring (`netsentry/monitoring`)** — PSI drift detection (offline report and
rolling serving gauges) and the prequential streaming study that closes the
drift → retrain → recover loop on the later-day stream.

**Intel & governance (`netsentry/intel`, `netsentry/governance`)** — MITRE ATT&CK
tactic/technique mapping in predictions and a coverage report; CycloneDX SBOM and a
model-integrity manifest with a `netsentry verify` CI gate.

**Explain (`netsentry/explain`)** — SHAP global importance and per-prediction
attributions (in the report and in API responses), plus counterfactual recourse:
the minimal feature change that would clear a flagged flow.

**Serving (`netsentry/serving`)** — FastAPI app loading the bundle once; predict /
batch / health / metrics; pydantic contract; Prometheus latency; selectable
threshold profile; benchmarked.

## Key design decisions (and why)

- **Two detectors, not one.** Supervised models only catch what they've seen;
  real intrusions include novel behavior. Pairing a classifier with a benign-only
  anomaly detector mirrors how production NIDS actually defend, and gives a
  genuine "detect the unknown" result to report.
- **Temporal split as the headline.** Random splits on CIC-IDS leak via
  near-duplicate flows within an attack burst. The temporal split is harder and
  more honest; reporting it (and the gap to the optimistic number) is the point.
- **Operational metrics over accuracy.** In a SOC, the binding constraint is
  analyst time, so detection at a fixed false-positive budget is the metric that
  matters. Accuracy on 80%-benign data is meaningless.
- **One artifact = pipeline + model.** Guarantees serve-time preprocessing equals
  train-time preprocessing, eliminating train/serve skew.
- **Explainability as a contract, not a plot.** A flagged flow is only actionable
  with a reason, so SHAP attributions are returned per prediction.
- **Config-driven + MLflow-tracked.** Every result is reproducible from a logged
  config + seed.
- **A lifecycle layer, not just a leaderboard.** Training is followed by decisions,
  each an exit-coded command: the seed-noise floor calibrates the promotion margins
  (`seeds`), a release gate re-asserts the honesty invariants on the artifact that
  ships and fails a too-good score (`gate`), champion/challenger promotion is a
  paired-bootstrap comparison behind a SHA-256-pinned registry (`promote`), every
  persisted bundle carries behavioral canaries the serving runtime must reproduce
  (`canary`), an optional shadow challenger gathers live disagreement evidence, and
  retrain triggers are priced against calendar retraining (`retrainpolicy`) before
  anyone wires the drift alarm to the lever.

## Tech stack

Python 3.11 · scikit-learn · LightGBM · PyTorch · SHAP · pandas/numpy/pyarrow ·
pydantic-settings + YAML · MLflow · FastAPI + uvicorn · prometheus-client ·
pytest/ruff/black/mypy/pre-commit · Docker · GitHub Actions.

## Data flow at serve time

`client → POST /predict (flow JSON) → pydantic validate → fitted pipeline.transform
→ LightGBM.predict_proba + anomaly score → threshold (profile) → SHAP top-k →
response {class, prob, anomaly_score, is_anomaly, top_features, version}`.

## Out of scope (and honestly stated as such in the model card)

Live/streaming packet capture and IPv6 (offline pcap/pcapng ingestion is
in scope via `netsentry pcap`; NetFlow extraction is not) and multi-node production
serving infrastructure. Adversarial evasion is
measured (`docs/reports/robustness.md`) *and* acted on (`docs/reports/hardening.md`
— adversarial training recovers full-mimicry detection, defending only the
perturbation it trains on, as the report states). NetSentry is a rigorous reference
implementation and demo, not a drop-in enterprise NIDS — and the model card says so.
