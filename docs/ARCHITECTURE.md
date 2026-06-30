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
P/R/F1, TPR@fixed-FPR, alerts/day), plots, and a report contrasting the honest
temporal split with the optimistic stratified split.

**Explain (`netsentry/explain`)** — SHAP global importance and per-prediction
attributions, surfaced both in the report and in API responses.

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

## Tech stack

Python 3.11 · scikit-learn · LightGBM · PyTorch · SHAP · pandas/numpy/pyarrow ·
pydantic-settings + YAML · MLflow · FastAPI + uvicorn · prometheus-client ·
pytest/ruff/black/mypy/pre-commit · Docker · GitHub Actions.

## Data flow at serve time

`client → POST /predict (flow JSON) → pydantic validate → fitted pipeline.transform
→ LightGBM.predict_proba + anomaly score → threshold (profile) → SHAP top-k →
response {class, prob, anomaly_score, is_anomaly, top_features, version}`.

## Out of scope (and honestly stated as such in the model card)

Live packet capture / NetFlow extraction (the model consumes pre-computed flow
features); production-grade retraining/serving infra; *adversarial-evasion
hardening* (evasion is now measured — see `docs/reports/robustness.md` — but the
model is not yet hardened against it). NetSentry is a rigorous reference
implementation and demo, not a drop-in enterprise NIDS — and the model card says so.
