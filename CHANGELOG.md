# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once released.

## [Unreleased]

### Added
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

### Changed
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
