# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once released.

## [Unreleased]

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
