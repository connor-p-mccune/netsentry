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
