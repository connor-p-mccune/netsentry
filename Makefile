# NetSentry developer tasks. On Windows without `make`, run the underlying
# `python -m ...` commands directly (see each target).
PY ?= python

.PHONY: help install install-all lint format typecheck test test-fast check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install: ## Editable install with dev + train extras, and pre-commit hooks
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,train]"
	pre-commit install

install-all: ## Editable install with every extra (adds torch + serve)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[all]"

lint: ## ruff + black --check
	$(PY) -m ruff check netsentry tests
	$(PY) -m black --check netsentry tests

format: ## Auto-fix lint + format
	$(PY) -m ruff check --fix netsentry tests
	$(PY) -m black netsentry tests

typecheck: ## mypy on the package
	$(PY) -m mypy netsentry

test: ## Full test suite with coverage
	$(PY) -m pytest --cov=netsentry --cov-report=term-missing

test-fast: ## Skip slow (heavy-dep / dataset) tests
	$(PY) -m pytest -m "not slow"

check: lint typecheck test ## Run before every commit

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage build dist ./*.egg-info
