# NetSentry developer tasks. On Windows without `make`, run the underlying
# `python -m ...` commands directly (see each target).
PY ?= python

.PHONY: help install install-all lint format typecheck test test-fast check clean \
	smoke analysis verify lifecycle docker-serve docker-train docker-up docker-monitor docker-down \
	helm-lint helm-template k8s-render k8s-apply

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

smoke: ## Run the full pipeline on a tiny synthetic dataset
	$(PY) -m netsentry.cli download -o configs/ci.yaml
	$(PY) -m netsentry.cli prep -o configs/ci.yaml
	$(PY) -m netsentry.cli train supervised -o configs/ci.yaml
	$(PY) -m netsentry.cli eval -o configs/ci.yaml

analysis: ## Regenerate every analysis report + the index (needs prep first)
	$(PY) -m netsentry.cli analyze

verify: ## Attest the deployed bundle: write SBOM + manifest, then check integrity
	$(PY) -m netsentry.cli provenance
	$(PY) -m netsentry.cli verify

lifecycle: ## Model-lifecycle gates: seed noise, release gate, promotion, retrain policy, canary
	$(PY) -m netsentry.cli seeds
	$(PY) -m netsentry.cli gate
	$(PY) -m netsentry.cli promote
	$(PY) -m netsentry.cli retrainpolicy
	$(PY) -m netsentry.cli canary

docker-serve: ## Build the serving image
	docker build -f docker/Dockerfile.serve -t netsentry-serve .

docker-train: ## Build the training image
	docker build -f docker/Dockerfile.train -t netsentry-train .

docker-up: ## Run the API via docker compose (builds a synthetic model on first run)
	docker compose -f docker/docker-compose.yml up --build

docker-monitor: ## Run the API + Prometheus + Grafana (dashboard at :3000, admin/admin)
	docker compose -f docker/docker-compose.yml --profile monitoring up --build

docker-down: ## Stop the compose stack (all profiles)
	docker compose -f docker/docker-compose.yml --profile monitoring --profile tracking down

helm-lint: ## Lint the serving Helm chart
	helm lint deploy/helm/netsentry

helm-template: ## Render the Helm chart to stdout (preview the manifests)
	helm template netsentry deploy/helm/netsentry

k8s-render: ## Render the raw Kustomize manifests to stdout
	kubectl kustomize deploy/k8s

k8s-apply: ## Apply the raw Kustomize manifests to the current kube-context
	kubectl create namespace netsentry --dry-run=client -o yaml | kubectl apply -f -
	kubectl -n netsentry apply -k deploy/k8s

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage build dist ./*.egg-info
