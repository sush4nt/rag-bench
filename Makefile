.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
CONFIG_SCIFACT := configs/scifact.yaml
CONFIG_FIQA := configs/fiqa.yaml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Sync dependencies (add --extra ragas for generation eval)
	$(UV) sync

.PHONY: install-ragas
install-ragas: ## Sync deps including the RAGAS extra
	$(UV) sync --extra ragas --extra dev

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
.PHONY: index-scifact
index-scifact: ## Download + index SciFact into Qdrant + bm25s
	$(UV) run python -m ragbench.indexing.build --config $(CONFIG_SCIFACT)

.PHONY: index-fiqa
index-fiqa: ## Download + index FiQA (takes 8-12 min)
	$(UV) run python -m ragbench.indexing.build --config $(CONFIG_FIQA)

.PHONY: index-all
index-all: index-scifact index-fiqa ## Index both datasets

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
.PHONY: serve
serve: ## Start FastAPI (uv run, no Docker) on :8080
	$(UV) run uvicorn ragbench.serving.app:app --host 0.0.0.0 --port 8080 --reload

.PHONY: frontend
frontend: ## Start the Vite dev server on :5173
	cd frontend && npm install && npm run dev

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
.PHONY: eval-scifact
eval-scifact: ## Run retrieval (+RAGAS) eval on SciFact, log to MLflow
	$(UV) run python -m ragbench.evaluation.runner --config $(CONFIG_SCIFACT)

.PHONY: eval-fiqa
eval-fiqa: ## Run retrieval (+RAGAS) eval on FiQA, log to MLflow
	$(UV) run python -m ragbench.evaluation.runner --config $(CONFIG_FIQA)

# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
.PHONY: docker-up
docker-up: ## Full stack: docker compose up (ragbench + qdrant + mlflow + prom + grafana)
	docker compose up --build

.PHONY: docker-down
docker-down: ## Tear down the stack
	docker compose down

.PHONY: load-test
load-test: ## k6 load test against local serving (FiQA)
	k6 run load_testing/fiqa_load.js

.PHONY: test
test: ## Run the pytest suite
	$(UV) run pytest

.PHONY: lint
lint: ## Ruff lint + format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests
