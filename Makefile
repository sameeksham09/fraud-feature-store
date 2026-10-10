.PHONY: up down logs produce jars spark train train-historical feast-apply api health test lint clean help

# Read API_KEYS (etc.) from .env if present, so `make health` sends the same key
# the API was started with. Falls back to the docker-compose dev default.
-include .env
comma := ,
API_KEY ?= $(firstword $(subst $(comma), ,$(or $(API_KEYS),dev-api-key-change-in-prod)))

# ── Infrastructure ──────────────────────────────────────────────────────
up:  ## Start all infra + build Spark & API images
	docker compose up -d --build
	@echo "✅ Infra up — kafka:9092  redis:6379  minio:9001  api:8000  grafana:3000  prometheus:9090"
	@echo "   Spark streaming: 4 queries (bronze JSON + gold Redis + silver Parquet)"

up-infra:  ## Start infra only (no app containers) — for local python spark/api
	docker compose up -d zookeeper kafka kafka-ui redis minio minio-init prometheus grafana
	@echo "✅ Infra only — kafka:9092  redis:6379  minio:9001"

up-prod:  ## Start with prod overrides (requires API_KEYS env var)
	@if [ -z "$$API_KEYS" ]; then echo "❌ API_KEYS is required — e.g. API_KEYS=supersecret make up-prod"; exit 1; fi
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
	@echo "✅ Prod infra up — replication=3, API keys from env"

down:  ## Stop all containers (keeps volumes)
	docker compose down

down-v:  ## Stop all + remove volumes (wipes Redis/MinIO/Spark checkpoints)
	docker compose down -v

logs:  ## Tail all container logs
	docker compose logs -f

logs-spark:  ## Tail Spark streaming logs
	docker compose logs -f spark

logs-api:  ## Tail API logs
	docker compose logs -f api

ps:  ## Show container status
	docker compose ps

build:  ## Build Spark + API images (no start)
	docker compose build spark api

# ── Python env ──────────────────────────────────────────────────────────
install:  ## Install all Python deps into active venv
	pip install -r requirements.txt
	pip install -r producer/requirements.txt
	pip install -r spark/requirements.txt

# ── Pipeline phases ─────────────────────────────────────────────────────
produce:  ## Run transaction producer (2 tx/sec)
	python producer/produce_transactions.py

jars:  ## Download Kafka + S3A connector JARs into spark/jars/ (gitignored; needed once per clone)
	./scripts/download_jars.sh

spark: jars  ## Run Spark streaming pipeline (5 queries: bronze→S3, windows→Redis, flags→Redis, silver→S3, labels→S3)
	python spark/feature_engineering.py

train:  ## Train XGBoost model (synthetic 10k rows)
	python models/train.py

train-historical:  ## Train on Feast point-in-time features (needs ~20 min of producer+spark data, then feast-apply)
	python models/train.py --historical

api:  ## Serve FastAPI scoring service on :8000
	uvicorn api.main:app --reload --port 8000

mlflow:  ## Launch MLflow UI on :5000
	mlflow ui --port 5000

feast-apply:  ## Register Feast feature view (MinIO creds default to minioadmin)
	cd feast && AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID:-minioadmin} AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY:-minioadmin} feast apply

# ── Verification ────────────────────────────────────────────────────────
health:  ## Hit API health + score endpoints
	curl -s http://localhost:8000/health | python -m json.tool
	@echo "---"
	curl -s -X POST http://localhost:8000/score \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $(API_KEY)" \
	  -d '{"user_id":"user_0001","amount":1500,"merchant_category":"electronics","merchant_country":"NG"}' \
	  | python -m json.tool

test:  ## Run pytest suite
	pytest -v

lint:  ## Lint with ruff (if installed)
	ruff check .

# ── Load testing ────────────────────────────────────────────────────────
load-k6:  ## k6 load test against /score (requires k6 + running api)
	@if ! command -v k6 >/dev/null 2>&1; then echo "❌ k6 not found — brew install k6  or: docker run --rm -i --network host -v \$$PWD/load:/load grafana/k6 run /load/k6_score.js"; exit 1; fi
	API_KEYS=$${API_KEYS:-dev-api-key-change-in-prod} k6 run load/k6_score.js

load-k6-docker:  ## k6 via Docker (no local install needed)
	docker run --rm -i --network host -v $(PWD)/load:/load grafana/k6 run /load/k6_score.js

load-locust:  ## Locust load test (requires locust)
	@if ! command -v locust >/dev/null 2>&1; then echo "❌ locust not found — pip install -r load/requirements.txt"; exit 1; fi
	locust -f load/locustfile.py --host http://localhost:8000

# ── Cleanup ─────────────────────────────────────────────────────────────
clean:  ## Remove local artifacts (checkpoints, mlruns kept unless --hard)
	rm -rf /tmp/checkpoints spark-warehouse metastore_db derby.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

clean-hard: clean  ## Also remove mlruns + feast registry (irreversible)
	rm -rf mlruns feast/registry.db dump.rdb

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
