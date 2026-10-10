# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end real-time fraud detection feature store pipeline:

**Producer → Kafka → Spark Streaming → Redis (online) / MinIO (offline) → XGBoost (MLflow) → FastAPI scoring service**

- `producer/` publishes synthetic transactions to Kafka topic `raw_transactions` and each transaction's fraud label to a **separate** topic `fraud_labels` (simulated chargeback feed). The transaction stream never carries the label.
- `spark/feature_engineering.py` (Structured Streaming) computes 5-minute windows sliding every 1 minute, plus per-transaction flags, and runs 5 queries: Bronze raw JSON → MinIO, windowed features → Redis, transaction flags → Redis, windowed features → MinIO Silver Parquet, labels → MinIO Parquet.
- `feast/features.py` declares a feature view over Silver Parquet. It is **not** on the live serving path. `models/train.py --historical` uses it via `get_historical_features()` for point-in-time-correct training data.
- `models/train.py` trains XGBoost (synthetic data by default, or Feast historical features) with MLflow tracking.
- `api/main.py` (FastAPI) loads the latest model from `mlruns/`, reads user features directly from Redis, and returns a fraud probability (+ optional SHAP explanations).

## Running the system

Infrastructure (Kafka, Zookeeper, kafka-ui, Redis, MinIO, Prometheus, Grafana) is dockerized. Spark, the producer, training and the API can run locally against it. The `Makefile` wraps everything (`make help`).

```bash
cp .env.example .env          # set API_KEYS — the API raises at import without it
make up                       # docker compose up -d
make jars                     # once per clone: Spark connector JARs → spark/jars/ (gitignored, ~290 MB)
source venv/bin/activate && make install
make produce                  # terminal A
make spark                    # terminal B (runs `make jars` first)
make train                    # synthetic training
make api                      # terminal C, uvicorn :8000
make health                   # smoke test (sends X-API-Key from .env)
make test                     # pytest

# Historical training (real point-in-time features):
# let producer + spark run 20+ min (Silver windows land ~15 min late: 5m window + 10m watermark)
make feast-apply && make train-historical
```

Endpoints: `POST /score` (`?explain=true` for SHAP), `POST /explain`, `GET /user/{id}`, `GET /health`, `GET /metrics`. All except `/health` and `/metrics` require the `X-API-Key` header.

## Critical files

- `features/__init__.py` — **the feature contract.** `FEATURE_ORDER`, thresholds, and `build_feature_vector(user_features, amount, merchant_country)`. Both training and serving must build model input through this function.
- `spark/feature_engineering.py` — streaming entry point. Session timezone is UTC, and Parquet timestamps are `TIMESTAMP_MICROS` (Feast needs tz-aware timestamps). Watermark 10 min. Needs the JARs in `spark/jars/`.
- `spark/windowing.py` — `select_trailing_window()`: which of a user's 5 overlapping windows goes to Redis (the oldest one containing their latest activity = trailing 5 min). Picking the latest `window_start` is a known bug: it only covers ~1 minute.
- `spark/redis_writer.py` — `user_features:{user_id}` (TTL **5 min**, so stale windows expire) and `txn_flags:{txn_id}` (TTL 1h).
- `spark/validation.py`, `spark/quarantine.py` — invalid rows go to `s3://features/quarantine/invalid_records/` (fallback `/tmp/quarantine/`). Quarantine never kills a query.
- `feast/features.py` + `feast/feature_store.yaml` — `timestamp_field="window_end"` (window_start would leak future transactions), TTL 2 min, `FileSource` at `s3://features/windowed_features/` with `s3_endpoint_override` for MinIO. Registry: `feast/registry.db`.
- `models/train.py` — `generate_synthetic_data()`, `load_labeled_transactions()` (Bronze ⨝ labels → entity df), `build_historical_training_set()` (Feast PIT join), `build_feature_matrix()`. `--historical` raises `HistoricalDataError` instead of falling back to synthetic data.
- `api/main.py` — model discovery scans `mlruns/*/*/meta.yaml` for `status == 3`. `_fetch_user_features()` reads Redis and never raises. Risk: `>0.7` HIGH, `>0.4` MEDIUM.
- `api/auth.py` — requires `API_KEYS` env var at import (comma-separated). `api/rate_limit.py`, `api/metrics.py`, `api/explain.py` (SHAP or importance fallback).
- `scripts/download_jars.sh` — JAR versions must stay in sync with `spark/Dockerfile` (Spark 3.5.1, Scala 2.12, kafka-clients 3.4.1).

## Conventions and gotchas

- **Never re-derive features inline.** Transaction flags (`is_foreign`, `is_high_amount`, `is_suspicious`) come from the scored transaction's own amount and country, never from aggregates like `max_amount_5min`. Missing windowed values (no Redis key / NaN from the offline join) become 0.0 in both paths. Parity tests: `tests/test_training.py::TestTrainingServingParity`, `tests/test_api.py::TestScoreUsesTrainingFeaturePath`. Changing the feature set requires retraining.
- **Feast file offline store quirks (0.40, still present in 0.66).** It crashes when an entity's user has no Silver rows, and silently drops entity rows whose windows are all outside the TTL. `build_historical_training_set()` works around both. Keep that workaround and `tests/test_historical_feast.py`.
- **Online vs offline window is an approximation.** Redis holds `[T-4m, T+1m)` around the latest event, while Feast uses the latest closed window `[t-5m, t)`.
- **Kafka dual listeners.** Host clients use `localhost:9092`, containers use `kafka:29092`. Don't change `KAFKA_ADVERTISED_LISTENERS` without updating both.
- **MinIO.** The `features` bucket is created by the `minio-init` sidecar. Endpoint/creds are wired in the Spark session config (`MINIO_*` env vars), on the Feast `FileSource` (`s3_endpoint_override`), and as `AWS_*` env vars for PyArrow (defaulted by `make feast-apply` and `train.py`).
- **CI runs Python 3.9** (`.github/workflows/ci.yml`). Don't use 3.10+ syntax such as `X | None` in annotations without `from __future__ import annotations`. CI installs deps explicitly (not from `requirements.txt`), so add new runtime imports there too. Jobs: Lint, Test, Security (Bandit, pip-audit, Trivy pinned by SHA).
- **Tests.** `pytest` (≈145 tests). `tests/test_windowing.py` and `tests/test_historical_feast.py` start a local Spark session and skip if Java/PySpark/Feast are missing. In sandboxes without hostname resolution, set `SPARK_LOCAL_IP=127.0.0.1 SPARK_LOCAL_HOSTNAME=localhost`.
- **Labels.** `is_fraud` exists only on the `fraud_labels` topic / `s3://features/labels/`. In production it would be a delayed chargeback feed.
- **Not committed:** `mlruns/`, `dump.rdb`, `venv/`, `.env`, `spark/jars/`, `feast/registry.db` (see `.gitignore`; CI fails if any are tracked).
