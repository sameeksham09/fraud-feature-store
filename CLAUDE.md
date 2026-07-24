# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end real-time fraud detection feature store pipeline:

**Producer → Kafka → Spark Streaming → Redis (online) / MinIO (offline) → XGBoost (MLflow) → FastAPI scoring service** *(Feast declares the offline schema but is not on the live serving path; the API reads directly from Redis.)*

A `producer` generates synthetic transactions onto Kafka topic `raw_transactions`. A Spark Structured Streaming job (`spark/feature_engineering.py`) consumes that topic, computes 5-minute tumbling windowed aggregates and per-transaction flags, and writes them to Redis (online store) and MinIO/S3 (offline store). A Feast feature view (`feast/features.py`) declares the schema over the offline Parquet (used for historical training only). An XGBoost model is trained on synthetic data (`models/train.py`) with MLflow tracking. A FastAPI service (`api/main.py`) loads the latest model from `mlruns/`, reads user features from Redis at request time, and returns a fraud probability.

## Running the system

Infrastructure (Kafka, Zookeeper, kafka-ui, Redis, MinIO) is dockerized. Spark runs locally against these services (note the dual Kafka listeners in `docker-compose.yml`: `kafka:29092` for inside the Docker network, `localhost:9092` for the host — Spark connects via `localhost:9092`).

```bash
# 1. Start infrastructure
docker compose up -d
# kafka-ui:    http://localhost:8080
# minio console: http://localhost:9001  (minioadmin / minioadmin)
# redis:       localhost:6379

# 2. Produce transactions (terminal A) — install in venv first
source venv/bin/activate
pip install -r producer/requirements.txt
python producer/produce_transactions.py

# 3. Run Spark feature pipeline (terminal B)
pip install -r spark/requirements.txt
python spark/feature_engineering.py

# 4. Train model (run once, or after schema changes)
python models/train.py   # requires xgboost, scikit-learn, mlflow in venv

# 5. Serve predictions (terminal C)
uvicorn api.main:app --reload --port 8000
#   POST /score     body: {user_id, amount, merchant_category, merchant_country}
#   GET  /user/{id} returns raw Redis features
#   GET  /health    model + redis status

# 6. Inspect runs
mlflow ui --port 5000   # experiment: fraud_detection
```

`feast apply` (run from the `feast/` directory) is **not required for the live API** — `/score` reads windowed aggregates directly from Redis at key `user_features:{user_id}`. Feast is only needed if you want to build historical training sets via `get_historical_features()` against the Parquet offline store. If you do run it, the registry is at `feast/registry.db`.

## Critical files

- `spark/feature_engineering.py` — Streaming entry point. Three `writeStream` queries: windowed features → Redis (30s), transaction flags → Redis (10s), windowed features → MinIO (30s, S3A). Kafka connector jars must be present in `spark/jars/`. Watermark is 10 minutes; window is 5min slide 1min.
- `spark/redis_writer.py` — `write_user_features_to_redis` (key `user_features:{user_id}`, 24h TTL) and `write_transaction_flags_to_redis` (key `txn_flags:{txn_id}`, 1h TTL).
- `producer/produce_transactions.py` — ~3% fraud rate, 200 users, 2 tx/sec. Emits to topic `raw_transactions` on `localhost:9092`.
- `feast/features.py` + `feast/feature_store.yaml` — single `user_transaction_features` view with 24h TTL, entity `user_id`, online store = Redis, offline `FileSource` reads from `s3://features/windowed_features/` (MinIO). The registry is the file at `feast/registry.db` (resolved relative to the Feast repo, not the cwd).
- `models/train.py` — Synthetic 10k-row dataset. Feature order must match `FEATURE_ORDER` in `api/main.py`. Logs to MLflow experiment `fraud_detection`, tracking URI `file:./mlruns`. Saves feature importance plot to `models/artifacts/`.
- `api/main.py` — On startup scans `mlruns/*/*/meta.yaml` for `status == 3` runs and loads the most recent `artifacts/model/` via `mlflow.xgboost.load_model`. Reads windowed user features directly from Redis at key `user_features:{user_id}` (the keys written by `spark/redis_writer.py`). Falls back gracefully if model or Redis is unavailable. Risk thresholds: `>0.7` HIGH, `>0.4` MEDIUM, else LOW.

## Conventions and gotchas

- **Feature order is the contract.** `FEATURE_ORDER` in `api/main.py` must match the column order used to train in `models/train.py` and the schema in `feast/features.py`. Changing the feature set requires retraining.
- **Kafka dual listeners.** The producer and Spark both connect on `localhost:9092` (external). Containers talk to each other on `kafka:29092` (internal). Don't change `KAFKA_ADVERTISED_LISTENERS` without updating both clients.
- **MinIO/S3 alignment.** Spark writes to `s3a://features/windowed_features/`, Feast reads from `s3://features/windowed_features/` (same bucket, different protocol). The `features` bucket is auto-created on `docker compose up` by the one-shot `minio-init` sidecar (`mc mb --ignore-existing`), so no manual `mc` step is required. The MinIO endpoint/credentials are wired in two places: `spark/feature_engineering.py:30-33` (S3A session config) and `feast/features.py` (`s3_endpoint_override` on the `FileSource`).
- **No `.gitignore` yet.** `mlruns/`, `dump.rdb` (Redis dump), and the `venv/` directory are tracked. Add a `.gitignore` before initial commit.
- **No `requirements.txt` at the repo root.** Dependencies are scattered: `producer/requirements.txt`, `spark/requirements.txt`, and inline imports for `api/` (FastAPI, uvicorn, mlflow, xgboost, scikit-learn, pandas, numpy, matplotlib, pyyaml, redis) and `models/` (mlflow, xgboost, scikit-learn, pandas, numpy, matplotlib). Feast is also needed.
- **No automated tests.** Validation is done by running each phase and inspecting Redis (`redis-cli HGETALL user_features:user_0001`), MinIO, MLflow UI, and the `/score` endpoint.
- **The `is_fraud` label is in the producer's Kafka payload** (`producer/produce_transactions.py:54`) as a comment acknowledges — in production this would not be present in the stream and would only exist as a delayed label.
