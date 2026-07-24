# Real-Time Fraud Detection Feature Store

End-to-end ML pipeline that detects fraudulent credit-card transactions in real time, built around a proper online/offline feature store architecture.

```
┌──────────────┐    ┌─────────┐    ┌──────────────────┐    ┌──────────────┐
│   Producer   │───▶│  Kafka  │───▶│  Spark Streaming │───▶│    Redis     │  online
│  (synthetic) │    │  topic  │    │  (5-min windows) │    │  (features)  │
└──────────────┘    └─────────┘    │                  │    └──────┬───────┘
                                   │                  │           │
                                   │                  │    ┌──────▼───────┐  offline
                                   │                  │───▶│   MinIO/S3   │
                                   └──────────────────┘    │  (parquet)   │
                                                           └──────┬───────┘
                                                                  │ historical
                                                                  ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────────┐
│   FastAPI    │───▶│     Redis       │    │   MLflow     │    │   XGBoost  │
│  /score      │    │ (direct read)   │    │  (registry)  │    │   model    │
└──────────────┘    └─────────────────┘    └──────────────┘    └────────────┘
       ▲
       │ POST {user_id, amount, merchant_category, merchant_country}
       │ → fraud_probability + risk_level
```

**Producer** generates synthetic transactions onto Kafka topic `raw_transactions`.
**Spark Structured Streaming** consumes the topic, computes 5-minute tumbling windowed aggregates and per-transaction flags, and writes them to **Redis** (online store) and **MinIO/S3** (offline store).
**Feast** declares a feature view over the offline Parquet and is used for historical training data via `get_historical_features()`. The live API does **not** go through Feast — it reads windowed aggregates directly from Redis.
**XGBoost** is trained on synthetic data, tracked in **MLflow**.
**FastAPI** reads user features from Redis at request time and returns a fraud probability.

## Why this architecture

Real-time fraud detection needs two very different kinds of features at the same time:

- **Windowed aggregates over recent history** (5-minute txn count, total amount, etc.) — these are *streaming* features that change continuously and must be served in single-digit milliseconds.
- **Historical features for offline training** (the same windowed features, but back-filled) — for the model to learn from, queried in batch.

The online/offline split is the answer: Redis holds the latest value per user for fast lookups, MinIO/S3 holds the immutable event log for training. **Feast** is the abstraction layer over the offline side — it gives the training pipeline point-in-time-correct historical features without hand-rolling that logic. The serving API skips Feast entirely and reads Redis directly, since it needs the live streaming aggregates rather than a materialized snapshot.

## Prerequisites

- **Docker** (for Kafka, Redis, MinIO, Zookeeper, kafka-ui)
- **Python 3.9+** with `venv`
- **Java 11+** (PySpark requirement)
- Kafka connector JARs in `spark/jars/` (already present: `spark-sql-kafka`, `kafka-clients`, `hadoop-aws`, `aws-java-sdk-bundle`, `commons-pool2`, `spark-token-provider-kafka`)

## Running the system

### 1. Start infrastructure

```bash
docker compose up -d
```

This brings up:
- **Kafka** on `localhost:9092` (external) / `kafka:29092` (internal Docker network)
- **Zookeeper** on `localhost:2181`
- **kafka-ui** at <http://localhost:8080>
- **Redis** on `localhost:6379`
- **MinIO** on `localhost:9000` (API) / `localhost:9001` (console — `minioadmin` / `minioadmin`)

A one-shot `minio-init` sidecar automatically creates the `features` bucket in MinIO on first start (using `mc mb --ignore-existing`, so it's safe to re-run). Spark's offline writer targets `s3a://features/windowed_features/` and relies on this bucket existing.

### 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r producer/requirements.txt
pip install -r spark/requirements.txt
pip install mlflow xgboost scikit-learn pandas numpy matplotlib pyyaml redis fastapi "uvicorn[standard]" pydantic feast s3fs
```

`s3fs` is needed by Feast's offline store to read Parquet from S3.

### 3. Produce transactions (terminal A)

```bash
python producer/produce_transactions.py
```

~3% fraud rate, 200 users, 2 transactions/second. You can watch them in <http://localhost:8080> (kafka-ui).

### 4. Run the Spark feature pipeline (terminal B)

```bash
python spark/feature_engineering.py
```

Three streaming queries:
- **query1** → windowed features → Redis (every 30s)
- **query2** → transaction flags → Redis (every 10s)
- **query3** → windowed features → MinIO/S3 (every 30s)

The 5-minute tumbling window uses a 1-minute slide. Watermark is 10 minutes.

### 5. (Optional) Register the feature view for historical training

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
cd feast
feast apply
cd ..
```

This writes the registry to `feast/registry.db` and is only required if you want to build historical training sets via Feast's `get_historical_features()` against the Parquet in `s3://features/windowed_features/`. **It is not required for the live `/score` API**, which reads windowed aggregates directly from Redis under the `user_features:{user_id}` keys written by Spark.

### 6. Train the model (run once, or after schema changes)

```bash
python models/train.py
```

Trains an XGBoost classifier on a synthetic 10k-row dataset. Feature order must match `FEATURE_ORDER` in `api/main.py`. Logs params, metrics, and a feature-importance plot to the `fraud_detection` MLflow experiment (`./mlruns`).

> **Note on training data.** The training set is generated synthetically with `is_fraud` as a deterministic function of the features (see `models/train.py:46-51`). The model is essentially learning a hand-coded rule, so the reported metrics measure *fit to that rule*, not real-world fraud detection. In production, the labels would come from delayed chargeback reports, not from the streaming pipeline.

### 7. Serve predictions (terminal C)

```bash
uvicorn api.main:app --reload --port 8000
```

- `POST /score` — body: `{user_id, amount, merchant_category, merchant_country}` → `{fraud_probability, risk_level, features_used, redis_features_found, latency_ms}`
- `GET /user/{id}` — returns raw Redis features for one user
- `GET /health` — model + redis status

Risk thresholds: `>0.7` HIGH, `>0.4` MEDIUM, else LOW.

### 8. Inspect runs

```bash
mlflow ui --port 5000
```

Experiment: `fraud_detection`.

## Verifying it works

```bash
# Check Kafka is receiving messages
open http://localhost:8080   # kafka-ui → raw_transactions topic

# Check Spark is writing features to Redis
redis-cli HGETALL user_features:user_0001

# Check Spark is writing Parquet to MinIO
docker exec -it $(docker ps -qf name=minio) mc ls --recursive local/features/

# Hit the scoring API
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_0001","amount":1500,"merchant_category":"electronics","merchant_country":"NG"}'
```

A healthy response will have `"redis_features_found": true` and all four `txn_count_5min`/`total_amount_5min`/`avg_amount_5min`/`max_amount_5min` fields populated in `features_used`.

## Architecture notes & tradeoffs

### Online/offline feature store

This is the core pattern. Spark writes the same `user_transaction_features` schema to two backends:

- **Online store (Redis)** — key `user_features:{user_id}`, hash with the 4 windowed features. The FastAPI API reads from these keys directly at request time.
- **Offline store (MinIO/S3)** — Parquet at `s3://features/windowed_features/`, written by Spark's `query3`. This is the immutable event log used for historical training.

`feast/features.py` declares the feature view over the offline Parquet. It is **not** used for live serving — it's there for `get_historical_features()` to build point-in-time-correct training sets when you graduate from the synthetic dataset in `models/train.py` to real labeled data.

If you ever want to use Feast for online serving too, you can run `feast materialize-incremental` on a schedule to copy the latest offline values into Feast's Redis keyspace. The current architecture deliberately skips this step so the API sees live streaming aggregates, not stale materialized snapshots.

### Kafka dual listeners

The producer and Spark both connect to Kafka on `localhost:9092` (the *external* listener, exposed by Docker to the host). Containers within the Docker network talk to each other on `kafka:29092` (the *internal* listener). Don't change `KAFKA_ADVERTISEDED_LISTENERS` in `docker-compose.yml` without updating both clients.

### MinIO/S3 alignment

The `minio-init` sidecar in `docker-compose.yml` creates the `features` bucket in MinIO on first start (idempotent via `mc mb --ignore-existing`). Once that runs, Spark's offline writer can land Parquet in `s3a://features/windowed_features/`, and the Feast `FileSource` (which reads from `s3://features/windowed_features/`) sees the same data — same bucket, different protocol.

- **For Spark writes** — `spark/feature_engineering.py:30-33` configures `spark.hadoop.fs.s3a.access.key`, `s3a.secret.key`, `s3a.endpoint`, and `s3a.path.style.access=true` directly in the Spark session.
- **For the Feast `FileSource`** — `feast/features.py` passes `s3_endpoint_override="http://localhost:9000"` to the `FileSource` constructor. This is the per-data-source knob that Feast 0.34 uses to point pyarrow's S3FileSystem at MinIO. PyArrow reads the AWS access key/secret from the environment.
- **For the API** — the API does **not** talk to MinIO directly. It reads windowed features from Redis (where Spark writes them), so no S3 env vars are needed at runtime.

When running the `feast` CLI directly (e.g. `feast apply`), set the AWS env vars in your shell so PyArrow can authenticate with MinIO:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

Feast 0.34's `file` offline store config in `feature_store.yaml` is intentionally minimal — only the `type` field is supported. S3 configuration lives on the `FileSource` itself, not at the repo level.

### Partitioning tradeoff

Spark's `query3` writes one Parquet file per micro-batch (every 30 seconds) without date partitioning. At demo scale this is fine. At higher volumes, you'd want `partitionBy("event_date")` in the writeStream to avoid a full bucket scan whenever `get_historical_features()` builds a training set from this Parquet. This is the most obvious next optimization.

### Feature contract

`FEATURE_ORDER` in `api/main.py` must match the column order used in `models/train.py` and the schema declared in `feast/features.py`. Changing the feature set requires retraining. The Feast feature view currently exposes only the 4 windowed features; the 3 transaction-level flags (`is_foreign`, `is_high_amount`, `is_suspicious`) are computed at request time in the API from the request payload, not served from the feature store.

## Project layout

```
fraud-feature-store/
├── api/                          # FastAPI scoring service
│   └── main.py
├── feast/                        # Feast feature store definition
│   ├── feature_store.yaml        # repo: project, registry, online/offline store
│   ├── features.py               # entity + FeatureView definitions
│   └── registry.db               # (generated by `feast apply`, lives at the repo root)
├── models/                       # XGBoost training
│   ├── train.py
│   └── artifacts/                # (generated: feature importance plot)
├── producer/                     # Synthetic transaction generator
│   ├── produce_transactions.py
│   └── requirements.txt
├── spark/                        # Spark Structured Streaming
│   ├── feature_engineering.py
│   ├── redis_writer.py
│   ├── jars/                     # Kafka + S3A connector JARs
│   └── requirements.txt
├── docker-compose.yml
├── mlruns/                       # MLflow tracking (gitignored)
└── README.md
```