"""
Phase 6: Fraud Scoring API
FastAPI endpoint that:
  - Loads the latest XGBoost model from mlruns/ on startup
  - Looks up user features from Redis (key `user_features:{user_id}`)
  - Returns fraud probability + risk level

Note: this service reads live aggregates directly from Redis. Feast is used
separately for offline training via the file source (see feast/features.py)
and does not need to be materialized for this API to work.
"""

import os
import sys
import time
import glob
from typing import Optional

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import mlflow.xgboost
import numpy as np
import redis
import yaml

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Response
from pydantic import BaseModel, validator
from fastapi.security import APIKeyHeader
from fastapi.security.api_key import APIKey

# Import our metrics module
from api.metrics import (
    API_REQUESTS_TOTAL,
    PREDICTIONS_TOTAL,
    REDIS_LOOKUPS_TOTAL,
    ERRORS_TOTAL,
    REQUEST_LATENCY,
    MODEL_INFERENCE_TIME,
    REDIS_LOOKUP_TIME,
    TRANSACTION_AMOUNT,
    MODEL_LOADED,
    REDIS_CONNECTED,
    LAST_FRAUD_PROBABILITY,
    metrics as prometheus_metrics,
)

# Import auth and rate limiting
from api.auth import get_api_key, API_KEY_HEADER
from api.rate_limit import check_rate_limit

# Import shared feature computation module
from features import (
    compute_is_foreign,
    compute_is_high_amount,
    compute_is_suspicious,
)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
REDIS_HOST   = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT   = int(os.environ.get("REDIS_PORT", "6379"))

FEATURE_ORDER = [
    "txn_count_5min",
    "total_amount_5min",
    "avg_amount_5min",
    "max_amount_5min",
    "is_foreign",
    "is_high_amount",
    "is_suspicious",
]

# ── App state ─────────────────────────────────────────────────────────────────
_model        = None   # XGBoost model
_redis_client: Optional[redis.Redis] = None
_redis_last_error: Optional[Exception] = None


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _redis_client

    # Startup
    try:
        model_uri = _find_latest_model_uri()
        _model = mlflow.xgboost.load_model(model_uri)
        MODEL_LOADED.set(1)  # Set gauge to 1 = loaded
        print(f"✅ Model loaded from: {model_uri}")
    except Exception as exc:
        MODEL_LOADED.set(0)  # Set gauge to 0 = not loaded
        ERRORS_TOTAL.labels(error_type='model_load').inc()
        print(f"⚠️  Model load failed: {exc}")

    try:
        _redis_client = _get_redis()
        REDIS_CONNECTED.set(1)  # Set gauge to 1 = connected
        print(f"✅ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as exc:
        REDIS_CONNECTED.set(0)  # Set gauge to 0 = not connected
        ERRORS_TOTAL.labels(error_type='redis_connection').inc()
        print(f"⚠️  Redis unavailable: {exc}")

    yield  # App is running

    # Shutdown (cleanup if needed)
    if _redis_client:
        _redis_client.close()
        print("🔌 Redis connection closed")


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_latest_model_uri() -> str:
    """
    Scan mlruns/ for all FINISHED runs (status=3) in any experiment,
    sort by end_time descending, return the artifact URI of the most
    recent one that has a model/ artifact directory.
    """
    run_metas = glob.glob(
        os.path.join(MLRUNS_DIR, "*", "*", "meta.yaml")
    )

    candidates = []
    for path in run_metas:
        with open(path) as f:
            meta = yaml.safe_load(f)

        # status 3 = FINISHED; skip experiment-level meta.yaml (no end_time)
        if meta.get("status") != 3:
            continue
        model_dir = os.path.join(
            os.path.dirname(path), "artifacts", "model"
        )
        if not os.path.isdir(model_dir):
            continue

        candidates.append((meta.get("end_time", 0), model_dir))

    if not candidates:
        raise RuntimeError(
            f"No finished MLflow runs with a model artifact found in {MLRUNS_DIR}"
        )

    candidates.sort(reverse=True)
    best_model_dir = candidates[0][1]
    return best_model_dir


def _get_redis() -> redis.Redis:
    """Create a new Redis connection."""
    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT,
        decode_responses=True,
        socket_keepalive=True,
        socket_connect_timeout=5,
    )
    client.ping()   # raises if unreachable
    return client


def _ensure_redis_connected():
    """Ensure Redis connection is alive, reconnect if needed."""
    global _redis_client, _redis_last_error

    if _redis_client is None:
        try:
            _redis_client = _get_redis()
            _redis_last_error = None
            REDIS_CONNECTED.set(1)
            print(f"✅ Redis reconnected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as exc:
            _redis_last_error = exc
            REDIS_CONNECTED.set(0)
            print(f"⚠️  Redis reconnection failed: {exc}")
            raise

    # Test if connection is still alive
    try:
        _redis_client.ping()
        _redis_last_error = None
    except Exception as exc:
        # Connection dead, try to reconnect
        _redis_last_error = exc
        try:
            _redis_client = _get_redis()
            _redis_last_error = None
            REDIS_CONNECTED.set(1)
            print(f"✅ Redis reconnected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as reconnect_exc:
            _redis_last_error = reconnect_exc
            REDIS_CONNECTED.set(0)
            print(f"⚠️  Redis reconnection failed: {reconnect_exc}")
            raise


# ── Request / Response schemas ────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    user_id: str
    amount: float
    merchant_category: str
    merchant_country: str

    @validator('amount')
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('amount must be positive')
        if v > 100000:
            raise ValueError('amount exceeds maximum allowed value')
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError('amount must be a finite number')
        return v

    @validator('user_id')
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('user_id cannot be empty')
        return v.strip()

    @validator('merchant_category')
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('merchant_category cannot be empty')
        return v.strip()

    @validator('merchant_country')
    def validate_country(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('merchant_country cannot be empty')
        return v.strip()


class ScoreResponse(BaseModel):
    user_id: str
    fraud_probability: float
    risk_level: str
    features_used: dict
    redis_features_found: bool
    latency_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model_loaded    = _model is not None
    redis_connected = False
    if _redis_client:
        try:
            _redis_client.ping()
            redis_connected = True
        except Exception:
            pass

    # Update gauges for health check
    MODEL_LOADED.set(1 if model_loaded else 0)
    REDIS_CONNECTED.set(1 if redis_connected else 0)

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "redis_connected": redis_connected,
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint"""
    return prometheus_metrics()


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request, response: Response, api_key: str = Depends(get_api_key)):
    # Check rate limit and include headers in response
    rate_limit_headers = check_rate_limit(request, api_key)
    for header_name, header_value in rate_limit_headers.items():
        response.headers[header_name] = header_value

    # Track request count
    API_REQUESTS_TOTAL.labels(endpoint='/score', method='POST').inc()

    t_start = time.time()

    if _model is None:
        ERRORS_TOTAL.labels(error_type='prediction').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Track transaction amount distribution
    TRANSACTION_AMOUNT.observe(req.amount)

    # ── 1. Compute transaction-level flags using shared module
    # (these come from the current request payload, not the feature store)
    is_foreign     = compute_is_foreign(req.merchant_country)
    is_high_amount = compute_is_high_amount(req.amount)
    is_suspicious  = compute_is_suspicious(req.merchant_country, req.amount)

    # ── 2. Fetch windowed user features from Redis (live aggregates written
    # by the Spark streaming job under key `user_features:{user_id}`).
    redis_features_found = False
    txn_count_5min    = 0.0
    total_amount_5min = 0.0
    avg_amount_5min   = 0.0
    max_amount_5min   = 0.0

    t_redis_start = time.time()
    try:
        # Ensure Redis is connected (will reconnect if needed)
        _ensure_redis_connected()
        if _redis_client is not None:
            raw = _redis_client.hgetall(f"user_features:{req.user_id}")
            if raw:
                redis_features_found = True
                txn_count_5min    = float(raw.get("txn_count_5min",    0))
                total_amount_5min = float(raw.get("total_amount_5min", 0))
                avg_amount_5min   = float(raw.get("avg_amount_5min",   0))
                max_amount_5min   = float(raw.get("max_amount_5min", 0))
    except Exception as exc:
        print(f"⚠️  Redis lookup failed: {exc}")
        ERRORS_TOTAL.labels(error_type='redis_connection').inc()

    # Track Redis lookup time
    redis_latency = time.time() - t_redis_start
    REDIS_LOOKUP_TIME.observe(redis_latency)
    REDIS_LOOKUPS_TOTAL.labels(found=str(redis_features_found).lower()).inc()

    # ── 3. Build feature vector
    features = {
        "txn_count_5min":    txn_count_5min,
        "total_amount_5min": total_amount_5min,
        "avg_amount_5min":   avg_amount_5min,
        "max_amount_5min":   max_amount_5min,
        "is_foreign":        float(is_foreign),
        "is_high_amount":    float(is_high_amount),
        "is_suspicious":     float(is_suspicious),
    }
    X = np.array([[features[f] for f in FEATURE_ORDER]], dtype=float)

    # ── 4. Score (with inference time tracking)
    t_inference_start = time.time()
    fraud_prob = float(_model.predict_proba(X)[0][1])
    inference_latency = time.time() - t_inference_start
    MODEL_INFERENCE_TIME.observe(inference_latency)

    if fraud_prob > 0.7:
        risk_level = "HIGH"
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Track prediction by risk level
    PREDICTIONS_TOTAL.labels(risk_level=risk_level).inc()

    # Track last fraud probability (gauge)
    LAST_FRAUD_PROBABILITY.set(fraud_prob)

    # Track total request latency
    total_latency = time.time() - t_start
    REQUEST_LATENCY.labels(endpoint='/score').observe(total_latency)

    latency_ms = total_latency * 1000

    return ScoreResponse(
        user_id=req.user_id,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        features_used=features,
        redis_features_found=redis_features_found,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/user/{user_id}")
def get_user_features(user_id: str, request: Request, response: Response, api_key: str = Depends(get_api_key)):
    # Check rate limit and include headers in response
    rate_limit_headers = check_rate_limit(request, api_key)
    for header_name, header_value in rate_limit_headers.items():
        response.headers[header_name] = header_value

    # Track request
    API_REQUESTS_TOTAL.labels(endpoint='/user', method='GET').inc()

    # Ensure Redis is connected (will reconnect if needed)
    try:
        _ensure_redis_connected()
        data = _redis_client.hgetall(f"user_features:{user_id}")
    except Exception as exc:
        ERRORS_TOTAL.labels(error_type='redis_connection').inc()
        raise HTTPException(status_code=503, detail="Redis unavailable")
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for user '{user_id}' in Redis",
        )
    return {"user_id": user_id, "features": data}