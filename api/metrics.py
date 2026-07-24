"""
Prometheus metrics for the Fraud Detection API.

Metric Types Explained:
- Counter: Only goes up (e.g., total requests). Use for things that accumulate.
- Histogram: Measures distribution (e.g., latency). Buckets capture percentiles.
- Gauge: Can go up and down (e.g., current temperature, queue size).
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# ─────────────────────────────────────────────────────────────────────────────
# COUNTERS - Things that only increase
# ─────────────────────────────────────────────────────────────────────────────

# Total API requests (by endpoint)
API_REQUESTS_TOTAL = Counter(
    'fraud_api_requests_total',
    'Total API requests',
    ['endpoint', 'method']  # Labels: allows slicing by endpoint
)

# Total predictions (by risk level)
PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['risk_level']  # Labels: HIGH, MEDIUM, LOW
)

# Feature store lookups
REDIS_LOOKUPS_TOTAL = Counter(
    'fraud_redis_lookups_total',
    'Total Redis feature lookups',
    ['found']  # Labels: 'true' or 'false'
)

# Errors
ERRORS_TOTAL = Counter(
    'fraud_errors_total',
    'Total errors',
    ['error_type']  # Labels: 'model_load', 'redis_connection', 'prediction'
)

# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAMS - Measure distributions (latency, sizes, etc.)
# ─────────────────────────────────────────────────────────────────────────────

# API request latency (seconds)
REQUEST_LATENCY = Histogram(
    'fraud_api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]  # ms buckets
)

# Model inference time
MODEL_INFERENCE_TIME = Histogram(
    'fraud_model_inference_seconds',
    'Model inference time in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]  # typical inference is <10ms
)

# Feature retrieval time from Redis
REDIS_LOOKUP_TIME = Histogram(
    'fraud_redis_lookup_seconds',
    'Redis feature lookup time in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

# Transaction amount (for monitoring input distribution)
TRANSACTION_AMOUNT = Histogram(
    'fraud_transaction_amount',
    'Transaction amount in USD',
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

# ─────────────────────────────────────────────────────────────────────────────
# GAUGES - Values that go up and down
# ─────────────────────────────────────────────────────────────────────────────

# Model status (1 = loaded, 0 = not loaded)
MODEL_LOADED = Gauge(
    'fraud_model_loaded',
    'Model loaded status (1=loaded, 0=not loaded)'
)

# Redis connection status (1 = connected, 0 = not connected)
REDIS_CONNECTED = Gauge(
    'fraud_redis_connected',
    'Redis connection status (1=connected, 0=not connected)'
)

# Current fraud probability (last prediction)
LAST_FRAUD_PROBABILITY = Gauge(
    'fraud_last_prediction_probability',
    'Last fraud probability prediction'
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper endpoint
# ─────────────────────────────────────────────────────────────────────────────

def metrics():
    """
    Prometheus endpoint - scrapes all metrics.
    Prometheus server hits this endpoint to collect metrics.
    """
    return Response(generate_latest(), media_type="text/plain")