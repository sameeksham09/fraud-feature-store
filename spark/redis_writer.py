import os
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# A "last 5 minutes" aggregate is meaningless once 5 minutes pass with no new
# activity, so the key expires instead of serving a stale window for hours.
# The API then sees no key and uses 0.0, which matches what training sees
# offline (Feast TTL in feast/features.py).
USER_FEATURES_TTL_SECONDS = 5 * 60
TXN_FLAGS_TTL_SECONDS = 60 * 60

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    decode_responses=True,
)


def write_user_features_to_redis(row):
    """
    Write user features to Redis atomically using pipeline.

    Uses hset with EX parameter for atomic write with TTL.
    Falls back to pipeline if EX not supported.
    """
    user_id = row['user_id']
    features = {
        "user_id": user_id,
        "txn_count_5min": str(row['txn_count_5min']),
        "total_amount_5min": str(round(row['total_amount_5min'], 2)),
        "avg_amount_5min": str(round(row['avg_amount_5min'], 2)),
        "max_amount_5min": str(round(row['max_amount_5min'], 2)),
        "window_start": str(row['window_start']),
        "window_end": str(row['window_end']),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    key = f"user_features:{user_id}"

    # Use pipeline for atomic write
    pipe = redis_client.pipeline()
    pipe.hset(key, mapping=features)
    pipe.expire(key, USER_FEATURES_TTL_SECONDS)
    pipe.execute()

    print(f"  → Redis write: {user_id} | txn_count={features['txn_count_5min']} | avg=${features['avg_amount_5min']}")


def write_transaction_flags_to_redis(row):
    """
    Write transaction flags to Redis atomically using pipeline.

    Uses pipeline for atomic write with TTL.
    """
    txn_id = row['transaction_id']
    flags = {
        "transaction_id": txn_id,
        "user_id": row['user_id'],
        "amount": str(row['amount']),
        "is_foreign": str(row['is_foreign']),
        "is_high_amount": str(row['is_high_amount']),
        "is_suspicious": str(row['is_suspicious']),
        "scored_at": datetime.now(timezone.utc).isoformat()
    }
    key = f"txn_flags:{txn_id}"

    # Use pipeline for atomic write
    pipe = redis_client.pipeline()
    pipe.hset(key, mapping=flags)
    pipe.expire(key, TXN_FLAGS_TTL_SECONDS)
    pipe.execute()
