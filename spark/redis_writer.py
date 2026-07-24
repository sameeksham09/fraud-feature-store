import os
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

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
    pipe.expire(key, 86400)  # 24 hours TTL
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
    pipe.expire(key, 3600)  # 1 hour TTL
    pipe.execute()
