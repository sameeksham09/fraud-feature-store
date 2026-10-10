"""
Quarantine sink for invalid records.

Invalid rows that fail validation in foreachBatch are written here instead of
being silently dropped. The helper is intentionally thin and best-effort:
any failure to quarantine must NOT kill the streaming query.

Sink order:
  1. Try S3/MinIO JSON at s3://features/quarantine/invalid_records/ via s3fs/boto3
  2. Fall back to structured log + local file under /tmp/quarantine/
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Lightweight in-process counter (visible in logs / for tests)
_quarantine_count: int = 0

# Optional prometheus counter — use if available on the driver
try:
    from prometheus_client import Counter as _Counter  # type: ignore

    QUARANTINED_TOTAL = _Counter(
        "fraud_quarantined_total",
        "Total invalid records quarantined",
        ["kind"],  # user_features | transaction_flags
    )
except Exception:  # prometheus_client not installed on Spark driver
    QUARANTINED_TOTAL = None  # type: ignore


def _s3_write(payload: Dict[str, Any]) -> bool:
    """Best-effort write to MinIO via s3fs or boto3. Returns True on success."""
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
    access = os.environ.get("MINIO_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"))
    secret = os.environ.get("MINIO_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    # s3://features is the bucket; quarantine/ is just a prefix — no extra bucket needed.
    key = f"quarantine/invalid_records/{payload.get('kind','unknown')}/{int(time.time()*1000)}_{os.getpid()}.json"

    # Try s3fs first (already a dep via feast/s3fs)
    try:
        import s3fs  # type: ignore

        fs = s3fs.S3FileSystem(
            key=access,
            secret=secret,
            client_kwargs={"endpoint_url": endpoint},
        )
        path = f"features/{key}"
        with fs.open(path, "w") as f:
            json.dump(payload, f, default=str)
        logger.warning("Quarantined %s -> s3://features/%s", payload.get("kind"), key)
        return True
    except Exception as e:
        logger.debug("s3fs quarantine write failed: %s", e)

    # Try boto3 as fallback
    try:
        import boto3  # type: ignore

        # endpoint like http://minio:9000 or http://localhost:9000
        import botocore.client  # noqa: F401

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
        )
        s3.put_object(
            Bucket="features",
            Key=key,
            Body=json.dumps(payload, default=str).encode(),
            ContentType="application/json",
        )
        logger.warning("Quarantined %s -> s3://features/%s (boto3)", payload.get("kind"), key)
        return True
    except Exception as e:
        logger.debug("boto3 quarantine write failed: %s", e)

    return False


def write_to_quarantine(
    row_dict: Dict[str, Any],
    errors: List[str],
    *,
    kind: str,
    batch_id: int | None = None,
) -> None:
    """
    Persist an invalid record to the quarantine sink.

    Args:
        row_dict: the original row as dict
        errors: validation errors
        kind: "user_features" or "transaction_flags" (used for prefix + metric label)
        batch_id: Spark batch id if known
    """
    global _quarantine_count
    _quarantine_count += 1

    if QUARANTINED_TOTAL is not None:
        try:
            QUARANTINED_TOTAL.labels(kind=kind).inc()
        except Exception:
            pass

    payload = {
        "kind": kind,
        "batch_id": batch_id,
        "errors": errors,
        "record": row_dict,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
    }

    # Never let quarantine I/O kill the streaming query
    try:
        ok = _s3_write(payload)
        if ok:
            return
    except Exception as exc:
        logger.debug("Quarantine S3 write threw: %s", exc)

    # Fallback: structured log + local file
    logger.warning("QUARANTINE kind=%s batch=%s errors=%s record=%s", kind, batch_id, errors, row_dict)
    try:
        fallback_dir = "/tmp/quarantine/invalid_records"
        os.makedirs(fallback_dir, exist_ok=True)
        fname = os.path.join(fallback_dir, f"{kind}_{int(time.time()*1000)}_{os.getpid()}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, default=str)
    except Exception as exc:
        logger.debug("Quarantine fallback file write failed: %s", exc)


def get_quarantine_count() -> int:
    return _quarantine_count


def reset_quarantine_count() -> None:
    global _quarantine_count
    _quarantine_count = 0
