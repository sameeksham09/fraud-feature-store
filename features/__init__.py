"""
Shared feature computation module.

This module provides a single source of truth for computing transaction-level
features (is_foreign, is_high_amount, is_suspicious) used by:
- Spark streaming job (spark/feature_engineering.py)
- FastAPI scoring service (api/main.py)
- Training pipeline (models/train.py)

IMPORTANT: Keep these thresholds synchronized across all consumers.
"""

# ── Feature Thresholds ─────────────────────────────────────────────────────────
# These values must be consistent across Spark, API, and training.

SUSPICIOUS_AMOUNT_THRESHOLD = 200  # Amount above which foreign transactions are suspicious
HIGH_AMOUNT_THRESHOLD = 500  # Amount above which a transaction is considered "high value"


# ── Feature Computation Functions ─────────────────────────────────────────────

def compute_is_foreign(merchant_country: str) -> int:
    """Check if a transaction is foreign (merchant not in US)."""
    return 1 if merchant_country != "US" else 0


def compute_is_high_amount(amount: float) -> int:
    """Check if transaction amount exceeds the high amount threshold."""
    return 1 if amount > HIGH_AMOUNT_THRESHOLD else 0


def compute_is_suspicious(merchant_country: str, amount: float) -> int:
    """
    Check if a transaction is suspicious.

    A transaction is suspicious if:
    - The merchant is foreign (not US), AND
    - The amount exceeds the suspicious threshold
    """
    is_foreign = merchant_country != "US"
    is_high = amount > SUSPICIOUS_AMOUNT_THRESHOLD
    return 1 if (is_foreign and is_high) else 0


def compute_all_transaction_flags(merchant_country: str, amount: float) -> dict:
    """
    Compute all transaction-level flags at once.

    Returns:
        dict with keys: is_foreign, is_high_amount, is_suspicious
    """
    return {
        "is_foreign": compute_is_foreign(merchant_country),
        "is_high_amount": compute_is_high_amount(amount),
        "is_suspicious": compute_is_suspicious(merchant_country, amount),
    }


# ── Model feature vector (the training/serving contract) ──────────────────────
# Training (models/train.py) and serving (api/main.py) MUST both build the model
# input through build_feature_vector(); never re-derive features inline.
#
#   * Windowed aggregates describe the user's trailing ~5 minutes and come from
#     the feature store (Redis online, Parquet/Feast offline).
#   * Transaction flags describe the transaction being scored and are computed
#     from that transaction's own amount and merchant country, never from
#     aggregates like max_amount_5min or avg_amount_5min.

WINDOWED_FEATURES = [
    "txn_count_5min",
    "total_amount_5min",
    "avg_amount_5min",
    "max_amount_5min",
]

TRANSACTION_FEATURES = [
    "is_foreign",
    "is_high_amount",
    "is_suspicious",
]

FEATURE_ORDER = WINDOWED_FEATURES + TRANSACTION_FEATURES


def build_feature_vector(user_features, amount: float, merchant_country: str) -> dict:
    """
    Build the model's input features for one transaction.

    Args:
        user_features: mapping with the windowed aggregates (missing keys,
            None or NaN default to 0.0, e.g. a user with no recent activity).
        amount: amount of the transaction being scored.
        merchant_country: merchant country of the transaction being scored.

    Returns:
        dict keyed by FEATURE_ORDER, all values float.
    """
    user_features = user_features or {}
    vector = {}
    for name in WINDOWED_FEATURES:
        value = user_features.get(name)
        # None (missing in Redis) or NaN (no match in an offline join) -> 0.0
        vector[name] = 0.0 if value is None or value != value else float(value)
    flags = compute_all_transaction_flags(merchant_country, amount)
    for name in TRANSACTION_FEATURES:
        vector[name] = float(flags[name])
    return vector
