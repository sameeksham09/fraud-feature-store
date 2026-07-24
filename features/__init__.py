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
