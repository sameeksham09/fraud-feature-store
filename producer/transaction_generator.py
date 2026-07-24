"""
Transaction generation logic - separated from Kafka producer for testability.
"""

import random
import uuid
from datetime import datetime
from faker import Faker

fake = Faker()

# ── Realistic merchant categories ────────────────────────────
MERCHANT_CATEGORIES = [
    'groceries', 'electronics', 'restaurants', 'travel',
    'gas_station', 'pharmacy', 'clothing', 'entertainment'
]

# ── Countries (mostly US, occasionally foreign = suspicious) ──
COUNTRIES = ['US'] * 85 + ['NG', 'RU', 'CN', 'BR', 'IN'] * 3

# ── A small pool of users (so we get repeat transactions) ─────
USER_IDS = [f"user_{i:04d}" for i in range(1, 201)]  # 200 users


def generate_transaction(user_id: str) -> dict:
    """Generate one realistic-ish transaction for a user."""
    # In production, fraud labels come from delayed chargeback reports,
    # not from the real-time stream. We simulate that here by NOT including
    # is_fraud in the payload. The label is kept locally only for the
    # producer's logging (to show fraud vs legit in stdout).
    _is_fraud = random.random() < 0.03   # ~3% fraud rate (realistic)

    if _is_fraud:
        # Fraud pattern: high amount, foreign country, unusual category
        amount = round(random.uniform(500, 5000), 2)
        country = random.choice(['NG', 'RU', 'CN'])
        category = random.choice(['electronics', 'travel'])
    else:
        amount = round(random.uniform(1, 400), 2)
        country = 'US'
        category = random.choice(MERCHANT_CATEGORIES)

    # Store locally only for logging purposes
    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": amount,
        "merchant_category": category,
        "merchant_country": country,
        "merchant_name": fake.company(),
        "card_last4": str(random.randint(1000, 9999)),
        "timestamp": datetime.utcnow().isoformat(),
    }
    # Attach label for local logging only (not sent to Kafka)
    transaction["_is_fraud"] = _is_fraud
    return transaction
