import json
import time
import random
import os
from dotenv import load_dotenv
from kafka import KafkaProducer

# Load environment variables
load_dotenv()

from transaction_generator import generate_transaction, MERCHANT_CATEGORIES, COUNTRIES, USER_IDS

# ── Configuration ──────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# ── Connect to Kafka ──────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',          # wait for all in-sync replicas, not just the leader
    retries=5,           # retry transient failures (e.g. leader election)
    retry_backoff_ms=200,
    linger_ms=10,         # small batching window, negligible added latency
    request_timeout_ms=30000,
)

TOPIC = 'raw_transactions'

# ── Retry helper ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5  # seconds

def send_with_retry(producer, topic, transaction):
    """Send to Kafka with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            future = producer.send(topic, value=transaction)
            future.get(timeout=5)  # wait for ack
            return True
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  ⚠️  Send failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {backoff}s: {exc}")
                time.sleep(backoff)
            else:
                print(f"  ❌ Send failed after {MAX_RETRIES} attempts: {exc}")
                return False
    return False

# ── Main loop ────────────────────────────────────────────────
print("🚀 Starting transaction producer... Press Ctrl+C to stop\n")

tx_count = 0
success_count = 0
fail_count = 0

while True:
    user_id = random.choice(USER_IDS)
    transaction = generate_transaction(user_id)

    # IMPORTANT: Pop _is_fraud BEFORE sending to avoid race condition with async serialization
    # The send() captures a reference and serializes asynchronously in a background thread
    is_fraud = transaction.pop("_is_fraud", False)

    # Send with retry (don't block on waiting for ack in the main loop)
    future = producer.send(TOPIC, value=transaction)
    future.add_errback(lambda exc: print(f"  ⚠️  Async send error: {exc}"))

    tx_count += 1
    success_count += 1

    # Throttled logging (see Fix #9) — print first 10, then every 20th

    should_print = tx_count <= 10 or (tx_count % 20 == 0)
    if should_print:
        flag = "🚨 FRAUD" if is_fraud else "✅ legit"
        print(f"[{tx_count:05d}] {flag} | {user_id} | "
              f"${transaction['amount']:>8.2f} | "
              f"{transaction['merchant_category']:<15} | "
              f"{transaction['merchant_country']}")
    elif is_fraud:
        # Always print fraud alerts
        print(f"[{tx_count:05d}] 🚨 FRAUD | {user_id} | "
              f"${transaction['amount']:>8.2f} | "
              f"{transaction['merchant_category']:<15} | "
              f"{transaction['merchant_country']}")

    time.sleep(0.5)   # 2 transactions per second