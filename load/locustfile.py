"""
Locust alternative to k6 — same /score profile.
Usage:
  pip install -r load/requirements.txt
  locust -f load/locustfile.py --host http://localhost:8000
  # then open http://localhost:8089
  # headless: locust -f load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 --run-time 30s
"""

import os
import random

from locust import HttpUser, task, between

API_KEY = os.environ.get("API_KEYS", "dev-api-key-change-in-prod").split(",")[0].strip()

PAYLOADS = [
    {"user_id": "user_0001", "amount": 1500, "merchant_category": "electronics", "merchant_country": "NG"},
    {"user_id": "user_0042", "amount": 45, "merchant_category": "groceries", "merchant_country": "US"},
    {"user_id": "user_0123", "amount": 2500, "merchant_category": "travel", "merchant_country": "RU"},
    {"user_id": "user_0007", "amount": 499, "merchant_category": "restaurants", "merchant_country": "US"},
]


class FraudUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(5)
    def score(self):
        self.client.post(
            "/score",
            json=random.choice(PAYLOADS),
            headers={"X-API-Key": API_KEY},
            name="/score",
        )

    @task(1)
    def score_explain(self):
        self.client.post(
            "/score?explain=true",
            json=random.choice(PAYLOADS),
            headers={"X-API-Key": API_KEY},
            name="/score?explain=true",
        )

    @task(1)
    def health(self):
        self.client.get("/health", headers={"X-API-Key": API_KEY}, name="/health")
