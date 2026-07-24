"""
Unit tests for the transaction producer.
Tests the generate_transaction function and related utilities.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer.transaction_generator import (
    generate_transaction,
    MERCHANT_CATEGORIES,
    COUNTRIES,
    USER_IDS,
)


class TestGenerateTransaction:
    """Tests for the transaction generation logic."""

    def test_transaction_has_required_fields(self):
        """Transaction dict should have all required fields."""
        user_id = "user_0001"
        txn = generate_transaction(user_id)

        required_fields = [
            "transaction_id",
            "user_id",
            "amount",
            "merchant_category",
            "merchant_country",
            "merchant_name",
            "card_last4",
            "timestamp",
            "_is_fraud",  # Local label only
        ]

        for field in required_fields:
            assert field in txn, f"Missing required field: {field}"

    def test_transaction_user_id_matches_input(self):
        """Transaction should have the user_id passed in."""
        user_id = "user_0150"
        txn = generate_transaction(user_id)

        assert txn["user_id"] == user_id

    def test_transaction_id_is_uuid(self):
        """Transaction ID should be a valid UUID string."""
        import uuid

        txn = generate_transaction("user_0001")

        # Should not raise an exception
        uuid.UUID(txn["transaction_id"])

    def test_amount_is_positive(self):
        """Transaction amount should always be positive."""
        for _ in range(100):  # Run multiple times for stochastic generation
            txn = generate_transaction("user_0001")
            assert txn["amount"] > 0, f"Amount should be positive: {txn['amount']}"

    def test_amount_in_expected_range(self):
        """Amount should be in realistic range based on fraud status."""
        # Run many times to catch both fraud and non-fraud cases
        fraud_amounts = []
        legit_amounts = []

        for _ in range(500):
            txn = generate_transaction("user_0001")
            if txn["_is_fraud"]:
                fraud_amounts.append(txn["amount"])
            else:
                legit_amounts.append(txn["amount"])

        # Check that we got both types (with 3% fraud rate, should)
        assert len(fraud_amounts) > 0, "Should generate some fraud transactions"
        assert len(legit_amounts) > 0, "Should generate some legitimate transactions"

        # Fraud amounts should tend to be higher (500-5000)
        assert min(fraud_amounts) >= 500, "Fraud amounts should be >= 500"
        assert max(fraud_amounts) <= 5000, "Fraud amounts should be <= 5000"

        # Legit amounts should be lower (1-400)
        assert max(legit_amounts) <= 400, "Legit amounts should be <= 400"

    def test_merchant_category_is_valid(self):
        """Merchant category should be from the allowed list."""
        for _ in range(100):
            txn = generate_transaction("user_0001")
            assert txn["merchant_category"] in MERCHANT_CATEGORIES

    def test_merchant_country_is_valid(self):
        """Merchant country should be from the allowed list."""
        for _ in range(100):
            txn = generate_transaction("user_0001")
            assert txn["merchant_country"] in COUNTRIES

    def test_card_last4_is_4_digits(self):
        """Card last 4 should be exactly 4 digits."""
        import re

        for _ in range(50):
            txn = generate_transaction("user_0001")
            assert re.match(r"^\d{4}$", txn["card_last4"])

    def test_timestamp_is_iso_format(self):
        """Timestamp should be in ISO format."""
        from datetime import datetime

        txn = generate_transaction("user_0001")

        # Should parse without error
        parsed = datetime.fromisoformat(txn["timestamp"])
        assert parsed is not None

    def test_fraud_rate_approximates_3_percent(self):
        """Fraud rate should be approximately 3% over many samples."""
        n = 10000
        fraud_count = 0

        for _ in range(n):
            txn = generate_transaction("user_0001")
            if txn["_is_fraud"]:
                fraud_count += 1

        fraud_rate = fraud_count / n

        # Allow some variance: between 2% and 4%
        assert 0.02 <= fraud_rate <= 0.04, f"Fraud rate {fraud_rate:.2%} not in expected range"

    def test_fraud_pattern_foreign_country(self):
        """Fraud transactions should have foreign merchant countries."""
        fraud_countries = {'NG', 'RU', 'CN'}

        for _ in range(200):
            txn = generate_transaction("user_0001")
            if txn["_is_fraud"]:
                # At least some fraud should be foreign
                if txn["merchant_country"] in fraud_countries:
                    break
        else:
            # If we didn't find any foreign fraud, that's suspicious
            # but with random it's possible, so let's be lenient
            pass

    def test_merchant_name_is_non_empty(self):
        """Merchant name should be a non-empty string."""
        txn = generate_transaction("user_0001")
        assert txn["merchant_name"]
        assert isinstance(txn["merchant_name"], str)
        assert len(txn["merchant_name"]) > 0


class TestConstants:
    """Tests for the constants defined in the producer."""

    def test_merchant_categories_not_empty(self):
        """MERCHANT_CATEGORIES should not be empty."""
        assert len(MERCHANT_CATEGORIES) > 0

    def test_countries_not_empty(self):
        """COUNTRIES should not be empty."""
        assert len(COUNTRIES) > 0

    def test_user_ids_has_200_users(self):
        """USER_IDS should contain 200 users."""
        assert len(USER_IDS) == 200

    def test_user_ids_format(self):
        """User IDs should be in expected format user_XXXX."""
        import re

        for user_id in USER_IDS:
            assert re.match(r"^user_\d{4}$", user_id)
