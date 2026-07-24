"""
Unit tests for data validation schemas.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.validation import (
    validate_transaction,
    validate_user_features,
    validate_transaction_flags,
    ValidationError,
    VALID_MERCHANT_CATEGORIES,
    VALID_COUNTRIES,
)


class TestTransactionValidation:
    """Tests for transaction schema validation."""

    def test_valid_transaction(self):
        """Valid transaction should pass validation."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": 100.50,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "merchant_name": "Walmart",
            "card_last4": "1234",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert is_valid, f"Expected valid, got errors: {errors}"

    def test_invalid_user_id_format(self):
        """Invalid user_id format should fail."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "invalid",
            "amount": 100.50,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid
        assert any("user_id" in e.lower() for e in errors)

    def test_negative_amount_fails(self):
        """Negative amount should fail validation."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": -100.50,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid
        assert any("amount" in e.lower() for e in errors)

    def test_zero_amount_fails(self):
        """Zero amount should fail validation."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": 0,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid

    def test_excessive_amount_fails(self):
        """Amount >= 100000 should fail validation."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": 100000,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid

    def test_invalid_merchant_category(self):
        """Invalid merchant category should fail."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": 100.50,
            "merchant_category": "invalid_category",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid

    def test_invalid_country(self):
        """Invalid country code should fail."""
        txn = {
            "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "user_id": "user_0001",
            "amount": 100.50,
            "merchant_category": "groceries",
            "merchant_country": "INVALID",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid

    def test_missing_required_fields(self):
        """Missing required fields should fail."""
        txn = {
            "user_id": "user_0001",
            # Missing transaction_id, amount, etc.
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_transaction_id_format(self):
        """Invalid transaction_id format should fail."""
        txn = {
            "transaction_id": "not-a-uuid",
            "user_id": "user_0001",
            "amount": 100.50,
            "merchant_category": "groceries",
            "merchant_country": "US",
            "timestamp": "2024-01-01T10:00:00",
        }

        is_valid, errors = validate_transaction(txn)
        assert not is_valid
        assert any("transaction_id" in e.lower() for e in errors)


class TestUserFeaturesValidation:
    """Tests for user features schema validation."""

    def test_valid_user_features(self):
        """Valid user features should pass."""
        features = {
            "user_id": "user_0150",
            "txn_count_5min": 5,
            "total_amount_5min": 500.00,
            "avg_amount_5min": 100.00,
            "max_amount_5min": 200.00,
        }

        is_valid, errors = validate_user_features(features)
        assert is_valid, f"Expected valid, got errors: {errors}"

    def test_invalid_user_id(self):
        """Invalid user_id should fail."""
        features = {
            "user_id": "bad_id",
            "txn_count_5min": 5,
            "total_amount_5min": 500.00,
            "avg_amount_5min": 100.00,
            "max_amount_5min": 200.00,
        }

        is_valid, errors = validate_user_features(features)
        assert not is_valid

    def test_negative_count_fails(self):
        """Negative transaction count should fail."""
        features = {
            "user_id": "user_0150",
            "txn_count_5min": -1,
            "total_amount_5min": 500.00,
            "avg_amount_5min": 100.00,
            "max_amount_5min": 200.00,
        }

        is_valid, errors = validate_user_features(features)
        assert not is_valid

    def test_negative_amount_fails(self):
        """Negative amounts should fail."""
        features = {
            "user_id": "user_0150",
            "txn_count_5min": 5,
            "total_amount_5min": -500.00,
            "avg_amount_5min": 100.00,
            "max_amount_5min": 200.00,
        }

        is_valid, errors = validate_user_features(features)
        assert not is_valid


class TestTransactionFlagsValidation:
    """Tests for transaction flags validation."""

    def test_valid_flags(self):
        """Valid flags should pass."""
        flags = {
            "transaction_id": "txn-12345",
            "user_id": "user_0001",
            "amount": 100.0,
            "is_foreign": True,
            "is_high_amount": False,
            "is_suspicious": True,
        }

        is_valid, errors = validate_transaction_flags(flags)
        assert is_valid, f"Expected valid, got errors: {errors}"

    def test_non_boolean_fails(self):
        """Non-boolean flags should fail."""
        flags = {
            "transaction_id": "txn-12345",
            "user_id": "user_0001",
            "amount": 100.0,
            "is_foreign": "yes",  # Should be bool
            "is_high_amount": False,
            "is_suspicious": True,
        }

        is_valid, errors = validate_transaction_flags(flags)
        assert not is_valid


class TestAllowedValues:
    """Tests for allowed values in schemas."""

    def test_all_merchant_categories(self):
        """All valid merchant categories should pass."""
        for category in VALID_MERCHANT_CATEGORIES:
            txn = {
                "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "user_id": "user_0001",
                "amount": 100.50,
                "merchant_category": category,
                "merchant_country": "US",
                "timestamp": "2024-01-01T10:00:00",
            }
            is_valid, errors = validate_transaction(txn)
            assert is_valid, f"Category {category} should be valid: {errors}"

    def test_all_countries(self):
        """All valid countries should pass."""
        for country in VALID_COUNTRIES:
            txn = {
                "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "user_id": "user_0001",
                "amount": 100.50,
                "merchant_category": "groceries",
                "merchant_country": country,
                "timestamp": "2024-01-01T10:00:00",
            }
            is_valid, errors = validate_transaction(txn)
            assert is_valid, f"Country {country} should be valid: {errors}"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_stores_errors(self):
        """ValidationError should store error list."""
        errors = ["error1", "error2"]
        exc = ValidationError(errors)
        assert exc.errors == errors

    def test_validation_error_message(self):
        """ValidationError should have readable message."""
        errors = ["error1", "error2"]
        exc = ValidationError(errors)
        assert "error1" in str(exc)
        assert "error2" in str(exc)

    def test_raise_on_error_flag(self):
        """validate_transaction should raise when raise_on_error=True."""
        txn = {"user_id": "invalid"}  # Invalid - missing required fields

        with pytest.raises(ValidationError):
            validate_transaction(txn, raise_on_error=True)
