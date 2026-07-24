"""
Unit tests for the shared features module.

These tests verify that the feature computation functions in features/__init__.py
work correctly and are used consistently across the codebase.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (
    compute_is_foreign,
    compute_is_high_amount,
    compute_is_suspicious,
    compute_all_transaction_flags,
    HIGH_AMOUNT_THRESHOLD,
    SUSPICIOUS_AMOUNT_THRESHOLD,
)


class TestIsForeign:
    """Tests for compute_is_foreign function."""

    def test_us_is_not_foreign(self):
        """US transactions should not be foreign."""
        assert compute_is_foreign("US") == 0

    def test_non_us_is_foreign(self):
        """Non-US transactions should be foreign."""
        for country in ["NG", "RU", "CN", "BR", "IN", "UK", "DE"]:
            assert compute_is_foreign(country) == 1, f"{country} should be foreign"

    def test_returns_integer(self):
        """Function should return integer 0 or 1."""
        result = compute_is_foreign("US")
        assert isinstance(result, int)
        assert result in [0, 1]


class TestIsHighAmount:
    """Tests for compute_is_high_amount function."""

    def test_below_threshold_is_not_high(self):
        """Amount below $500 should not be high."""
        for amount in [1, 100, 499, 500]:
            assert compute_is_high_amount(amount) == 0

    def test_above_threshold_is_high(self):
        """Amount above $500 should be high."""
        for amount in [501, 1000, 5000, 99999]:
            assert compute_is_high_amount(amount) == 1

    def test_threshold_values(self):
        """Verify the threshold is 500."""
        assert HIGH_AMOUNT_THRESHOLD == 500


class TestIsSuspicious:
    """Tests for compute_is_suspicious function."""

    def test_domestic_not_suspicious(self):
        """Domestic transactions should never be suspicious."""
        for amount in [1, 200, 500, 1000, 10000]:
            assert compute_is_suspicious("US", amount) == 0

    def test_foreign_below_threshold_not_suspicious(self):
        """Foreign transactions below $200 should not be suspicious."""
        for amount in [1, 100, 199, 200]:
            assert compute_is_suspicious("CN", amount) == 0

    def test_foreign_above_threshold_is_suspicious(self):
        """Foreign transactions above $200 should be suspicious."""
        for amount in [201, 500, 1000, 5000]:
            assert compute_is_suspicious("NG", amount) == 1

    def test_threshold_values(self):
        """Verify the suspicious threshold is 200."""
        assert SUSPICIOUS_AMOUNT_THRESHOLD == 200

    def test_boundary_at_threshold(self):
        """Amount exactly at threshold ($200) should NOT be suspicious."""
        assert compute_is_suspicious("CN", 200) == 0


class TestComputeAllTransactionFlags:
    """Tests for compute_all_transaction_flags function."""

    def test_returns_all_flags(self):
        """Should return dict with all three flags."""
        result = compute_all_transaction_flags("US", 100)
        assert "is_foreign" in result
        assert "is_high_amount" in result
        assert "is_suspicious" in result

    def test_us_transaction(self):
        """US transaction should have is_foreign=0."""
        result = compute_all_transaction_flags("US", 100)
        assert result["is_foreign"] == 0
        assert result["is_suspicious"] == 0

    def test_foreign_high_amount(self):
        """Foreign + high amount should be suspicious."""
        result = compute_all_transaction_flags("CN", 600)
        assert result["is_foreign"] == 1
        assert result["is_high_amount"] == 1
        assert result["is_suspicious"] == 1

    def test_foreign_low_amount(self):
        """Foreign + low amount should not be suspicious."""
        result = compute_all_transaction_flags("BR", 50)
        assert result["is_foreign"] == 1
        assert result["is_high_amount"] == 0
        assert result["is_suspicious"] == 0


class TestThresholdConsistency:
    """Tests to verify thresholds are consistent."""

    def test_suspicious_threshold_lower_than_high_amount(self):
        """Suspicious threshold ($200) should be lower than high amount ($500)."""
        assert SUSPICIOUS_AMOUNT_THRESHOLD < HIGH_AMOUNT_THRESHOLD

    def test_thresholds_are_positive(self):
        """Both thresholds should be positive."""
        assert HIGH_AMOUNT_THRESHOLD > 0
        assert SUSPICIOUS_AMOUNT_THRESHOLD > 0
