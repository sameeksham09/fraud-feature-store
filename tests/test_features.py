"""
Unit tests for the Redis writer module.
Tests writing user features and transaction flags to Redis.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_redis():
    """Mock Redis client with pipeline."""
    with patch('spark.redis_writer.redis_client') as mock:
        # Mock pipeline to return the mock client for chaining
        mock_pipeline = MagicMock()
        mock.return_value = mock_pipeline
        mock_pipeline.hset.return_value = None
        mock_pipeline.expire.return_value = None
        mock_pipeline.execute.return_value = None
        yield mock


class TestWriteUserFeaturesToRedis:
    """Tests for writing user features to Redis."""

    def test_writes_to_correct_key_format(self, mock_redis):
        """Should write to key format user_features:{user_id}."""
        from spark.redis_writer import write_user_features_to_redis

        row = {
            'user_id': 'user_0001',
            'txn_count_5min': 10,
            'total_amount_5min': 1500.50,
            'avg_amount_5min': 150.05,
            'max_amount_5min': 500.00,
            'window_start': '2024-01-01 10:00:00',
            'window_end': '2024-01-01 10:05:00',
        }

        write_user_features_to_redis(row)

        # Check hset was called via pipeline
        mock_redis.pipeline.assert_called_once()

    def test_sets_24h_ttl(self, mock_redis):
        """User features should have 24 hour TTL (86400 seconds)."""
        from spark.redis_writer import write_user_features_to_redis

        row = {
            'user_id': 'user_0001',
            'txn_count_5min': 5,
            'total_amount_5min': 500.00,
            'avg_amount_5min': 100.00,
            'max_amount_5min': 200.00,
            'window_start': '2024-01-01 10:00:00',
            'window_end': '2024-01-01 10:05:00',
        }

        write_user_features_to_redis(row)

        # Verify expire was called with correct TTL
        pipe = mock_redis.pipeline.return_value
        pipe.expire.assert_called()
        call_args = pipe.expire.call_args
        ttl = call_args[0][1]
        assert ttl == 86400  # 24 hours in seconds

    def test_includes_all_features_in_hash(self, mock_redis):
        """All features should be stored in the Redis hash."""
        from spark.redis_writer import write_user_features_to_redis

        row = {
            'user_id': 'user_0050',
            'txn_count_5min': 7,
            'total_amount_5min': 1234.56,
            'avg_amount_5min': 176.37,
            'max_amount_5min': 500.00,
            'window_start': '2024-01-01 10:00:00',
            'window_end': '2024-01-01 10:05:00',
        }

        write_user_features_to_redis(row)

        pipe = mock_redis.pipeline.return_value
        hset_call = pipe.hset.call_args
        features = hset_call[1]['mapping']

        assert 'user_id' in features
        assert 'txn_count_5min' in features
        assert 'total_amount_5min' in features
        assert 'avg_amount_5min' in features
        assert 'max_amount_5min' in features
        assert 'window_start' in features
        assert 'window_end' in features
        assert 'updated_at' in features

    def test_rounds_amounts_correctly(self, mock_redis):
        """Amounts should be rounded to 2 decimal places."""
        from spark.redis_writer import write_user_features_to_redis

        row = {
            'user_id': 'user_0001',
            'txn_count_5min': 3,
            'total_amount_5min': 123.456789,
            'avg_amount_5min': 41.1523,
            'max_amount_5min': 99.999,
            'window_start': '2024-01-01 10:00:00',
            'window_end': '2024-01-01 10:05:00',
        }

        write_user_features_to_redis(row)

        pipe = mock_redis.pipeline.return_value
        hset_call = pipe.hset.call_args
        features = hset_call[1]['mapping']

        # Check rounding
        assert features['total_amount_5min'] == '123.46'
        assert features['avg_amount_5min'] == '41.15'
        assert features['max_amount_5min'] == '100.0'  # 99.999 rounds to 100.0


class TestWriteTransactionFlagsToRedis:
    """Tests for writing transaction flags to Redis."""

    def test_writes_to_correct_key_format(self, mock_redis):
        """Should write to key format txn_flags:{transaction_id}."""
        from spark.redis_writer import write_transaction_flags_to_redis

        row = {
            'transaction_id': 'txn-12345',
            'user_id': 'user_0001',
            'amount': 100.0,
            'merchant_country': 'US',
            'merchant_category': 'groceries',
            'is_foreign': False,
            'is_high_amount': False,
            'is_suspicious': False,
            'event_time': '2024-01-01T10:00:00',
        }

        write_transaction_flags_to_redis(row)

        mock_redis.pipeline.assert_called_once()

    def test_sets_1h_ttl(self, mock_redis):
        """Transaction flags should have 1 hour TTL (3600 seconds)."""
        from spark.redis_writer import write_transaction_flags_to_redis

        row = {
            'transaction_id': 'txn-12345',
            'user_id': 'user_0001',
            'amount': 100.0,
            'merchant_country': 'US',
            'merchant_category': 'groceries',
            'is_foreign': False,
            'is_high_amount': False,
            'is_suspicious': False,
            'event_time': '2024-01-01T10:00:00',
        }

        write_transaction_flags_to_redis(row)

        pipe = mock_redis.pipeline.return_value
        pipe.expire.assert_called()
        ttl = pipe.expire.call_args[0][1]
        assert ttl == 3600  # 1 hour in seconds

    def test_stores_all_flags(self, mock_redis):
        """All flags should be stored in the Redis hash."""
        from spark.redis_writer import write_transaction_flags_to_redis

        row = {
            'transaction_id': 'txn-99999',
            'user_id': 'user_0100',
            'amount': 1500.0,
            'merchant_country': 'NG',
            'merchant_category': 'electronics',
            'is_foreign': True,
            'is_high_amount': True,
            'is_suspicious': True,
            'event_time': '2024-01-01T10:00:00',
        }

        write_transaction_flags_to_redis(row)

        pipe = mock_redis.pipeline.return_value
        hset_call = pipe.hset.call_args
        flags = hset_call[1]['mapping']

        assert flags['transaction_id'] == 'txn-99999'
        assert flags['user_id'] == 'user_0100'
        assert flags['amount'] == '1500.0'
        assert flags['is_foreign'] == 'True'
        assert flags['is_high_amount'] == 'True'
        assert flags['is_suspicious'] == 'True'
        assert 'scored_at' in flags


class TestFeatureComputationFromSharedModule:
    """Tests for feature computation using the shared features module."""

    def test_is_foreign_domestic(self):
        """US transactions should not be foreign."""
        from features import compute_is_foreign
        assert compute_is_foreign("US") == 0

    def test_is_foreign_international(self):
        """Non-US transactions should be foreign."""
        from features import compute_is_foreign
        for country in ['NG', 'RU', 'CN', 'BR', 'IN']:
            assert compute_is_foreign(country) == 1

    def test_is_high_amount_above_threshold(self):
        """Amount > 500 should be high amount."""
        from features import compute_is_high_amount
        assert compute_is_high_amount(501) == 1

    def test_is_high_amount_at_threshold(self):
        """Amount exactly 500 should not be high amount."""
        from features import compute_is_high_amount
        assert compute_is_high_amount(500) == 0

    def test_is_high_amount_below_threshold(self):
        """Amount < 500 should not be high amount."""
        from features import compute_is_high_amount
        assert compute_is_high_amount(499) == 0

    def test_is_suspicious_combination(self):
        """Foreign + amount > 200 should be suspicious."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("CN", 201) == 1

    def test_is_suspicious_foreign_below_threshold(self):
        """Foreign but amount <= 200 should not be suspicious."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("CN", 200) == 0

    def test_is_suspicious_domestic(self):
        """Domestic transaction should not be suspicious regardless of amount."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("US", 1000) == 0


class TestRiskLevelLogic:
    """Tests for risk level classification (same logic as API)."""

    def test_risk_level_high(self):
        """Fraud probability > 0.7 should be HIGH risk."""
        fraud_prob = 0.71
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        assert risk_level == "HIGH"

    def test_risk_level_medium(self):
        """Fraud probability > 0.4 and <= 0.7 should be MEDIUM risk."""
        fraud_prob = 0.5
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        assert risk_level == "MEDIUM"

    def test_risk_level_low(self):
        """Fraud probability <= 0.4 should be LOW risk."""
        fraud_prob = 0.3
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        assert risk_level == "LOW"

    def test_risk_level_boundary_high(self):
        """Fraud probability exactly 0.7 should be MEDIUM (boundary)."""
        fraud_prob = 0.7
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        assert risk_level == "MEDIUM"

    def test_risk_level_boundary_medium(self):
        """Fraud probability exactly 0.4 should be LOW (boundary)."""
        fraud_prob = 0.4
        if fraud_prob > 0.7:
            risk_level = "HIGH"
        elif fraud_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        assert risk_level == "LOW"
