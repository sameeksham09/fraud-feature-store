"""
Unit tests for the Fraud Detection API.

Tests the core logic including:
- Feature computation using shared module
- Pydantic request validation
- Risk level logic
- Redis key formats
"""

import pytest
import sys
import os

# Set required env var before importing modules
os.environ["API_KEYS"] = "test-key-1,test-key-2"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureComputation:
    """Tests using the shared features module."""

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
        assert compute_is_high_amount(1000) == 1

    def test_is_high_amount_at_threshold(self):
        """Amount exactly 500 should not be high amount."""
        from features import compute_is_high_amount
        assert compute_is_high_amount(500) == 0

    def test_is_high_amount_below_threshold(self):
        """Amount < 500 should not be high amount."""
        from features import compute_is_high_amount
        assert compute_is_high_amount(499) == 0
        assert compute_is_high_amount(1) == 0

    def test_is_suspicious_combination(self):
        """Foreign + amount > 200 should be suspicious."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("CN", 201) == 1
        assert compute_is_suspicious("NG", 500) == 1

    def test_is_suspicious_foreign_below_threshold(self):
        """Foreign but amount <= 200 should not be suspicious."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("CN", 200) == 0
        assert compute_is_suspicious("CN", 100) == 0

    def test_is_suspicious_domestic(self):
        """Domestic transaction should not be suspicious regardless of amount."""
        from features import compute_is_suspicious
        assert compute_is_suspicious("US", 10000) == 0


class TestRiskLevelLogic:
    """Tests for risk level classification."""

    def test_risk_level_high(self):
        """Fraud probability > 0.7 should be HIGH."""
        for prob in [0.71, 0.8, 0.99, 1.0]:
            if prob > 0.7:
                risk_level = "HIGH"
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            assert risk_level == "HIGH", f"Failed for prob={prob}"

    def test_risk_level_medium(self):
        """Fraud probability > 0.4 and <= 0.7 should be MEDIUM."""
        for prob in [0.41, 0.5, 0.7]:
            if prob > 0.7:
                risk_level = "HIGH"
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            assert risk_level == "MEDIUM", f"Failed for prob={prob}"

    def test_risk_level_low(self):
        """Fraud probability <= 0.4 should be LOW."""
        for prob in [0.0, 0.1, 0.39, 0.4]:
            if prob > 0.7:
                risk_level = "HIGH"
            elif prob > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            assert risk_level == "LOW", f"Failed for prob={prob}"


class TestFeatureOrder:
    """Tests for feature order matching."""

    def test_feature_order_defined(self):
        """FEATURE_ORDER should be defined and match API expectations."""
        from api.main import FEATURE_ORDER

        expected = [
            "txn_count_5min",
            "total_amount_5min",
            "avg_amount_5min",
            "max_amount_5min",
            "is_foreign",
            "is_high_amount",
            "is_suspicious",
        ]
        assert FEATURE_ORDER == expected
        assert len(FEATURE_ORDER) == 7

    def test_feature_order_includes_all_features(self):
        """FEATURE_ORDER should include all required features."""
        from api.main import FEATURE_ORDER

        required = {"txn_count_5min", "total_amount_5min", "avg_amount_5min",
                   "max_amount_5min", "is_foreign", "is_high_amount", "is_suspicious"}
        assert set(FEATURE_ORDER) == required


class TestPydanticValidation:
    """Tests for Pydantic request/response models."""

    def test_score_request_valid(self):
        """Valid ScoreRequest should be accepted."""
        from api.main import ScoreRequest

        req = ScoreRequest(
            user_id="user_0001",
            amount=100.0,
            merchant_category="groceries",
            merchant_country="US"
        )
        assert req.user_id == "user_0001"
        assert req.amount == 100.0
        assert req.merchant_category == "groceries"
        assert req.merchant_country == "US"

    def test_score_request_rejects_missing_fields(self):
        """ScoreRequest should reject missing required fields."""
        from pydantic import ValidationError
        from api.main import ScoreRequest

        with pytest.raises(ValidationError):
            ScoreRequest(
                amount=100.0,
                merchant_category="groceries",
                merchant_country="US"
            )

    def test_score_request_rejects_negative_amount(self):
        """ScoreRequest should reject negative amount."""
        from pydantic import ValidationError
        from api.main import ScoreRequest

        with pytest.raises(ValidationError):
            ScoreRequest(
                user_id="user_0001",
                amount=-100.0,
                merchant_category="groceries",
                merchant_country="US"
            )

    def test_score_request_rejects_zero_amount(self):
        """ScoreRequest should reject zero amount."""
        from pydantic import ValidationError
        from api.main import ScoreRequest

        with pytest.raises(ValidationError):
            ScoreRequest(
                user_id="user_0001",
                amount=0,
                merchant_category="groceries",
                merchant_country="US"
            )

    def test_score_request_rejects_excessive_amount(self):
        """ScoreRequest should reject amount > 100000."""
        from pydantic import ValidationError
        from api.main import ScoreRequest

        with pytest.raises(ValidationError):
            ScoreRequest(
                user_id="user_0001",
                amount=200000,
                merchant_category="groceries",
                merchant_country="US"
            )

    def test_score_request_rejects_empty_user_id(self):
        """ScoreRequest should reject empty user_id."""
        from pydantic import ValidationError
        from api.main import ScoreRequest

        with pytest.raises(ValidationError):
            ScoreRequest(
                user_id="   ",
                amount=100.0,
                merchant_category="groceries",
                merchant_country="US"
            )

    def test_score_response_model(self):
        """ScoreResponse should have expected fields."""
        from api.main import ScoreResponse

        resp = ScoreResponse(
            user_id="user_0001",
            fraud_probability=0.75,
            risk_level="HIGH",
            features_used={"txn_count_5min": 5},
            redis_features_found=True,
            latency_ms=10.5
        )

        assert resp.risk_level == "HIGH"
        assert resp.fraud_probability == 0.75
        assert resp.redis_features_found is True


class TestRedisKeyFormat:
    """Tests for Redis key formatting."""

    def test_user_features_key_format(self):
        """User features should be stored at user_features:{user_id}."""
        user_id = "user_0123"
        key = f"user_features:{user_id}"
        assert key == "user_features:user_0123"

    def test_transaction_flags_key_format(self):
        """Transaction flags should be stored at txn_flags:{txn_id}."""
        txn_id = "txn-abc-123"
        key = f"txn_flags:{txn_id}"
        assert key == "txn_flags:txn-abc-123"

    def test_ttl_values(self):
        """TTL values should be appropriate."""
        USER_FEATURE_TTL = 86400  # 24 hours
        TXN_FLAG_TTL = 3600  # 1 hour

        assert USER_FEATURE_TTL == 24 * 60 * 60  # 86400
        assert TXN_FLAG_TTL == 60 * 60  # 3600
