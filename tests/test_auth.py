"""
Unit tests for API authentication and rate limiting.
"""

import pytest
import sys
import os

# Set required env var before importing
os.environ["API_KEYS"] = "test-key-1,test-key-2"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAPIKeyHeader:
    """Tests for API key header configuration."""

    def test_api_key_header_name(self):
        """API key header should be X-API-Key."""
        from api.auth import API_KEY_HEADER
        assert API_KEY_HEADER == "X-API-Key"

    def test_api_keys_loaded_from_env(self):
        """API keys should be loaded from environment."""
        from api.auth import API_KEYS
        assert len(API_KEYS) == 2
        assert "test-key-1" in API_KEYS
        assert "test-key-2" in API_KEYS


class TestRateLimiter:
    """Tests for rate limiting logic."""

    def test_rate_limiter_init(self):
        """Rate limiter should initialize with correct defaults."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter()
        assert limiter.requests_per_minute == 100
        assert limiter.authenticated_requests_per_minute == 1000

    def test_rate_limiter_custom_limits(self):
        """Rate limiter should accept custom limits."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=50, authenticated_requests_per_minute=500)
        assert limiter.requests_per_minute == 50
        assert limiter.authenticated_requests_per_minute == 500

    def test_client_key_format_ip(self):
        """Client key should include IP prefix."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter()

        class MockRequest:
            class client:
                host = "192.168.1.1"

        key = limiter._get_client_key(MockRequest(), api_key=None)
        assert key == "ip:192.168.1.1"

    def test_client_key_format_api_key(self):
        """Client key should include API key prefix when authenticated."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter()

        class MockRequest:
            class client:
                host = "192.168.1.1"

        key = limiter._get_client_key(MockRequest(), api_key="test-key")
        assert key == "api_key:test-key"

    def test_first_request_allowed(self):
        """First request should always be allowed."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10)

        class MockRequest:
            class client:
                host = "192.168.1.100"

        is_allowed, headers = limiter.check_rate_limit(MockRequest(), api_key=None)
        assert is_allowed is True
        assert "X-RateLimit-Limit" in headers

    def test_rate_limit_exceeded(self):
        """Requests exceeding limit should be blocked."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=2)

        class MockRequest:
            class client:
                host = "192.168.1.200"

        # Make 2 requests (should be allowed)
        is_allowed1, _ = limiter.check_rate_limit(MockRequest(), api_key=None)
        is_allowed2, _ = limiter.check_rate_limit(MockRequest(), api_key=None)

        # Third request should be blocked
        is_allowed3, headers = limiter.check_rate_limit(MockRequest(), api_key=None)

        assert is_allowed1 is True
        assert is_allowed2 is True
        assert is_allowed3 is False
        assert "Retry-After" in headers

    def test_authenticated_has_higher_limit(self):
        """Authenticated requests should have higher rate limit."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, authenticated_requests_per_minute=100)

        class MockRequest:
            class client:
                host = "192.168.1.50"

        # Unauthenticated
        is_allowed_unauth, headers_unauth = limiter.check_rate_limit(MockRequest(), api_key=None)

        # Authenticated
        is_allowed_auth, headers_auth = limiter.check_rate_limit(MockRequest(), api_key="test-key")

        assert headers_unauth["X-RateLimit-Limit"] == "10"
        assert headers_auth["X-RateLimit-Limit"] == "100"


class TestRateLimitHeaders:
    """Tests for rate limit response headers."""

    def test_headers_contain_limit(self):
        """Response should include rate limit headers."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=50)

        class MockRequest:
            class client:
                host = "10.0.0.1"

        is_allowed, headers = limiter.check_rate_limit(MockRequest())

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert headers["X-RateLimit-Limit"] == "50"

    def test_headers_contain_remaining(self):
        """Response should show remaining requests."""
        from api.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10)

        class MockRequest:
            class client:
                host = "10.0.0.2"

        is_allowed, headers = limiter.check_rate_limit(MockRequest())

        assert "X-RateLimit-Remaining" in headers
        # First request: 10 - 0 - 1 = 9 remaining
        assert int(headers["X-RateLimit-Remaining"]) == 9
