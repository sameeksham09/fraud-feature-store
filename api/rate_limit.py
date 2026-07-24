"""
Rate limiting middleware using in-memory storage.

For production, consider using Redis-backed rate limiting
(e.g., slowapi with Redis backend).
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status


class RateLimiter:
    """
    Simple in-memory rate limiter.

    Limits:
    - Default: 100 requests per minute per IP
    - Authenticated: 1000 requests per minute per API key
    """

    def __init__(self, requests_per_minute: int = 100, authenticated_requests_per_minute: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.authenticated_requests_per_minute = authenticated_requests_per_minute

        # Store request counts: {key: [(timestamp, count)]}
        self._requests: Dict[str, list] = defaultdict(list)

        # Cleanup old entries every 60 seconds
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        """Remove entries older than 1 minute."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:
            return

        cutoff = current_time - 60
        for key in list(self._requests.keys()):
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            if not self._requests[key]:
                del self._requests[key]

        self._last_cleanup = current_time

    def _get_client_key(self, request: Request, api_key: str = None) -> str:
        """Get the rate limit key (API key if authenticated, otherwise IP)."""
        if api_key:
            return f"api_key:{api_key}"
        return f"ip:{request.client.host}"

    def check_rate_limit(self, request: Request, api_key: str = None) -> Tuple[bool, Dict[str, int]]:
        """
        Check if the request should be rate limited.

        Args:
            request: The FastAPI request
            api_key: The API key (if authenticated)

        Returns:
            (is_allowed, headers)
            - is_allowed: True if request is allowed, False if rate limited
            - headers: Dict with rate limit headers (X-RateLimit-Limit, X-RateLimit-Remaining)
        """
        self._cleanup_old_entries()

        current_time = time.time()
        client_key = self._get_client_key(request, api_key)

        # Determine limit based on authentication
        limit = self.authenticated_requests_per_minute if api_key else self.requests_per_minute

        # Get existing requests in the last minute
        requests = self._requests[client_key]
        recent_requests = [t for t in requests if t > current_time - 60]

        if len(recent_requests) >= limit:
            # Calculate retry-after
            oldest = min(recent_requests)
            retry_after = int(60 - (current_time - oldest))

            headers = {
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(retry_after),
            }
            return False, headers

        # Add current request
        self._requests[client_key].append(current_time)

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(limit - len(recent_requests) - 1),
        }
        return True, headers


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request, api_key: str = None) -> Dict[str, int]:
    """
    FastAPI dependency for rate limiting.

    Usage:
        @app.get("/endpoint")
        async def endpoint(request: Request, api_key: str = Depends(get_api_key)):
            check_rate_limit(request, api_key)
            ...
    """
    is_allowed, headers = rate_limiter.check_rate_limit(request, api_key)

    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers=headers,
        )

    return headers
