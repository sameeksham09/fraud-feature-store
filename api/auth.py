"""
API Key authentication middleware.
"""

import os
import hmac
import hashlib
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Header name for API key
API_KEY_HEADER = "X-API-Key"

# Get API keys from environment - fail fast if not configured
# In production, set multiple keys as: API_KEYS="key1,key2,key3"
_api_keys_env = os.environ.get("API_KEYS")

if _api_keys_env is None or _api_keys_env.strip() == "":
    raise RuntimeError(
        "API_KEYS environment variable is not set. "
        "Please set it before starting the API (e.g., API_KEYS='your-secret-key')."
    )

API_KEYS = [key.strip() for key in _api_keys_env.split(",") if key.strip()]


def _constant_time_compare(val1: str, val2: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.

    Uses hmac.compare_digest for secure comparison.
    """
    return hmac.compare_digest(val1, val2)


def get_api_key(api_key: str = Security(APIKeyHeader(name=API_KEY_HEADER, auto_error=False))) -> str:
    """
    Validate the API key from the request header.

    Args:
        api_key: The API key from the request header

    Returns:
        The validated API key

    Raises:
        HTTPException: If the API key is invalid or missing
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Add '{API_KEY_HEADER}' header with a valid key.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # Use constant-time comparison to prevent timing attacks
    for valid_key in API_KEYS:
        if _constant_time_compare(api_key, valid_key):
            return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


def get_optional_api_key(api_key: str = Security(APIKeyHeader(name=API_KEY_HEADER, auto_error=False))) -> Optional[str]:
    """
    Optionally validate the API key (for endpoints that work with or without auth).

    Args:
        api_key: The API key from the request header (optional)

    Returns:
        The API key if provided and valid, None otherwise
    """
    if api_key is None:
        return None

    # Use constant-time comparison
    for valid_key in API_KEYS:
        if _constant_time_compare(api_key, valid_key):
            return api_key

    return None
