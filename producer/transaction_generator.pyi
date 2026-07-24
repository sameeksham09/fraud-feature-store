"""
Type stubs for fraud feature store modules.
"""

from typing import Any

# Module type stubs
def generate_transaction(user_id: str) -> dict[str, Any]: ...
def send_with_retry(producer: Any, topic: str, transaction: dict[str, Any]) -> bool: ...

MERCHANT_CATEGORIES: list[str]
COUNTRIES: list[str]
USER_IDS: list[str]
