"""
Data validation for transactions and features.

Simple validation functions that can be used to validate:
1. Kafka message format (transaction schema)
2. Redis stored features
"""

import re
import logging
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Custom Exception ────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# ── Constants ───────────────────────────────────────────────────────────────────

VALID_MERCHANT_CATEGORIES = [
    'groceries', 'electronics', 'restaurants', 'travel',
    'gas_station', 'pharmacy', 'clothing', 'entertainment'
]

VALID_COUNTRIES = ['US', 'NG', 'RU', 'CN', 'BR', 'IN']

USER_ID_PATTERN = re.compile(r'^user_\d{4}$')

UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


# ── Validation Functions ────────────────────────────────────────────────────────

def validate_transaction(data: Dict[str, Any], raise_on_error: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate a transaction dictionary.

    Args:
        data: The transaction dictionary to validate
        raise_on_error: If True, raise ValidationError on failure instead of returning

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Required fields
    required_fields = ['transaction_id', 'user_id', 'amount', 'merchant_category',
                      'merchant_country', 'timestamp']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        if raise_on_error:
            raise ValidationError(errors)
        return False, errors

    # transaction_id: UUID format
    txn_id = data.get('transaction_id', '')
    if not UUID_PATTERN.match(txn_id):
        errors.append(f"Invalid transaction_id format: {txn_id}")

    # user_id: user_XXXX format
    user_id = data.get('user_id', '')
    if not USER_ID_PATTERN.match(user_id):
        errors.append(f"Invalid user_id format: {user_id}")

    # amount: positive number
    amount = data.get('amount')
    if amount is None:
        errors.append("amount is required")
    elif not isinstance(amount, (int, float)) or amount <= 0:
        errors.append(f"amount must be positive: {amount}")
    elif amount >= 100000:
        errors.append(f"amount exceeds maximum: {amount}")

    # merchant_category: must be in allowed list
    category = data.get('merchant_category')
    if category not in VALID_MERCHANT_CATEGORIES:
        errors.append(f"Invalid merchant_category: {category}")

    # merchant_country: must be in allowed list
    country = data.get('merchant_country')
    if country not in VALID_COUNTRIES:
        errors.append(f"Invalid merchant_country: {country}")

    # timestamp: non-empty string
    timestamp = data.get('timestamp')
    if not timestamp or not isinstance(timestamp, str):
        errors.append(f"Invalid timestamp: {timestamp}")

    if errors and raise_on_error:
        raise ValidationError(errors)

    return len(errors) == 0, errors


def validate_user_features(data: Dict[str, Any], raise_on_error: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate user features dictionary from Redis.

    Args:
        data: The user features dictionary to validate
        raise_on_error: If True, raise ValidationError on failure

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Required fields
    required_fields = ['user_id', 'txn_count_5min', 'total_amount_5min',
                      'avg_amount_5min', 'max_amount_5min']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        if raise_on_error:
            raise ValidationError(errors)
        return False, errors

    # user_id: user_XXXX format
    user_id = data.get('user_id', '')
    if not USER_ID_PATTERN.match(user_id):
        errors.append(f"Invalid user_id format: {user_id}")

    # Numeric fields: must be non-negative
    numeric_fields = ['txn_count_5min', 'total_amount_5min', 'avg_amount_5min', 'max_amount_5min']
    for field in numeric_fields:
        value = data.get(field)
        if value is None:
            errors.append(f"{field} is required")
        elif not isinstance(value, (int, float)) or value < 0:
            errors.append(f"{field} must be non-negative: {value}")

    if errors and raise_on_error:
        raise ValidationError(errors)

    return len(errors) == 0, errors


def validate_transaction_flags(data: Dict[str, Any], raise_on_error: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate transaction flags dictionary from Redis.

    Args:
        data: The transaction flags dictionary to validate
        raise_on_error: If True, raise ValidationError on failure

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Required fields
    required_fields = ['transaction_id', 'user_id', 'amount', 'is_foreign',
                      'is_high_amount', 'is_suspicious']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if errors:
        if raise_on_error:
            raise ValidationError(errors)
        return False, errors

    # user_id: user_XXXX format
    user_id = data.get('user_id', '')
    if not USER_ID_PATTERN.match(user_id):
        errors.append(f"Invalid user_id format: {user_id}")

    # amount: positive number
    amount = data.get('amount')
    if not isinstance(amount, (int, float)) or amount <= 0:
        errors.append(f"amount must be positive: {amount}")

    # Boolean fields
    bool_fields = ['is_foreign', 'is_high_amount', 'is_suspicious']
    for field in bool_fields:
        value = data.get(field)
        if not isinstance(value, bool):
            errors.append(f"{field} must be boolean: {value}")

    if errors and raise_on_error:
        raise ValidationError(errors)

    return len(errors) == 0, errors
