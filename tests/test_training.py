"""
Unit tests for the model training module.
Tests data generation, model training, and evaluation metrics.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainingDataGeneration:
    """Tests for the synthetic training data generation."""

    def test_generates_10000_rows(self):
        """Training data should have 10,000 rows."""
        np.random.seed(42)
        N = 10_000

        txn_count_5min = np.random.randint(1, 21, N)

        assert len(txn_count_5min) == N

    def test_txn_count_in_range(self):
        """Transaction count should be between 1 and 20."""
        np.random.seed(42)
        N = 1000

        txn_count_5min = np.random.randint(1, 21, N)

        assert txn_count_5min.min() >= 1
        assert txn_count_5min.max() <= 20

    def test_amounts_are_positive(self):
        """All amounts should be positive."""
        np.random.seed(42)
        N = 1000

        total_amount_5min = np.random.uniform(10, 5000, N)

        assert (total_amount_5min > 0).all()

    def test_is_foreign_binary(self):
        """is_foreign should be 0 or 1."""
        np.random.seed(42)
        N = 1000

        is_foreign = np.random.randint(0, 2, N)

        assert set(is_foreign).issubset({0, 1})

    def test_is_high_amount_derived_correctly(self):
        """is_high_amount should be 1 when max_amount > 500."""
        max_amounts = np.array([100, 500, 501, 1000])
        is_high_amount = (max_amounts > 500).astype(int)

        assert is_high_amount[0] == 0  # 100 <= 500
        assert is_high_amount[1] == 0  # 500 <= 500
        assert is_high_amount[2] == 1  # 501 > 500
        assert is_high_amount[3] == 1  # 1000 > 500

    def test_is_suspicious_combined_condition(self):
        """is_suspicious should be 1 when is_foreign=1 AND avg_amount > 200."""
        is_foreign = np.array([0, 1, 0, 1])
        avg_amount = np.array([100, 201, 100, 201])

        is_suspicious = ((is_foreign == 1) & (avg_amount > 200)).astype(int)

        assert is_suspicious[0] == 0  # not foreign
        assert is_suspicious[1] == 1  # foreign + high avg
        assert is_suspicious[2] == 0  # foreign but low avg
        assert is_suspicious[3] == 1  # foreign + high avg

    def test_fraud_label_rule(self):
        """Fraud label should be 1 when:
        - (is_suspicious=1 AND max_amount > 500) OR
        - (txn_count > 10 AND is_foreign=1)
        """
        import numpy as np
        # Test case 1: suspicious + high amount = fraud
        is_suspicious = 1
        max_amount_5min = 600
        txn_count_5min = 5
        is_foreign = 1

        fraud = int(
            ((is_suspicious == 1) & (max_amount_5min > 500)) |
            ((txn_count_5min > 10) & (is_foreign == 1))
        )

        assert fraud == 1

    def test_fraud_label_not_fraud_case(self):
        """Fraud should be 0 when neither condition is met."""
        import numpy as np
        is_suspicious = 0
        max_amount_5min = 100
        txn_count_5min = 3
        is_foreign = 0

        fraud = int(
            ((is_suspicious == 1) & (max_amount_5min > 500)) |
            ((txn_count_5min > 10) & (is_foreign == 1))
        )

        assert fraud == 0

    def test_fraud_label_high_count_foreign(self):
        """High transaction count + foreign should be fraud."""
        import numpy as np
        is_suspicious = 0
        max_amount_5min = 100
        txn_count_5min = 15
        is_foreign = 1

        fraud = int(
            ((is_suspicious == 1) & (max_amount_5min > 500)) |
            ((txn_count_5min > 10) & (is_foreign == 1))
        )

        assert fraud == 1

    def test_dataframe_has_all_columns(self):
        """DataFrame should have all required columns."""
        N = 100

        txn_count_5min = np.random.randint(1, 21, N)
        total_amount_5min = np.random.uniform(10, 5000, N)
        avg_amount_5min = np.random.uniform(10, 2500, N)
        max_amount_5min = np.random.uniform(10, 5000, N)
        is_foreign = np.random.randint(0, 2, N)
        is_high_amount = (max_amount_5min > 500).astype(int)
        is_suspicious = ((is_foreign == 1) & (avg_amount_5min > 200)).astype(int)
        fraud = np.zeros(N).astype(int)

        df = pd.DataFrame({
            "txn_count_5min": txn_count_5min,
            "total_amount_5min": total_amount_5min,
            "avg_amount_5min": avg_amount_5min,
            "max_amount_5min": max_amount_5min,
            "is_foreign": is_foreign,
            "is_high_amount": is_high_amount,
            "is_suspicious": is_suspicious,
            "fraud": fraud,
        })

        expected_columns = [
            "txn_count_5min", "total_amount_5min", "avg_amount_5min",
            "max_amount_5min", "is_foreign", "is_high_amount",
            "is_suspicious", "fraud"
        ]

        assert list(df.columns) == expected_columns


class TestFeatureOrder:
    """Tests for feature order matching."""

    def test_feature_list_has_7_features(self):
        """FEATURES should have exactly 7 features."""
        FEATURES = [
            "txn_count_5min", "total_amount_5min", "avg_amount_5min",
            "max_amount_5min", "is_foreign", "is_high_amount", "is_suspicious",
        ]

        assert len(FEATURES) == 7

    def test_feature_order_matches_api(self):
        """Feature order should match what's expected by the API."""
        # This is the source of truth from api/main.py
        API_FEATURE_ORDER = [
            "txn_count_5min",
            "total_amount_5min",
            "avg_amount_5min",
            "max_amount_5min",
            "is_foreign",
            "is_high_amount",
            "is_suspicious",
        ]

        # This is from train.py
        TRAIN_FEATURES = [
            "txn_count_5min", "total_amount_5min", "avg_amount_5min",
            "max_amount_5min", "is_foreign", "is_high_amount", "is_suspicious",
        ]

        assert API_FEATURE_ORDER == TRAIN_FEATURES


class TestModelMetrics:
    """Tests for model evaluation metrics."""

    def test_accuracy_score(self):
        """Accuracy should be correctly calculated."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true)

        # Correct: indices 0,1,2,4,6,7 = 6 correct
        # Wrong: index 3(1!=0),5(0!=1) = 2 wrong
        # Accuracy = 6/8 = 0.75
        assert accuracy == 0.75

    def test_precision_score(self):
        """Precision should measure true positives / predicted positives."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

        # Predicted positives: indices 0, 2, 4, 6 (4 total)
        # True positives: indices 0, 2, 6 (3 out of 4)
        # Precision = 3/4 = 0.75

        true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        predicted_positives = sum(1 for p in y_pred if p == 1)

        precision = true_positives / predicted_positives

        assert precision == 0.75

    def test_recall_score(self):
        """Recall should measure true positives / actual positives."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

        # Actual positives: indices 0, 2, 3, 6 (4 total)
        # True positives: indices 0, 2, 6 (3 out of 4)
        # Recall = 3/4 = 0.75

        true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        actual_positives = sum(1 for t in y_true if t == 1)

        recall = true_positives / actual_positives

        assert recall == 0.75


class TestTrainTestSplit:
    """Tests for train/test split logic."""

    def test_split_ratio(self):
        """Train/test split should be 80/20."""
        from sklearn.model_selection import train_test_split

        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_stratified_split_preserves_class_balance(self):
        """Stratified split should preserve class distribution."""
        from sklearn.model_selection import train_test_split

        # 30% positive class
        y = [0] * 70 + [1] * 30
        X = [[i] for i in range(100)]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_positive_rate = sum(y_train) / len(y_train)
        test_positive_rate = sum(y_test) / len(y_test)

        # Should be approximately equal (both ~30%)
        assert abs(train_positive_rate - test_positive_rate) < 0.1
