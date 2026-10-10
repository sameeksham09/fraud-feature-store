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
    """Tests that exercise models/train.py itself (not re-implementations of it)."""

    @pytest.fixture(scope="class")
    def raw(self):
        from models.train import generate_synthetic_data
        return generate_synthetic_data(n=2000, seed=7)

    def test_row_count(self, raw):
        assert len(raw) == 2000

    def test_has_transaction_and_window_columns(self, raw):
        from features import WINDOWED_FEATURES
        for col in WINDOWED_FEATURES + ["amount", "merchant_country", "fraud"]:
            assert col in raw.columns

    def test_txn_count_in_range(self, raw):
        assert raw["txn_count_5min"].between(1, 20).all()

    def test_amounts_positive(self, raw):
        assert (raw["amount"] > 0).all()
        assert (raw["total_amount_5min"] > 0).all()

    def test_both_classes_present(self, raw):
        assert set(raw["fraud"].unique()) == {0, 1}

    def test_deterministic_with_seed(self):
        from models.train import generate_synthetic_data
        a = generate_synthetic_data(n=200, seed=1)
        b = generate_synthetic_data(n=200, seed=1)
        pd.testing.assert_frame_equal(a, b)


class TestTrainingServingParity:
    """
    Guards against training/serving skew: training must derive features exactly
    the way the API does at request time.
    """

    @pytest.fixture(scope="class")
    def raw(self):
        from models.train import generate_synthetic_data
        return generate_synthetic_data(n=500, seed=3)

    def test_feature_matrix_columns_match_api_order(self, raw):
        from models.train import build_feature_matrix
        from features import FEATURE_ORDER
        assert list(build_feature_matrix(raw).columns) == FEATURE_ORDER

    def test_every_row_matches_serving_path(self, raw):
        """Rebuild each row the way the API does (Redis hash of strings) and compare."""
        from models.train import build_feature_matrix
        from features import build_feature_vector, WINDOWED_FEATURES, FEATURE_ORDER

        X = build_feature_matrix(raw)
        for i, rec in raw.iterrows():
            redis_hash = {k: str(rec[k]) for k in WINDOWED_FEATURES}  # Redis stores strings
            served = build_feature_vector(redis_hash, rec["amount"], rec["merchant_country"])
            assert [served[f] for f in FEATURE_ORDER] == pytest.approx(X.loc[i].tolist())

    def test_flags_use_transaction_amount_not_aggregates(self):
        """A small transaction stays 'not high' even if the user's 5-min max is huge."""
        from models.train import build_feature_matrix
        row = pd.DataFrame([{
            "txn_count_5min": 3, "total_amount_5min": 9000.0,
            "avg_amount_5min": 3000.0, "max_amount_5min": 8000.0,
            "amount": 25.0, "merchant_country": "NG",
        }])
        X = build_feature_matrix(row).iloc[0]
        assert X["is_high_amount"] == 0.0   # 25 <= 500, regardless of max_amount_5min
        assert X["is_suspicious"] == 0.0    # 25 <= 200, regardless of avg_amount_5min
        assert X["is_foreign"] == 1.0

    def test_missing_windowed_features_default_to_zero(self):
        """NaN from an offline join == missing Redis key at serving time."""
        from models.train import build_feature_matrix
        from features import build_feature_vector
        row = pd.DataFrame([{
            "txn_count_5min": np.nan, "total_amount_5min": np.nan,
            "avg_amount_5min": np.nan, "max_amount_5min": np.nan,
            "amount": 700.0, "merchant_country": "US",
        }])
        offline = build_feature_matrix(row).iloc[0].to_dict()
        online = build_feature_vector({}, 700.0, "US")
        assert offline == online


class TestHistoricalMode:
    """--historical must fail loudly, never silently train on synthetic data."""

    def test_rejects_too_little_data(self):
        from models.train import validate_training_data, HistoricalDataError
        df = pd.DataFrame({"fraud": [0] * 50 + [1] * 2})
        with pytest.raises(HistoricalDataError):
            validate_training_data(df, min_rows=200, min_fraud=5)

    def test_accepts_enough_data(self):
        from models.train import validate_training_data
        df = pd.DataFrame({"fraud": [0] * 300 + [1] * 10})
        validate_training_data(df, min_rows=200, min_fraud=5)

    def test_missing_feast_raises(self, monkeypatch):
        import models.train as train
        monkeypatch.setattr(train, "FEAST_AVAILABLE", False)
        with pytest.raises(train.HistoricalDataError):
            train.load_historical_data()

    def test_labeled_transactions_join(self, tmp_path):
        """Bronze JSON + label Parquet -> Feast entity df."""
        from models.train import load_labeled_transactions
        bronze = tmp_path / "bronze"
        labels = tmp_path / "labels"
        bronze.mkdir()
        labels.mkdir()
        pd.DataFrame([
            {"transaction_id": "t1", "user_id": "u1", "amount": 10.0,
             "merchant_country": "US", "timestamp": "2026-01-01T12:00:30"},
            {"transaction_id": "t2", "user_id": "u2", "amount": 900.0,
             "merchant_country": "NG", "timestamp": "2026-01-01T12:01:00"},
            {"transaction_id": "t3", "user_id": "u3", "amount": 5.0,
             "merchant_country": "US", "timestamp": "2026-01-01T12:02:00"},  # unlabeled
        ]).to_json(bronze / "part-00000.json", orient="records", lines=True)
        pd.DataFrame({"transaction_id": ["t1", "t2"], "is_fraud": [False, True]}) \
            .to_parquet(labels / "part-00000.parquet")

        df = load_labeled_transactions(str(bronze), str(labels)).set_index("transaction_id")
        assert set(df.index) == {"t1", "t2"}            # inner join drops unlabeled
        assert df.loc["t2", "fraud"] == 1
        assert str(df["event_timestamp"].dt.tz) == "UTC"
        assert df.loc["t1", "event_timestamp"] == pd.Timestamp("2026-01-01T12:00:30Z")


class TestFeatureOrder:
    """Feature order is defined once, in features/, and shared."""

    def test_feature_list_has_7_features(self):
        from features import FEATURE_ORDER
        assert len(FEATURE_ORDER) == 7

    def test_api_uses_shared_feature_order(self):
        from features import FEATURE_ORDER
        from api.main import FEATURE_ORDER as API_ORDER
        assert API_ORDER is FEATURE_ORDER


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
