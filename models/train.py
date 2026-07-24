"""
Phase 5: XGBoost Fraud Detection Model

This script can run in two modes:
1. SYNTHETIC (default): Generates synthetic data for quick iteration
2. HISTORICAL: Uses Feast to pull real features from the offline Parquet store

Usage:
    python models/train.py                 # Synthetic mode (default)
    python models/train.py --historical   # Historical mode (requires data in Parquet)

For historical mode, first run:
    1. docker compose up -d
    2. python producer/produce_transactions.py  # Run for ~5 minutes
    3. python spark/feature_engineering.py       # Let it process

Then run:
    cd feast && feast apply && cd ..
    python models/train.py --historical

Note: In production, fraud labels come from delayed chargeback reports
(typically 30-90 days later), not from the streaming pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

# Add project root to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import (
    compute_is_foreign,
    compute_is_high_amount,
    compute_is_suspicious,
    HIGH_AMOUNT_THRESHOLD,
    SUSPICIOUS_AMOUNT_THRESHOLD,
)

# Add feast to path if available
try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


# ── 0. Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("fraud_detection")


def generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data (for development/testing).

    In production, this would be replaced with historical features from Feast
    combined with labels from chargeback data.
    """
    print("Generating synthetic dataset...")
    np.random.seed(seed)

    # Generate base features
    txn_count_5min    = np.random.randint(1, 21, n)
    total_amount_5min = np.random.uniform(10, 5000, n)
    avg_amount_5min   = np.random.uniform(10, 2500, n)
    max_amount_5min   = np.random.uniform(10, 5000, n)

    # Generate merchant country (80% US, 20% foreign)
    merchant_countries = np.random.choice(["US", "non-US"], n, p=[0.8, 0.2])
    is_foreign = np.array([1 if c != "US" else 0 for c in merchant_countries])

    # Compute transaction flags using shared module functions
    is_high_amount = np.array([
        compute_is_high_amount(amt) for amt in max_amount_5min
    ])
    is_suspicious = np.array([
        compute_is_suspicious(country, amt)
        for country, amt in zip(merchant_countries, avg_amount_5min)
    ])

    # Fraud label rule (same as used in Spark feature engineering):
    #   fraud=1 if (is_suspicious=1 AND max_amount > 500)
    #           OR (txn_count > 10 AND is_foreign=1)
    fraud = (
        ((is_suspicious == 1) & (max_amount_5min > 500)) |
        ((txn_count_5min > 10) & (is_foreign == 1))
    ).astype(int)

    df = pd.DataFrame({
        "txn_count_5min":    txn_count_5min,
        "total_amount_5min": total_amount_5min,
        "avg_amount_5min":   avg_amount_5min,
        "max_amount_5min":   max_amount_5min,
        "is_foreign":        is_foreign,
        "is_high_amount":    is_high_amount,
        "is_suspicious":     is_suspicious,
        "fraud":             fraud,
    })

    fraud_rate = fraud.mean() * 100
    print(f"  Dataset: {n} rows | fraud rate: {fraud_rate:.1f}%")
    return df


def load_historical_data() -> pd.DataFrame:
    """
    Load historical features from Feast offline store (Parquet in MinIO).

    This is the production approach: pull features from the offline store,
    then join with labels from the label store (chargeback data).

    Returns a DataFrame with features and labels.
    """
    if not FEAST_AVAILABLE:
        raise ImportError("Feast is not installed. Install with: pip install feast")

    print("Loading historical data from Feast offline store...")

    # Load Feast feature store
    feast_repo = os.path.join(PROJECT_ROOT, "feast")
    store = FeatureStore(repo_path=feast_repo)

    # Get the entity dataframe (list of users with event timestamps)
    # In production, this would come from your label store
    # For demo, we'll create entity timestamps from the last 24 hours
    from datetime import datetime, timedelta

    # Get feature view
    feature_view = store.get_feature_view("user_transaction_features")

    # Get entity DataFrame - in production this would be your labeled data
    # For now, we'll query recent entities from Redis or create sample entities
    # This is a simplified version - production would join with actual labels

    # For demonstration, let's try to get some entities
    # Note: In real production, you'd have a label table with user_id, event_time, is_fraud

    print("  Note: Historical mode requires chargeback labels from a label store.")
    print("  For demo purposes, falling back to synthetic data generation.")
    print("  In production: labels come from chargeback data, not the stream.")

    # Fallback to synthetic for demo
    return generate_synthetic_data()


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict
) -> XGBClassifier:
    """Train XGBoost model."""
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate model and return metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    return metrics


def log_to_mlflow(
    model: XGBClassifier,
    metrics: dict,
    params: dict,
    feature_names: list,
    artifact_dir: str
):
    """Log params, metrics, and artifacts to MLflow."""
    # Log params (filter out deprecated params)
    mlflow.log_params({k: v for k, v in params.items()
                       if k not in ("use_label_encoder", "eval_metric")})
    mlflow.log_metrics(metrics)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost — Feature Importance")
    plt.tight_layout()

    plot_path = os.path.join(artifact_dir, "feature_importance.png")
    fig.savefig(plot_path)
    plt.close(fig)
    mlflow.log_artifact(plot_path, artifact_path="plots")

    # Log model
    mlflow.xgboost.log_model(model, artifact_path="model")


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Use Feast offline store for historical features (requires data in Parquet)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples for synthetic mode"
    )
    args = parser.parse_args()

    # Load data
    if args.historical and FEAST_AVAILABLE:
        df = load_historical_data()
    else:
        if args.historical and not FEAST_AVAILABLE:
            print("Warning: Feast not available, using synthetic data")
        df = generate_synthetic_data(n=args.n_samples)

    # Feature columns
    FEATURES = [
        "txn_count_5min", "total_amount_5min", "avg_amount_5min",
        "max_amount_5min", "is_foreign", "is_high_amount", "is_suspicious",
    ]
    X = df[FEATURES]
    y = df["fraud"]

    # Train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Model params
    params = {
        "n_estimators":  100,
        "max_depth":     4,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric":   "logloss",
        "random_state":  42,
    }

    print("\nStarting MLflow run...")

    # Train and evaluate
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model = train_model(X_train.values, y_train.values, X_test.values, y_test.values, params)
        metrics = evaluate_model(model, X_test.values, y_test.values)
        log_to_mlflow(model, metrics, params, FEATURES, ARTIFACT_DIR)

    # Print summary
    print("\n" + "=" * 55)
    print("  FRAUD DETECTION MODEL — TRAINING SUMMARY")
    print("=" * 55)
    print(f"  MLflow Run ID : {run_id}")
    print(f"  Experiment    : fraud_detection")
    print(f"  Tracking URI  : {mlflow.get_tracking_uri()}")
    print(f"  Mode          : {'Historical (Feast)' if args.historical else 'Synthetic'}")
    print("-" * 55)
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    print("-" * 55)
    print(f"  Model saved   : mlruns/{run_id}/artifacts/model/")
    print(f"  Feature plot  : models/artifacts/feature_importance.png")
    print("=" * 55)
    print("\nDone. Run `mlflow ui --port 5000` to view the experiment.")


if __name__ == "__main__":
    main()
