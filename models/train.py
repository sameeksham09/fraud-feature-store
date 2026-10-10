"""
Phase 5: XGBoost Fraud Detection Model

Two data modes, ONE feature path. Both modes build the model input with
features.build_feature_vector(), the same function the API calls at serving
time, so training and serving can't drift apart.

1. SYNTHETIC (default): generated windowed aggregates + a transaction.
2. HISTORICAL (--historical): real point-in-time features from Feast.
   Entity rows = transactions from the Bronze log joined with labels from the
   simulated chargeback feed; Feast's get_historical_features() attaches the
   windowed aggregates as they were *before* each transaction.

Usage:
    python models/train.py                 # synthetic (default)
    python models/train.py --historical    # Feast offline store

For historical mode:
    1. docker compose up -d
    2. python producer/produce_transactions.py   # leave running ~20+ minutes
    3. python spark/feature_engineering.py       # Silver windows are written
                                                 # ~15 min after the fact
                                                 # (5m window + 10m watermark)
    4. make feast-apply                          # registers the feature view
    5. python models/train.py --historical

Historical mode fails with a clear error (it does NOT fall back to synthetic
data) if Feast isn't installed or there isn't enough labeled data yet.

Note: in production, fraud labels come from delayed chargeback reports
(typically 30-90 days later), not from the streaming pipeline.
"""

import os
import sys
import glob
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
from features import (  # noqa: E402
    FEATURE_ORDER,
    WINDOWED_FEATURES,
    build_feature_vector,
)

try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FeatureStore = None  # type: ignore
    FEAST_AVAILABLE = False


# ── 0. Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "models", "artifacts")
FEAST_REPO   = os.path.join(PROJECT_ROOT, "feast")
FEATURE_VIEW = "user_transaction_features"

# Offline store locations (written by spark/feature_engineering.py)
BRONZE_TRANSACTIONS_PATH = "s3://features/bronze/raw_transactions/"
LABELS_PATH              = "s3://features/labels/"

MIN_HISTORICAL_ROWS  = 200
MIN_HISTORICAL_FRAUD = 5


class HistoricalDataError(RuntimeError):
    """Raised when historical training can't run; never silently falls back."""


# ── 1. Feature matrix (shared with serving) ──────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn rows of {windowed aggregates, amount, merchant_country} into the model
    input, using the exact function the API uses at serving time.
    """
    rows = [
        build_feature_vector(
            {name: rec.get(name) for name in WINDOWED_FEATURES},
            rec["amount"],
            rec["merchant_country"],
        )
        for rec in df.to_dict("records")
    ]
    return pd.DataFrame(rows, columns=FEATURE_ORDER, index=df.index)


# ── 2. Synthetic data ─────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic raw training rows (for development/testing).

    Each row is a transaction being scored (amount, merchant_country) plus the
    user's trailing-5-minute aggregates, i.e. exactly what the API has at
    request time. Features are derived later by build_feature_matrix().
    """
    print("Generating synthetic dataset...")
    rng = np.random.RandomState(seed)

    df = pd.DataFrame({
        "txn_count_5min":    rng.randint(1, 21, n),
        "total_amount_5min": rng.uniform(10, 5000, n),
        "avg_amount_5min":   rng.uniform(10, 2500, n),
        "max_amount_5min":   rng.uniform(10, 5000, n),
        # The transaction being scored
        "amount":            rng.uniform(1, 5000, n).round(2),
        "merchant_country":  rng.choice(["US", "NG", "RU", "CN", "GB"], n,
                                        p=[0.8, 0.05, 0.05, 0.05, 0.05]),
    })

    feats = build_feature_matrix(df)

    # Synthetic fraud rule, expressed on the same features the model sees:
    #   fraud=1 if (suspicious transaction AND it's a high amount)
    #           OR (bursty user, >10 txns in 5 min, AND foreign transaction)
    df["fraud"] = (
        ((feats["is_suspicious"] == 1) & (feats["is_high_amount"] == 1)) |
        ((feats["txn_count_5min"] > 10) & (feats["is_foreign"] == 1))
    ).astype(int)

    print(f"  Dataset: {n} rows | fraud rate: {df['fraud'].mean() * 100:.1f}%")
    return df


# ── 3. Historical data (Feast) ────────────────────────────────────────────────

def _minio_storage_options() -> dict:
    """fsspec/s3fs options for MinIO, from the same env vars Spark uses."""
    return {
        "key": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        "secret": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        "client_kwargs": {
            "endpoint_url": os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        },
    }


def _configure_feast_s3_env() -> None:
    """Feast's file offline store reads s3:// via PyArrow, which uses AWS_* env vars."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", "minioadmin"))
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def _read_json_lines(path: str, storage_options=None) -> pd.DataFrame:
    """Read every Spark JSON part file under a directory (local or s3://)."""
    if path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(
            key=storage_options["key"], secret=storage_options["secret"],
            client_kwargs=storage_options["client_kwargs"],
        )
        files = [f for f in fs.glob(path.rstrip("/") + "/part-*.json")]
        frames = []
        for f in files:
            with fs.open(f) as fh:
                frames.append(pd.read_json(fh, lines=True))
    else:
        files = sorted(glob.glob(os.path.join(path, "part-*.json")))
        frames = [pd.read_json(f, lines=True) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_labeled_transactions(
    transactions_path: str = BRONZE_TRANSACTIONS_PATH,
    labels_path: str = LABELS_PATH,
    storage_options=None,
) -> pd.DataFrame:
    """
    Build the Feast entity DataFrame: one row per labeled transaction with
    user_id, event_timestamp (when the transaction happened), the transaction's
    own amount + merchant_country, and the fraud label.
    """
    if storage_options is None and transactions_path.startswith("s3://"):
        storage_options = _minio_storage_options()

    try:
        txns = _read_json_lines(transactions_path, storage_options)
        labels = pd.read_parquet(labels_path, storage_options=storage_options)
    except ImportError as exc:
        raise HistoricalDataError(
            f"Missing dependency for reading MinIO ({exc}). Run `make install` "
            "(needs s3fs from requirements.txt)."
        ) from exc
    except FileNotFoundError as exc:
        raise HistoricalDataError(
            f"Offline data not found ({exc}). Run the producer and the Spark job "
            "for a while first so Bronze transactions and labels exist in MinIO."
        ) from exc
    except Exception as exc:  # e.g. MinIO not running, bad credentials
        raise HistoricalDataError(
            f"Could not read offline data from {transactions_path} / {labels_path}: "
            f"{type(exc).__name__}: {exc}. Is MinIO up (`make up`)?"
        ) from exc

    if txns.empty or labels.empty:
        raise HistoricalDataError(
            "No transactions or labels in the offline store yet. Run "
            "producer/produce_transactions.py and spark/feature_engineering.py first."
        )

    txns = txns[["transaction_id", "user_id", "amount", "merchant_country", "timestamp"]]
    labels = labels[["transaction_id", "is_fraud"]].drop_duplicates("transaction_id")

    entity_df = txns.merge(labels, on="transaction_id", how="inner")
    # Producer timestamps are naive UTC ISO strings
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["timestamp"], utc=True)
    entity_df["fraud"] = entity_df["is_fraud"].astype(int)
    return entity_df.drop(columns=["timestamp", "is_fraud"]).reset_index(drop=True)


def _users_with_feature_rows(store) -> set:
    """user_ids that have at least one window in the feature view's offline source."""
    import pyarrow.dataset as pads
    from feast.infra.offline_stores.file_source import FileSource as FeastFileSource

    source = store.get_feature_view(FEATURE_VIEW).batch_source
    filesystem, path = FeastFileSource.create_filesystem_and_path(
        source.path, source.s3_endpoint_override
    )
    # Spark's _spark_metadata/ is skipped (default ignore_prefixes: "." and "_")
    dataset = pads.dataset(path, filesystem=filesystem, format="parquet")
    return set(dataset.to_table(columns=["user_id"]).column("user_id").to_pylist())


def build_historical_training_set(store, entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach point-in-time windowed features to each labeled transaction.

    For a transaction at time t, Feast joins the latest window with
    window_end <= t (within the feature view TTL), so no feature can include
    the transaction itself or anything after it.

    Every input row is kept. A transaction with no prior window (e.g. a
    user's first transaction) gets NaN windowed features, which
    build_feature_vector() turns into 0.0, the same as a missing Redis key at
    serving time. Feast's file offline store (dask) doesn't handle that case
    itself: it crashes if a user has no windows at all, and silently drops rows
    whose windows all fall outside the TTL. Those are often first-time foreign
    transactions, i.e. disproportionately fraud. So we only send users that
    have windows, and left-join the result back onto every entity row.
    """
    feature_refs = [f"{FEATURE_VIEW}:{name}" for name in WINDOWED_FEATURES]

    entity_df = entity_df.reset_index(drop=True).copy()
    entity_df["_row_id"] = range(len(entity_df))

    known_users = _users_with_feature_rows(store)
    to_query = entity_df[entity_df["user_id"].isin(known_users)]

    if len(to_query):
        feats = store.get_historical_features(
            entity_df=to_query,
            features=feature_refs,
        ).to_df()[["_row_id"] + WINDOWED_FEATURES]
    else:
        feats = pd.DataFrame(columns=["_row_id"] + WINDOWED_FEATURES)

    out = entity_df.merge(feats, on="_row_id", how="left")
    assert len(out) == len(entity_df), "point-in-time join must keep every labeled row"
    return out.drop(columns="_row_id")


def load_historical_data(
    min_rows: int = MIN_HISTORICAL_ROWS,
    min_fraud: int = MIN_HISTORICAL_FRAUD,
) -> pd.DataFrame:
    """Labeled transactions + point-in-time features from the Feast offline store."""
    if not FEAST_AVAILABLE:
        raise HistoricalDataError("Feast is not installed. Install with: pip install feast")
    if not os.path.exists(os.path.join(FEAST_REPO, "registry.db")):
        raise HistoricalDataError("Feast registry not found. Run `make feast-apply` first.")

    print("Loading historical data from Feast offline store...")
    _configure_feast_s3_env()

    entity_df = load_labeled_transactions()
    print(f"  Labeled transactions: {len(entity_df)} "
          f"({int(entity_df['fraud'].sum())} fraud)")

    store = FeatureStore(repo_path=FEAST_REPO)
    df = build_historical_training_set(store, entity_df)

    matched = df[WINDOWED_FEATURES].notna().all(axis=1).sum()
    print(f"  Rows with point-in-time features: {matched}/{len(df)} "
          "(rest = no activity in the prior window -> 0.0)")

    validate_training_data(df, min_rows=min_rows, min_fraud=min_fraud)
    return df


def validate_training_data(df: pd.DataFrame, min_rows: int, min_fraud: int) -> None:
    """Refuse to train on too little data instead of producing a meaningless model."""
    n_fraud = int(df["fraud"].sum())
    n_legit = len(df) - n_fraud
    if len(df) < min_rows or n_fraud < min_fraud or n_legit < min_fraud:
        raise HistoricalDataError(
            f"Not enough labeled history to train: {len(df)} rows, {n_fraud} fraud "
            f"(need >= {min_rows} rows and >= {min_fraud} of each class). "
            "Leave the producer + Spark job running longer; Silver windows land "
            "~15 minutes after the transactions they summarize."
        )


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

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("fraud_detection")

    # Load raw rows (windowed aggregates + transaction + label)
    if args.historical:
        try:
            df = load_historical_data()
        except HistoricalDataError as exc:
            print(f"\n❌ Historical training unavailable: {exc}")
            sys.exit(1)
    else:
        df = generate_synthetic_data(n=args.n_samples)

    # Same feature builder as the API (features.build_feature_vector)
    FEATURES = FEATURE_ORDER
    X = build_feature_matrix(df)
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
