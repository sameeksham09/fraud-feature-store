"""
End-to-end test of the offline training path:

    Spark (same window agg + Parquet settings as the streaming job)
      -> Silver Parquet -> Feast get_historical_features()
      -> models.train.build_historical_training_set / build_feature_matrix

Checks point-in-time correctness (no leakage of the transaction itself or later
ones) and that no labeled row is dropped. Skipped if Spark/Java or Feast are
unavailable.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("pyspark")
pytest.importorskip("feast")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture(scope="module")
def offline_store(tmp_path_factory):
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession, functions as F

    tmp = tmp_path_factory.mktemp("offline")
    try:
        spark = (
            SparkSession.builder.master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            # Same settings as spark/feature_engineering.py
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .getOrCreate()
        )
    except Exception as exc:
        pytest.skip(f"Spark unavailable: {exc}")
    spark.sparkContext.setLogLevel("ERROR")

    # user_1: $100 every minute 12:00:30..12:06:30 (US)
    # user_2: a single $900 NG transaction at 12:03:10 (no prior activity, fraud)
    # user_3: transacts but has no closed windows yet (no Silver rows at all)
    txns = [(f"u1-{m}", "user_1", 100.0, "US", BASE + timedelta(minutes=m, seconds=30), False)
            for m in range(7)]
    txns += [("u2-0", "user_2", 900.0, "NG", BASE + timedelta(minutes=3, seconds=10), True)]
    silver_input = [t for t in txns]  # user_3 deliberately absent from Silver
    txns += [("u3-0", "user_3", 50.0, "US", BASE + timedelta(minutes=5), False)]

    df = spark.createDataFrame(
        [(t[1], t[2], t[4]) for t in silver_input], ["user_id", "amount", "event_time"]
    )
    (
        df.groupBy(F.window("event_time", "5 minutes", "1 minute"), "user_id")
        .agg(
            F.count("*").alias("txn_count_5min"),
            F.sum("amount").alias("total_amount_5min"),
            F.avg("amount").alias("avg_amount_5min"),
            F.max("amount").alias("max_amount_5min"),
        )
        .select(
            "user_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "txn_count_5min", "total_amount_5min", "avg_amount_5min", "max_amount_5min",
        )
        .repartition(3)  # several part files, like a streaming sink
        .write.parquet(str(tmp / "silver"))
    )
    spark.stop()

    # Bronze JSON + label Parquet, as written by the streaming job
    (tmp / "bronze").mkdir()
    (tmp / "labels").mkdir()
    pd.DataFrame([
        {"transaction_id": t[0], "user_id": t[1], "amount": t[2],
         "merchant_country": t[3], "timestamp": t[4].isoformat()}
        for t in txns
    ]).to_json(tmp / "bronze" / "part-00000.json", orient="records", lines=True)
    pd.DataFrame({"transaction_id": [t[0] for t in txns], "is_fraud": [t[5] for t in txns]}) \
        .to_parquet(tmp / "labels" / "part-00000.parquet")

    # Feast repo using the real feature view definition, pointed at local Silver
    repo = tmp / "feast_repo"
    repo.mkdir()
    src = open(os.path.join(REPO_ROOT, "feast", "features.py")).read()
    src = src.replace('path="s3://features/windowed_features/",', f'path="{tmp / "silver"}",')
    src = src.replace('    s3_endpoint_override="http://localhost:9000",\n', "")
    (repo / "feature_store.yaml").write_text(
        "project: test_offline\nregistry: registry.db\nprovider: local\n"
        "online_store:\n    type: sqlite\n    path: online.db\n"
        "offline_store:\n    type: file\nentity_key_serialization_version: 2\n"
    )
    from feast import FeatureStore

    definitions = {}
    exec(compile(src, "feast/features.py", "exec"), definitions)
    store = FeatureStore(repo_path=str(repo))
    store.apply([definitions["user"], definitions["user_transaction_features"]])
    return store, tmp


@pytest.fixture(scope="module")
def training_rows(offline_store):
    from models.train import load_labeled_transactions, build_historical_training_set
    store, tmp = offline_store
    entity_df = load_labeled_transactions(str(tmp / "bronze"), str(tmp / "labels"))
    return entity_df, build_historical_training_set(store, entity_df).set_index("transaction_id")


def test_no_labeled_row_is_dropped(training_rows):
    entity_df, df = training_rows
    assert len(df) == len(entity_df) == 9
    assert df.loc["u2-0", "fraud"] == 1


def test_point_in_time_excludes_current_and_future(training_rows):
    _, df = training_rows
    # At 12:05:30 Feast must use window [12:00, 12:05): u1-0..u1-4, not u1-5 itself
    assert df.loc["u1-5", "txn_count_5min"] == 5
    assert df.loc["u1-5", "total_amount_5min"] == 500.0
    # At 12:02:30 only u1-0 and u1-1 had happened (window [11:57, 12:02))
    assert df.loc["u1-2", "txn_count_5min"] == 2


def test_rows_without_history_become_zero_like_serving(training_rows):
    from models.train import build_feature_matrix
    from features import build_feature_vector, FEATURE_ORDER
    _, df = training_rows
    X = build_feature_matrix(df.reset_index()).set_index(df.index)
    for tid in ["u1-0", "u2-0", "u3-0"]:  # first txn / outside TTL / user with no windows
        served = build_feature_vector({}, df.loc[tid, "amount"], df.loc[tid, "merchant_country"])
        assert X.loc[tid, FEATURE_ORDER].tolist() == [served[f] for f in FEATURE_ORDER]
    assert X.loc["u2-0", "is_suspicious"] == 1.0  # flag from its own $900 NG amount
