"""
Tests for spark/windowing.py — which sliding window gets written to Redis.

Runs a real local SparkSession; skipped automatically if Java/Spark can't start.
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession, functions as F  # noqa: E402

from spark.windowing import select_trailing_window  # noqa: E402

BASE = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture(scope="module")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("test_windowing")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as exc:  # no Java, etc.
        pytest.skip(f"Spark unavailable: {exc}")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _windowed(spark, rows):
    """Same aggregation as spark/feature_engineering.py (5m window, 1m slide)."""
    df = spark.createDataFrame(rows, ["user_id", "amount", "event_time"])
    return (
        df.groupBy(F.window("event_time", "5 minutes", "1 minute"), "user_id")
        .agg(
            F.count("*").alias("txn_count_5min"),
            F.sum("amount").alias("total_amount_5min"),
            F.max("amount").alias("max_amount_5min"),
        )
        .select(
            "user_id",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "txn_count_5min",
            "total_amount_5min",
            "max_amount_5min",
        )
    )


def _one_txn_per_minute(user="user_1", minutes=5, amount=100.0):
    return [(user, amount, BASE + timedelta(minutes=m, seconds=30)) for m in range(minutes)]


def test_old_logic_only_covered_last_minute(spark):
    """Regression guard: documents the bug the fix addresses."""
    agg = _windowed(spark, _one_txn_per_minute())
    old = agg.orderBy(F.col("window_start").desc()).dropDuplicates(["user_id"]).collect()[0]
    assert old["txn_count_5min"] == 1  # 5 txns happened, old logic saw 1


def test_selects_window_covering_trailing_five_minutes(spark):
    rows = select_trailing_window(_windowed(spark, _one_txn_per_minute())).collect()
    assert len(rows) == 1
    r = rows[0]
    assert r["txn_count_5min"] == 5
    assert r["total_amount_5min"] == 500.0
    # [12:00, 12:05) contains the latest event at 12:04:30
    assert r["window_start"] == BASE
    assert r["window_end"] == BASE + timedelta(minutes=5)


def test_excludes_activity_older_than_window(spark):
    # 7 minutes of activity: only the last 5 should count
    rows = select_trailing_window(_windowed(spark, _one_txn_per_minute(minutes=7))).collect()
    assert rows[0]["txn_count_5min"] == 5


def test_one_row_per_user_and_users_independent(spark):
    data = _one_txn_per_minute("user_1", 5) + [
        ("user_2", 900.0, BASE + timedelta(minutes=2, seconds=10)),
    ]
    rows = {r["user_id"]: r for r in select_trailing_window(_windowed(spark, data)).collect()}
    assert set(rows) == {"user_1", "user_2"}
    assert rows["user_1"]["txn_count_5min"] == 5
    assert rows["user_2"]["txn_count_5min"] == 1
    assert rows["user_2"]["max_amount_5min"] == 900.0


def test_preserves_columns(spark):
    agg = _windowed(spark, _one_txn_per_minute())
    assert select_trailing_window(agg).columns == agg.columns
