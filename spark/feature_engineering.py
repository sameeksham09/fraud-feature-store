import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging for this module
logger = logging.getLogger(__name__)

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features import (
    compute_is_foreign,
    compute_is_high_amount,
    compute_is_suspicious,
    HIGH_AMOUNT_THRESHOLD,
    SUSPICIOUS_AMOUNT_THRESHOLD,
)
from spark.redis_writer import write_user_features_to_redis, write_transaction_flags_to_redis
from spark.validation import (
    validate_transaction,
    validate_user_features,
    validate_transaction_flags,
    ValidationError,
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, BooleanType, IntegerType,
)

# ── 1. Create Spark Session ───────────────────────────────────────────────────
JAR_PATH = os.path.join(os.path.dirname(__file__), "jars")
JARS = ",".join([
    f"{JAR_PATH}/spark-sql-kafka.jar",
    f"{JAR_PATH}/kafka-clients.jar",
    f"{JAR_PATH}/spark-token-provider-kafka.jar",
    f"{JAR_PATH}/commons-pool2.jar",
    f"{JAR_PATH}/hadoop-aws.jar",
    f"{JAR_PATH}/aws-java-sdk-bundle.jar"
])

spark = SparkSession.builder \
    .appName("FraudFeatureEngineering") \
    .master("local[*]") \
    .config("spark.jars", JARS) \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")) \
    .config("spark.hadoop.fs.s3a.access.key", os.environ.get("MINIO_ACCESS_KEY", "minioadmin")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("MINIO_SECRET_KEY", "minioadmin")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("✅ Spark session created\n")

# ── 2. Define the Schema of Our Kafka Messages ───────────────────────────────
transaction_schema = StructType([
    StructField("transaction_id",     StringType(),    True),
    StructField("user_id",            StringType(),    True),
    StructField("amount",             DoubleType(),    True),
    StructField("merchant_category",  StringType(),    True),
    StructField("merchant_country",   StringType(),    True),
    StructField("merchant_name",      StringType(),    True),
    StructField("card_last4",         StringType(),    True),
    StructField("timestamp",          StringType(),    True),
    # Note: is_fraud is NOT in the stream — labels come from delayed chargebacks
])

# ── 3. Read the Kafka Stream ──────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", "raw_transactions") \
    .option("startingOffsets", "latest") \
    .load()

print("✅ Connected to Kafka topic: raw_transactions\n")

# ── 4. Parse the Raw Bytes into Structured Columns ───────────────────────────
transactions = raw_stream \
    .select(
        F.from_json(
            F.col("value").cast("string"),
            transaction_schema
        ).alias("data")
    ) \
    .select("data.*") \
    .withColumn("event_time", F.to_timestamp("timestamp"))

# ── 5. Feature 1 — Transaction Count per User (Last 5 Minutes) ───────────────
txn_count_5min = transactions \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(
        F.window("event_time", "5 minutes", "1 minute"),
        F.col("user_id")
    ) \
    .agg(
        F.count("*").alias("txn_count_5min"),
        F.sum("amount").alias("total_amount_5min"),
        F.avg("amount").alias("avg_amount_5min"),
        F.max("amount").alias("max_amount_5min")
    ) \
    .select(
        F.col("user_id"),
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        F.col("txn_count_5min"),
        F.col("total_amount_5min"),
        F.col("avg_amount_5min"),
        F.col("max_amount_5min")
    )

# ── 6. Feature 2 — Transaction Flags (using shared module) ───────────────────
# Create UDFs from shared module functions
compute_is_foreign_udf = F.udf(compute_is_foreign, IntegerType())
compute_is_high_amount_udf = F.udf(compute_is_high_amount, IntegerType())
compute_is_suspicious_udf = F.udf(compute_is_suspicious, IntegerType())

transactions_with_flags = transactions \
    .withColumn(
        "is_foreign",
        compute_is_foreign_udf(F.col("merchant_country"))
    ) \
    .withColumn(
        "is_high_amount",
        compute_is_high_amount_udf(F.col("amount"))
    ) \
    .withColumn(
        "is_suspicious",
        compute_is_suspicious_udf(F.col("merchant_country"), F.col("amount"))
    )

# ── 7. Redis batch writers with validation ────────────────────────────────────
def write_windowed_batch_to_redis(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return
    # A 5-minute window with a 1-minute slide means a single micro-batch can
    # contain multiple overlapping windows for the same user. collect() does
    # not guarantee row order, so writing every row risks an older window
    # overwriting a newer one in Redis. Keep only the most recent window
    # (by window_start) per user before writing.
    latest_per_user = (
        batch_df
        .orderBy(F.col("window_start").desc())
        .dropDuplicates(["user_id"])
    )
    for row in latest_per_user.collect():
        row_dict = row.asDict()
        # Validate before writing to Redis
        is_valid, errors = validate_user_features(row_dict)
        if not is_valid:
            logger.warning(f"Skipping invalid user features: {errors}")
            continue
        write_user_features_to_redis(row_dict)


def write_flags_batch_to_redis(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return
    for row in batch_df.collect():
        row_dict = row.asDict()
        # Validate before writing to Redis
        is_valid, errors = validate_transaction_flags(row_dict)
        if not is_valid:
            logger.warning(f"Skipping invalid transaction flags: {errors}")
            continue
        write_transaction_flags_to_redis(row_dict)

# ── 8. Query 1 — Windowed features → Redis (online store) ───────────────────
query1 = txn_count_5min \
    .writeStream \
    .outputMode("update") \
    .foreachBatch(write_windowed_batch_to_redis) \
    .option("checkpointLocation", "/tmp/checkpoints/windowed_redis") \
    .trigger(processingTime="30 seconds") \
    .queryName("windowed_features_redis") \
    .start()

# ── 9. Query 2 — Transaction flags → Redis (online store) ─────────────────────
query2 = transactions_with_flags \
    .select(
        "transaction_id", "user_id", "amount",
        "merchant_country", "merchant_category",
        "is_foreign", "is_high_amount", "is_suspicious", "event_time"
    ) \
    .writeStream \
    .outputMode("append") \
    .foreachBatch(write_flags_batch_to_redis) \
    .option("checkpointLocation", "/tmp/checkpoints/transaction_flags_redis") \
    .trigger(processingTime="10 seconds") \
    .queryName("transaction_flags_redis") \
    .start()

# ── 10. Query 3 — Windowed features → S3/MinIO (offline store) ───────────────
query3 = txn_count_5min \
    .writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "s3a://features/windowed_features/") \
    .option("checkpointLocation", "/tmp/checkpoints/windowed_s3") \
    .trigger(processingTime="30 seconds") \
    .queryName("windowed_features_s3") \
    .start()

print("🚀 Streaming queries started. Waiting for data...\n")
print("   query1 → windowed features → Redis      (every 30 seconds)")
print("   query2 → transaction flags  → Redis      (every 10 seconds)")
print("   query3 → windowed features → S3/MinIO     (every 30 seconds)")

spark.streams.awaitAnyTermination()
