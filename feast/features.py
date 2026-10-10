from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(
    name="user_id",
    description="User making transactions"
)

# Silver Parquet written by spark/feature_engineering.py (query3). It holds
# every 5-minute window sliding by 1 minute, emitted once the watermark closes it.
#
# timestamp_field is window_END, not window_start: at training time t, Feast
# picks the latest window with window_end <= t, i.e. the trailing 5 minutes
# that had fully happened before the transaction. Using window_start would
# join a window that is still open at t and contains FUTURE transactions
# (label/feature leakage).
windowed_features_source = FileSource(
    path="s3://features/windowed_features/",
    timestamp_field="window_end",
    s3_endpoint_override="http://localhost:9000",
)

user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    # A window ends every minute while a user is active, and the last one ends
    # 5 minutes after their last transaction. A short TTL makes inactive users
    # read as "no recent activity" (features -> 0) instead of reusing a stale
    # window from hours ago. Online, Redis keys expire after 5 minutes for the
    # same reason (spark/redis_writer.py).
    ttl=timedelta(minutes=2),
    schema=[
        Field(name="txn_count_5min",    dtype=Int64),
        Field(name="total_amount_5min", dtype=Float32),
        Field(name="avg_amount_5min",   dtype=Float32),
        Field(name="max_amount_5min",   dtype=Float32),
    ],
    source=windowed_features_source,
    online=True
)
