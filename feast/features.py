from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(
    name="user_id",
    description="User making transactions"
)

windowed_features_source = FileSource(
    path="s3://features/windowed_features/",
    timestamp_field="window_start",
    s3_endpoint_override="http://localhost:9000",
)

user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="txn_count_5min",    dtype=Int64),
        Field(name="total_amount_5min", dtype=Float32),
        Field(name="avg_amount_5min",   dtype=Float32),
        Field(name="max_amount_5min",   dtype=Float32),
    ],
    source=windowed_features_source,
    online=True
)
