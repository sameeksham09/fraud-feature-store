"""
Window selection for the online store.

The streaming job aggregates with a 5-minute window sliding every 1 minute, so
every event lands in 5 overlapping windows. In `update` output mode a
micro-batch emits all windows a user's events touched, and only one of them
should be written to Redis as "the user's last 5 minutes".

Picking the window with the latest `window_start` is wrong: that window has
only just opened, so it covers roughly the last minute of activity, not five.

The correct choice is the *earliest-starting* window that still contains the
user's most recent activity, i.e. the oldest window whose end is after the
latest window_start. With a 5m/1m window that's [T-4m, T+1m) where T is the
minute the latest event falls in: a trailing 4-5 minute view that includes the
newest event.

Kept in its own module (no SparkSession at import) so it can be unit tested.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def select_trailing_window(batch_df: DataFrame) -> DataFrame:
    """
    For each user_id, keep the single window that best represents the
    trailing window ending at the user's most recent activity.

    Expects columns: user_id, window_start, window_end (plus any aggregates).
    Returns the same columns, one row per user_id.
    """
    by_user = Window.partitionBy("user_id")
    by_user_oldest_first = by_user.orderBy(F.col("window_start").asc())

    return (
        batch_df
        .withColumn("_latest_start", F.max("window_start").over(by_user))
        # Windows that contain the user's latest activity minute
        .filter(F.col("window_end") > F.col("_latest_start"))
        # ...and of those, the one that reaches furthest back
        .withColumn("_rn", F.row_number().over(by_user_oldest_first))
        .filter(F.col("_rn") == 1)
        .drop("_latest_start", "_rn")
    )
