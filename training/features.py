import polars as pl


def get_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.from_epoch("timestamp", time_unit="ms").dt.hour().alias("hour"),
        pl.from_epoch("timestamp", time_unit="ms").dt.weekday().alias("weekday"),
        pl.from_epoch("timestamp", time_unit="ms").dt.month().alias("month"),
        "amount",
        "is_fraud",
    )
