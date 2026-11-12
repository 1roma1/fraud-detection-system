import polars as pl


SCHEMA = {
    "user": pl.Int32,
    "card": pl.Int32,
    "year": pl.Int32,
    "month": pl.Int32,
    "day": pl.Int32,
    "time": pl.String,
    "amount": pl.Float32,
    "use_chip": pl.String,
    "merchant_name": pl.Int64,
    "merchant_city": pl.String,
    "merchant_state": pl.String,
    "zip": pl.Int32,
    "mcc": pl.Int32,
    "is_fraud": pl.Int16
}


def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.rename(lambda column: column.lower().replace(" ", "_").replace("?", ""))
        .with_columns(
            pl.col("is_fraud").replace({"No": 0, "Yes": 1}),
            pl.col("amount").str.replace("$", "", literal=True)
        )
        .drop("errors")
        .cast(SCHEMA)
    )


def split_data(df: pl.DataFrame, val_ratio=0.15, test_ratio=0.15, simul_ratio=0.1) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:   
    df_sorted = df.sort(by=["year", "month", "day"])

    total_rows = len(df_sorted)
    val_size = int(total_rows * val_ratio)
    test_size = int(total_rows * test_ratio)
    simul_size = int(total_rows * simul_ratio)
    train_size = total_rows - (val_size + test_size + simul_size)

    train_df = df_sorted.slice(0, train_size)
    val_df = df_sorted.slice(train_size, val_size)
    test_df = df_sorted.slice(train_size + val_size, test_size)
    simul_df = df_sorted.slice(train_size + val_size + test_size, simul_size)

    return train_df, val_df, test_df, simul_df
