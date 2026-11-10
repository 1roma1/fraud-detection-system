import polars as pl


def split_data(df: pl.DataFrame, val_ratio=0.15, test_ratio=0.15, simul_ratio=0.1) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:   
    df_sorted = df.sort("TransactionDT")

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
