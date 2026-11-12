import os
from pathlib import Path

import polars as pl

from training.data import split_data


TRANSACTION_DATA = Path("data") / "prep" / "prep_transaction.parquet"
SPLIT_DIR = Path("data") / "split"


if __name__ == "__main__":
    transaction_df = pl.read_parquet(TRANSACTION_DATA)

    train_df, val_df, test_df, simul_df = split_data(transaction_df)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_df.write_parquet(SPLIT_DIR / "train_transaction.parquet")
    val_df.write_parquet(SPLIT_DIR / "val_transaction.parquet")
    test_df.write_parquet(SPLIT_DIR / "test_transaction.parquet")
    simul_df.write_parquet(SPLIT_DIR / "simul_transaction.parquet")
