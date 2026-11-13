import os

import polars as pl

from training.config import DataConfig
from training.data import split_data

if __name__ == "__main__":
    config = DataConfig()
    transaction_df = pl.read_parquet(config.prep_dataset_path)

    train_df, val_df, test_df, simul_df = split_data(transaction_df)

    os.makedirs(config.train_dataset_path.parent, exist_ok=True)
    train_df.write_parquet(config.train_dataset_path)
    val_df.write_parquet(config.val_dataset_path)
    test_df.write_parquet(config.test_dataset_path)
    simul_df.write_parquet(config.simul_dataset_path)
