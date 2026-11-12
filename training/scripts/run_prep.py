import os
import polars as pl

from training.data import preprocess_data
from training.config import DataConfig


if __name__ == "__main__":
    config = DataConfig()
    transaction_df = pl.read_csv(config.raw_dataset_path)

    prep_df = preprocess_data(transaction_df)

    os.makedirs(config.prep_dataset_path.parent, exist_ok=True)
    prep_df.write_parquet(config.prep_dataset_path)
