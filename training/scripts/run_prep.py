import os
from pathlib import Path

import polars as pl

from training.data import preprocess_data


RAW_TRANSACTION_DATA = Path("data") / "raw" / "credit_card_transactions-ibm_v2.csv"
PREP_DIR = Path("data") / "prep"


if __name__ == "__main__":
    transaction_df = pl.read_csv(RAW_TRANSACTION_DATA)

    prep_df = preprocess_data(transaction_df)

    os.makedirs(PREP_DIR, exist_ok=True)
    prep_df.write_parquet(PREP_DIR / "prep_transaction.parquet")
