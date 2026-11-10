import os
import joblib
from pathlib import Path

import polars as pl

from training.train import train_pipeline, evaluation


SPLIT_DIR = Path("data") / "split"
TRAIN_DATA = SPLIT_DIR / "train_transaction.parquet"
VAL_DATA = SPLIT_DIR / "val_transaction.parquet"
TEST_DATA = SPLIT_DIR / "test_transaction.parquet"

MODEL_DIR = Path("models")


if __name__ == "__main__":
    train_df = pl.read_parquet(TRAIN_DATA)
    val_df = pl.read_parquet(VAL_DATA)
    test_df = pl.read_parquet(TEST_DATA)

    train_result = train_pipeline(train_df, val_df)
    eval_result = evaluation(test_df, train_result.model)
    
    print(train_result)
    print(eval_result)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(train_result.model, MODEL_DIR / "logreg.pk")
