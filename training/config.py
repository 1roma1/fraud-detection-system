from pathlib import Path
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dataset_path: Path = Path("data/raw/raw_transactions.csv")
    prep_dataset_path: Path = Path("data/prep/prep_transactions.parquet")
    train_dataset_path: Path = Path("data/split/train_transactions.parquet")
    val_dataset_path: Path = Path("data/split/val_transactions.parquet")
    test_dataset_path: Path = Path("data/split/test_transactions.parquet")
    simul_dataset_path: Path = Path("data/split/simul_transactions.parquet")

    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    simul_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
