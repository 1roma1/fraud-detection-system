import os

from training.config import DataConfig
from training.data import ingest_data, preprocess_data, split_data

if __name__ == "__main__":
    config = DataConfig()

    kaggle_data = ingest_data(config.kaggle_dataset, config.kaggle_file_path)

    os.makedirs(config.raw_dataset_path.parent, exist_ok=True)
    kaggle_data.write_parquet(config.raw_dataset_path)

    prep_data = preprocess_data(kaggle_data)
    os.makedirs(config.prep_dataset_path.parent, exist_ok=True)
    prep_data.write_parquet(config.prep_dataset_path)

    train_data, val_data, test_data, simul_data = split_data(prep_data)

    os.makedirs(config.train_dataset_path.parent, exist_ok=True)
    train_data.write_parquet(config.train_dataset_path)
    val_data.write_parquet(config.val_dataset_path)
    test_data.write_parquet(config.test_dataset_path)
    simul_data.write_parquet(config.simul_dataset_path)
