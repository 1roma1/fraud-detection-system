import polars as pl

from training.config import DataConfig, ExperimentConfig
from training.experiment import Experiment
from training.features import get_features

if __name__ == "__main__":
    data_config = DataConfig()
    exp_config = ExperimentConfig()

    train_df = pl.read_parquet(data_config.train_dataset_path)
    val_df = pl.read_parquet(data_config.val_dataset_path)
    test_df = pl.read_parquet(data_config.test_dataset_path)

    train_df = get_features(train_df)
    val_df = get_features(val_df)
    test_df = get_features(test_df)

    experiment = Experiment(exp_config)
    experiment.run_experiment(train_df, val_df, test_df)
