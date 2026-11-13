from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from training.config import ExperimentConfig
from training.models.base import BaseModel
from training.models.register import REGISTER


@dataclass
class TrainResult:
    model: BaseModel

    val_log_loss: float
    val_roc_auc: float
    val_ap: float


@dataclass
class EvaluationResult:
    test_log_loss: float
    test_roc_auc: float
    test_ap: float


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
    ) -> TrainResult:
        model = REGISTER[self.config.model_name]()
        model.fit(train_features, train_labels)

        val_pred_probs = model.predict_proba(val_features)

        return TrainResult(
            model=model,
            val_log_loss=np.round(log_loss(val_labels, val_pred_probs), 3),
            val_roc_auc=np.round(roc_auc_score(val_labels, val_pred_probs), 3),
            val_ap=np.round(average_precision_score(val_labels, val_pred_probs), 3),
        )

    def _evaluate(
        self, model: BaseModel, test_features: np.ndarray, test_labels: np.ndarray
    ) -> EvaluationResult:
        test_pred_probs = model.predict_proba(test_features)

        return EvaluationResult(
            test_log_loss=np.round(log_loss(test_labels, test_pred_probs), 3),
            test_roc_auc=np.round(roc_auc_score(test_labels, test_pred_probs), 3),
            test_ap=np.round(average_precision_score(test_labels, test_pred_probs), 3),
        )

    def run_experiment(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
    ):
        train_features = train_df.drop("is_fraud").to_numpy()
        train_labels = train_df["is_fraud"].to_numpy()
        val_features = val_df.drop("is_fraud").to_numpy()
        val_labels = val_df["is_fraud"].to_numpy()
        test_features = test_df.drop("is_fraud").to_numpy()
        test_labels = test_df["is_fraud"].to_numpy()

        train_results = self._train(
            train_features, train_labels, val_features, val_labels
        )
        evaluation_results = self._evaluate(
            train_results.model, test_features, test_labels
        )

        print(train_results)
        print(evaluation_results)

        train_results.model.save(self.config.models_path / "boosting.ubj")
