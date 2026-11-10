from dataclasses import dataclass

import numpy as np
import polars as pl
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


@dataclass
class TrainResult:
    model: BaseEstimator

    val_log_loss: float
    val_roc_auc: float
    val_ap: float


@dataclass
class EvaluationResult:
    test_log_loss: float
    test_roc_auc: float
    test_ap: float


def train_pipeline(train_df, val_df) -> TrainResult:
    train_features = train_df.select(["TransactionAmt", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]).to_numpy()
    train_labels = train_df.select(["isFraud"]).to_numpy().ravel()
    val_features = val_df.select(["TransactionAmt", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]).to_numpy()
    val_labels = val_df.select(["isFraud"]).to_numpy().ravel()

    model = LogisticRegression(max_iter=5000)
    model.fit(train_features, train_labels)

    val_pred_probs = model.predict(val_features)

    return TrainResult(
        model=model,
        val_log_loss=np.round(log_loss(val_labels, val_pred_probs), 3),
        val_roc_auc=np.round(roc_auc_score(val_labels, val_pred_probs), 3),
        val_ap=np.round(average_precision_score(val_labels, val_pred_probs), 3)
    )


def evaluation(test_df, model):
    test_features = test_df.select(["TransactionAmt", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]).to_numpy()
    test_labels = test_df.select(["isFraud"]).to_numpy().ravel()

    test_pred_probs = model.predict(test_features)

    return EvaluationResult(
        test_log_loss=np.round(log_loss(test_labels, test_pred_probs), 3),
        test_roc_auc=np.round(roc_auc_score(test_labels, test_pred_probs), 3),
        test_ap=np.round(average_precision_score(test_labels, test_pred_probs), 3)
    )

