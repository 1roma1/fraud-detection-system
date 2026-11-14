from pathlib import Path

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from training.models.base import BaseModel


class XGBModel(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.model = xgb.XGBClassifier()
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: Path):
        self.model.save_model(path)

    def load():
        pass


class LGBModel(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.model = lgb.LGBMClassifier()
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load():
        pass


class CBModel(BaseModel):
    def __init__(self):
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.model = cb.CatBoostClassifier()
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: Path):
        self.model.save_model(path)

    def load():
        pass
