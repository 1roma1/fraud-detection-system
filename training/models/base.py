from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    @abstractmethod
    def load(self):
        pass
