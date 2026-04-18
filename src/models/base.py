from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """All models must implement this interface so the pipeline can treat them uniformly."""

    name: str

    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass
