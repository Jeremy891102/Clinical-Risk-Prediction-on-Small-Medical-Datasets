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

    @abstractmethod
    def get_param_grid(self) -> dict:
        """Return hyperparameter grid for GridSearchCV.

        Keys are raw parameter names (no `classifier__` prefix yet — that's
        added in Phase 3.2 when the model is wrapped in a Pipeline). Values
        are lists. Return ``{}`` when there is nothing to tune.
        """
        pass

    def get_estimator(self):
        """Return the underlying sklearn-compatible estimator for use in Pipeline."""
        return self.model
