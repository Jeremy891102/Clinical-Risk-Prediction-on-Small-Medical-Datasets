import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseModel


class LogisticRegressionModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(penalty="l2", C=1.0, max_iter=1000, solver="lbfgs")
        defaults.update(kwargs)
        self.name = "logistic_regression"
        self.model = LogisticRegression(random_state=random_state, **defaults)

    def fit(self, X, y) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_param_grid(self) -> dict:
        return {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
        }
