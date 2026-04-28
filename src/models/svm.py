import numpy as np
from sklearn.svm import SVC

from src.models.base import BaseModel


class SVMModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(kernel="rbf", C=1.0, gamma="scale", probability=True)
        defaults.update(kwargs)
        self.name = "svm"
        self.model = SVC(random_state=random_state, **defaults)

    def fit(self, X, y) -> "SVMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_param_grid(self) -> dict:
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.01, 0.1, 1.0],
            "kernel": ["rbf"],
        }
