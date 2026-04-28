import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base import BaseModel


class MLPModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(hidden_layer_sizes=(64, 32), activation="relu",
                        max_iter=500, early_stopping=True)
        defaults.update(kwargs)
        self.name = "mlp"
        self.model = MLPClassifier(random_state=random_state, **defaults)

    def fit(self, X, y) -> "MLPModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_param_grid(self) -> dict:
        return {
            "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        }
