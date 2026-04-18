import numpy as np
from catboost import CatBoostClassifier

from src.models.base import BaseModel


class CatBoostModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(iterations=200, depth=6, learning_rate=0.1, verbose=False)
        defaults.update(kwargs)
        self.name = "catboost"
        self.model = CatBoostClassifier(random_seed=random_state, **defaults)

    def fit(self, X, y) -> "CatBoostModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X).flatten()

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
