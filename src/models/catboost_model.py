import numpy as np
from catboost import CatBoostClassifier

from src.models.base import BaseModel


class CatBoostModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(
            iterations=200, depth=6, learning_rate=0.1, verbose=False,
            thread_count=1,  # avoid OOM: GridSearchCV n_jobs=-1 already parallelizes
            allow_writing_files=False,  # don't litter catboost_info/
        )
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

    def get_param_grid(self) -> dict:
        return {
            "iterations": [100, 300, 500],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.3],
        }
