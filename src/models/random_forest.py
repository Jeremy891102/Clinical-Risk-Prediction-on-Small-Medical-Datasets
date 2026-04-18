import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(n_estimators=200, max_depth=None, min_samples_split=2, n_jobs=-1)
        defaults.update(kwargs)
        self.name = "random_forest"
        self.model = RandomForestClassifier(random_state=random_state, **defaults)

    def fit(self, X, y) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
