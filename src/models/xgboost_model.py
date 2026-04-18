import numpy as np
from xgboost import XGBClassifier

from src.models.base import BaseModel


class XGBoostModel(BaseModel):

    def __init__(self, random_state: int = 42, **kwargs):
        defaults = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                        eval_metric="logloss")
        defaults.update(kwargs)
        self.name = "xgboost"
        self.model = XGBClassifier(random_state=random_state, **defaults)

    def fit(self, X, y) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
