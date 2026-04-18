import numpy as np

from src.models.base import BaseModel


class TabPFNModel(BaseModel):

    def __init__(self, random_state: int = 42, device: str | None = None, **kwargs):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise ImportError(
                "TabPFN is not installed. Install with: pip install tabpfn torch"
            )

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: TabPFN running on CPU — this will be slow.")

        self.name = "tabpfn"
        # TODO: Verify we are loading the pure-synthetic TabPFN-2.5 checkpoint,
        # not Real-TabPFN-2.5 (which is fine-tuned on real-world data and could
        # cause contamination concerns). Check TabPFN docs for the correct
        # model parameter to specify the non-fine-tuned variant.
        defaults = dict(device=device)
        defaults.update(kwargs)
        self.model = TabPFNClassifier(**defaults)

    def fit(self, X, y) -> "TabPFNModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)
