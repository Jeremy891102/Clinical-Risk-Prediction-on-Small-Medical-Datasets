"""TabPFN wrapper.

Checkpoint selection (contamination-sensitive):
We pin TabPFN v2.5's pure-synthetic default checkpoint
(`tabpfn-v2.5-classifier-v2.5_default.ckpt`) via `ModelVersion.V2_5`.
The v2.5 family also ships `..._real.ckpt` (Real-TabPFN-2.5), which is
fine-tuned on 43 OpenML/Kaggle real-world datasets and would create a
contamination risk for our medical-tabular benchmarks.

Filename conventions across versions (for reference):
  - `tabpfn-v2-classifier.ckpt`                 <- pure synthetic v2
  - `tabpfn-v2-classifier-finetuned-*.ckpt`     <- fine-tuned v2 variants
  - `tabpfn-v2.5-classifier-v2.5_default.ckpt`  <- pure synthetic v2.5 (this one)
  - `tabpfn-v2.5-classifier-v2.5_real.ckpt`     <- Real-TabPFN-2.5 (fine-tuned)
  - `tabpfn-v2.6-classifier-v2.6_default.ckpt`  <- v2.6 default (package auto)

v2.5+ weights require a Prior Labs license token (cached via
`tabpfn.browser_auth.save_token` or `TABPFN_TOKEN` env var). The package's
auto default (`TabPFNClassifier()` with no args) selects v2.6 in
tabpfn==7.1.1, so we explicitly pin v2.5 here for reproducibility and
to keep the synthetic-vs-real distinction crystal clear in filenames.
See https://github.com/PriorLabs/TabPFN.
"""

import numpy as np

from src.models.base import BaseModel


class TabPFNModel(BaseModel):
    name = "tabpfn"

    def __init__(self, random_state: int = 42, device: str | None = None, **kwargs):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as e:
            raise ImportError(
                "TabPFN is not installed. Install with: pip install tabpfn torch"
            ) from e

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: TabPFN running on CPU — inference will be slow.")

        self.model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            device=device,
            random_state=random_state,
            ignore_pretraining_limits=True,
            **kwargs,
        )
        print(f"TabPFN checkpoint: {self.model.model_path}")

    @staticmethod
    def _to_numpy(X):
        return X.values if hasattr(X, "values") else X

    def fit(self, X, y) -> "TabPFNModel":
        self.model.fit(self._to_numpy(X), self._to_numpy(y))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._to_numpy(X))

    def get_param_grid(self) -> dict:
        # TabPFN is a foundation model trained for in-context tabular
        # classification. The default n_estimators (ensemble size) could
        # in principle be tuned, but we follow the original paper's
        # recommendation to use the default settings; tuning would obscure
        # fair comparison with the other (tuned) baselines.
        return {}
