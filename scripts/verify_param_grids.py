"""Verify each hyperparameter in get_param_grid is actually accepted by the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import make_classification

from src.models.registry import MODEL_REGISTRY, get_model

X, y = make_classification(n_samples=50, n_features=5, random_state=0)

for name in MODEL_REGISTRY:
    model = get_model(name)
    grid = model.get_param_grid()
    print(f"\n{name}: {len(grid)} hyperparameters")
    if not grid:
        print("  (no hyperparameters — skipping)")
        continue

    for param_name, values in grid.items():
        first_val = values[0]
        try:
            test_model = get_model(name, **{param_name: first_val})
            test_model.fit(X, y)
            print(f"  ok  {param_name}={first_val}")
        except Exception as e:
            print(f"  FAIL {param_name}={first_val}: {e}")
