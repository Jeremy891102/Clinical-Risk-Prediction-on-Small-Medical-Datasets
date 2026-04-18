"""Phase 1 verification: check all dataset loaders produce correct labels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.registry import DATASET_REGISTRY

for name, loader in DATASET_REGISTRY.items():
    print("=" * 60)
    print(f"Dataset: {name}")
    print("=" * 60)

    X, y, meta = loader()

    print(f"  X shape:          {X.shape}")
    print(f"  y.unique():       {sorted(y.unique())}")
    print(f"  y.dtype:          {y.dtype}")
    print(f"  y.mean():         {y.mean():.4f}")
    print(f"  y.value_counts(): {y.value_counts().to_dict()}")
    print(f"  y.isna().sum():   {y.isna().sum()}")
    print(f"  len(X)==len(y):   {len(X) == len(y)}")

    num_cols = X.select_dtypes(include="number").shape[1]
    obj_cols = X.select_dtypes(exclude="number").shape[1]
    print(f"  X numeric cols:   {num_cols}")
    print(f"  X object cols:    {obj_cols}")
    print(f"  Rows w/ any NaN:  {X.isna().any(axis=1).sum()}")
    print()
