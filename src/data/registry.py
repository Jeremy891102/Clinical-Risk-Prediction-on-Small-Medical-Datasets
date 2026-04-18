from src.data.loaders import load_heart, load_pima, load_breast_cancer, load_liver, load_ckd

DATASET_REGISTRY = {
    "heart": load_heart,
    "pima": load_pima,
    "breast_cancer": load_breast_cancer,
    "liver": load_liver,
    "ckd": load_ckd,
}


def get_dataset(name: str):
    """Returns (X, y, meta)."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]()
