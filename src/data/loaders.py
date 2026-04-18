import pandas as pd
from pathlib import Path

DATA_CACHE = Path("./data_cache")
DATA_CACHE.mkdir(exist_ok=True)


def _summarize(name: str, X: pd.DataFrame, y: pd.Series):
    print(f"{name}: {len(X)} samples, {X.shape[1]} features, class balance = {y.mean():.2f}")


def load_heart() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Heart Disease (Cleveland) from UCI (id=45). Binarized: y = (num > 0)."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = (dataset.data.targets["num"] > 0).astype(int)

    meta = {
        "name": "Heart Disease (Cleveland)",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "source": "UCI id=45",
    }
    _summarize(meta["name"], X, y)
    return X, y, meta


def load_pima() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Pima Indians Diabetes from OpenML (id=37)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=37, as_frame=True, data_home=str(DATA_CACHE))
    X = data.data.copy()
    y = (data.target == "tested_positive").astype(int)

    meta = {
        "name": "Pima Diabetes",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "source": "OpenML id=37",
    }
    _summarize(meta["name"], X, y)
    return X, y, meta


def load_breast_cancer() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Breast Cancer Wisconsin from sklearn."""
    from sklearn.datasets import load_breast_cancer as _load

    data = _load(as_frame=True)
    X = data.data.copy()
    y = data.target.copy()

    meta = {
        "name": "Breast Cancer Wisconsin",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "source": "sklearn.datasets",
    }
    _summarize(meta["name"], X, y)
    return X, y, meta


def load_liver() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Indian Liver Patient from UCI (id=225). Remap 1/2 -> 1/0."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=225)
    X = dataset.data.features.copy()
    y = (dataset.data.targets.iloc[:, 0] == 1).astype(int)

    meta = {
        "name": "Indian Liver Patient",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "source": "UCI id=225",
    }
    _summarize(meta["name"], X, y)
    return X, y, meta


def load_ckd() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Chronic Kidney Disease from UCI (id=336). Clean whitespace in labels and features."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=336)
    X = dataset.data.features.copy()

    # Clean target: strip whitespace/tabs, map to 0/1
    raw_y = dataset.data.targets["class"].str.strip()
    y = raw_y.map({"ckd": 1, "notckd": 0}).astype(int)

    # Fix numeric columns stored as object due to stray whitespace
    for col in X.columns:
        if X[col].dtype == object:
            converted = pd.to_numeric(X[col], errors="coerce")
            # If most non-null values converted successfully, treat as numeric
            if converted.notna().sum() > X[col].notna().sum() * 0.5:
                X[col] = converted

    meta = {
        "name": "Chronic Kidney Disease",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "source": "UCI id=336",
    }
    _summarize(meta["name"], X, y)
    return X, y, meta
