import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import get_dataset
from src.evaluation.metrics import compute_metrics


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Auto-detect numeric vs categorical columns and build a preprocessor."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def run_single_experiment(model, dataset_name: str, seed: int = 42, test_size: float = 0.2) -> dict:
    """Run a single train/test split experiment."""
    X, y, meta = get_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    t0 = time.time()
    model.fit(X_train_proc, y_train)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    return {
        "dataset": dataset_name,
        "model": getattr(model, "name", type(model).__name__),
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fit_time_sec": round(fit_time, 4),
        **metrics,
    }
