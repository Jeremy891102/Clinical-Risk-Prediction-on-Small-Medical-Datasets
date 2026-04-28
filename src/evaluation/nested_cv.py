"""Nested cross-validation: outer K-fold for unbiased estimates, inner GridSearchCV for HPO.

Preprocessing is fit inside each outer training fold (via Pipeline) so no
information from the outer test fold or inner validation folds leaks in.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import get_dataset
from src.evaluation.metrics import compute_metrics
from src.models.registry import get_model


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _aggregate(fold_results: list[dict]) -> dict:
    metric_names = fold_results[0]["metrics"].keys()
    out: dict = {}
    for m in metric_names:
        vals = [fr["metrics"][m] for fr in fold_results]
        out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    return out


def run_nested_cv(
    model_name: str,
    dataset_name: str,
    seed: int = 42,
    outer_folds: int = 10,
    inner_folds: int = 5,
    output_dir: Path = Path("results/raw"),
    verbose: bool = True,
) -> dict:
    """Nested CV for one (model, dataset) pair. Saves JSON; returns the same dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}_{model_name}_seed{seed}.json"

    X, y, _meta = get_dataset(dataset_name)
    y = np.asarray(y)

    wrapper = get_model(model_name)
    estimator = wrapper.get_estimator()
    raw_grid = wrapper.get_param_grid()
    param_grid = {f"classifier__{k}": v for k, v in raw_grid.items()}

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    fold_results: list[dict] = []
    t_start = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("preprocessor", _build_preprocessor(X_tr)),
            ("classifier", estimator),
        ])

        t0 = time.perf_counter()
        if param_grid:
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            search = GridSearchCV(
                pipe, param_grid, cv=inner,
                scoring="roc_auc", n_jobs=-1, refit=True,
            )
            search.fit(X_tr, y_tr)
            fitted = search.best_estimator_
            best_params = {k: v for k, v in search.best_params_.items()}
        else:
            pipe.fit(X_tr, y_tr)
            fitted = pipe
            best_params = {}
        fit_time = time.perf_counter() - t0

        y_pred = fitted.predict(X_te)
        y_proba = fitted.predict_proba(X_te)
        metrics = compute_metrics(y_te, y_pred, y_proba)

        fold_results.append({
            "fold": fold,
            "best_params": best_params,
            "metrics": metrics,
            "fit_time_sec": round(fit_time, 4),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_indices": test_idx.tolist(),
            "y_true": y_te.tolist(),
            "y_pred": np.asarray(y_pred).astype(int).tolist(),
            "y_proba": y_proba[:, 1].tolist(),
        })
        if verbose:
            print(f"  fold {fold}: AUROC={metrics['roc_auc']:.3f}  "
                  f"F1={metrics['f1']:.3f}  fit={fit_time:.1f}s  "
                  f"params={best_params}")

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "fold_results": fold_results,
        "aggregate": _aggregate(fold_results),
        "total_time_sec": round(time.perf_counter() - t_start, 2),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, default=_json_default, indent=2)
    if verbose:
        print(f"Saved {out_path}")
    return result
