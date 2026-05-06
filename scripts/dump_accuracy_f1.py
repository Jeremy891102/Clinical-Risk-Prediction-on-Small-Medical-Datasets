"""Dump pooled OOF accuracy + weighted F1 for the seed=42 nested-CV results.

The 10 outer folds of StratifiedKFold partition the dataset, so concatenating
the per-fold y_pred yields one out-of-fold prediction per sample. We compute
accuracy and F1 (average='weighted') on those pooled predictions — no
retraining is needed because each fold's y_true / y_pred is already saved
in results/raw/<dataset>_<model>_seed42.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "accuracy_f1_summary.csv"

DATASETS = ["heart", "pima", "breast_cancer", "liver", "ckd"]
MODELS = ["logistic_regression", "svm", "random_forest", "mlp",
          "xgboost", "catboost", "tabpfn"]


def pooled_oof(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.load(open(path))
    y_true, y_pred = [], []
    for fr in d["fold_results"]:
        y_true.extend(fr["y_true"])
        y_pred.extend(fr["y_pred"])
    return np.asarray(y_true), np.asarray(y_pred)


def main():
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            f = RAW / f"{ds}_{m}_seed42.json"
            if not f.exists():
                print(f"[miss] {f.name}")
                continue
            y_true, y_pred = pooled_oof(f)
            rows.append({
                "dataset": ds,
                "model": m,
                "n_samples": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            })

    df = pd.DataFrame(rows)
    df["accuracy"] = df["accuracy"].round(4)
    df["f1_weighted"] = df["f1_weighted"].round(4)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    print()
    print("=== Accuracy (pooled OOF, seed=42) ===")
    print(df.pivot(index="model", columns="dataset", values="accuracy")
            .reindex(MODELS)[DATASETS])
    print()
    print("=== F1 weighted (pooled OOF, seed=42) ===")
    print(df.pivot(index="model", columns="dataset", values="f1_weighted")
            .reindex(MODELS)[DATASETS])


if __name__ == "__main__":
    main()
