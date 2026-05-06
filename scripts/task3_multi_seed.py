"""Task 3: Multi-seed validation on heart, pima (seeds 0, 1, 7).

Why: AUROC std across folds is ~0.05; single-seed (=42) ranks may be fragile.
We re-run the 2 borderline datasets at 3 additional seeds and check whether
the model ordering remains stable. The 3 saturated datasets (breast_cancer,
ckd) are skipped — multi-seed can't change their conclusions. Liver is a
3-way tie at single-seed (TabPFN/CatBoost/RF within 0.002 AUROC); also
skipped per task spec.

To stay under 30 minutes wall-time on a single machine, CatBoost and Random
Forest use a *reduced* hyperparameter grid (subset around the seed-42 winners).
LR / SVM / MLP / XGBoost / TabPFN keep the full grid.

Outputs:
    results/raw_multi_seed/{dataset}_{model}_seed{s}.json   (per-run details)
    results/tables/multi_seed_heart.csv
    results/tables/multi_seed_pima.csv
    results/tables/multi_seed_rank_comparison.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.nested_cv import run_nested_cv  # noqa: E402

OUT_RAW = ROOT / "results" / "raw_multi_seed"
TABLES_DIR = ROOT / "results" / "tables"
SEED42_RAW = ROOT / "results" / "raw"

NEW_SEEDS = [0, 1, 7]
DATASETS = ["heart", "pima"]
MODELS = [
    "logistic_regression", "svm", "random_forest", "mlp",
    "xgboost", "catboost", "tabpfn",
]

# Reduced grids (centered on the seed-42 most-frequent winners).
# CatBoost full grid: 27 combos; reduced: 4 combos (~7x speed-up).
# Random Forest full grid: 24 combos; reduced: 4 combos (~6x speed-up).
REDUCED_GRIDS = {
    "catboost": {
        "iterations": [100, 300],
        "depth": [4, 6],
        "learning_rate": [0.03, 0.1],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [None, 5],
        "min_samples_split": [2],
    },
}


def grid_for(model: str) -> tuple[dict | None, str]:
    if model in REDUCED_GRIDS:
        return REDUCED_GRIDS[model], "reduced"
    return None, "full"


def run_one(model: str, dataset: str, seed: int):
    out_path = OUT_RAW / f"{dataset}_{model}_seed{seed}.json"
    if out_path.exists():
        print(f"[skip] {dataset}×{model} seed={seed} — exists")
        return
    grid, label = grid_for(model)
    print(f"\n=== {dataset} × {model} (seed {seed}, grid={label}) ===")
    t0 = time.perf_counter()
    try:
        run_nested_cv(
            model_name=model, dataset_name=dataset, seed=seed,
            outer_folds=10, inner_folds=5,
            output_dir=OUT_RAW,
            param_grid_override=grid,
            grid_label=label,
            verbose=False,  # quiet for the long sweep
        )
        dt = time.perf_counter() - t0
        # quick AUROC summary by re-reading the saved JSON
        d = json.load(open(out_path))
        a = d["aggregate"]["roc_auc"]
        print(f"  done in {dt:.1f}s — AUROC = {a['mean']:.4f}±{a['std']:.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


def collect(dataset: str) -> pd.DataFrame:
    """Pull all seeds (single-seed=42 + multi-seed=0/1/7) into a long DataFrame."""
    rows = []
    # seed 42 from results/raw/
    for m in MODELS:
        f = SEED42_RAW / f"{dataset}_{m}_seed42.json"
        if f.exists():
            d = json.load(open(f))
            a = d["aggregate"]
            rows.append({
                "dataset": dataset, "model": m, "seed": 42,
                "grid_label": d.get("grid_label", "full"),
                "auroc": a["roc_auc"]["mean"], "auroc_std_folds": a["roc_auc"]["std"],
                "f1": a["f1"]["mean"], "brier": a["brier_score"]["mean"],
                "total_time_sec": d["total_time_sec"],
            })
    # new seeds from raw_multi_seed/
    for s in NEW_SEEDS:
        for m in MODELS:
            f = OUT_RAW / f"{dataset}_{m}_seed{s}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            a = d["aggregate"]
            rows.append({
                "dataset": dataset, "model": m, "seed": s,
                "grid_label": d.get("grid_label", "full"),
                "auroc": a["roc_auc"]["mean"], "auroc_std_folds": a["roc_auc"]["std"],
                "f1": a["f1"]["mean"], "brier": a["brier_score"]["mean"],
                "total_time_sec": d["total_time_sec"],
            })
    return pd.DataFrame(rows)


def aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds: mean ± std of (per-seed mean AUROC)."""
    g = long_df.groupby(["dataset", "model", "grid_label"], observed=True)
    agg = g["auroc"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "auroc_seed_mean", "std": "auroc_seed_std",
                              "count": "n_seeds"})
    # rank within dataset by mean AUROC (1 = best)
    agg["rank"] = agg.groupby("dataset", observed=True)["auroc_seed_mean"].rank(
        ascending=False, method="min"
    )
    return agg.sort_values(["dataset", "rank"]).reset_index(drop=True)


def rank_comparison(long_df: pd.DataFrame) -> pd.DataFrame:
    """Single-seed (42) rank vs multi-seed mean rank, per dataset."""
    rows = []
    for ds in DATASETS:
        sub = long_df[long_df["dataset"] == ds].copy()
        # Ranks per seed
        sub["rank_in_seed"] = sub.groupby("seed", observed=True)["auroc"].rank(
            ascending=False, method="min"
        )
        single = sub[sub["seed"] == 42][["model", "auroc", "rank_in_seed"]].rename(
            columns={"auroc": "auroc_seed42", "rank_in_seed": "rank_seed42"}
        )
        multi = (sub.groupby("model", observed=True)
                    .agg(auroc_mean=("auroc", "mean"),
                         auroc_std=("auroc", "std"),
                         mean_rank=("rank_in_seed", "mean"),
                         n_seeds=("seed", "nunique"))
                    .reset_index())
        merged = single.merge(multi, on="model")
        merged.insert(0, "dataset", ds)
        merged["rank_change"] = (merged["rank_seed42"] - merged["mean_rank"]).round(2)
        merged = merged.sort_values("mean_rank").reset_index(drop=True)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def warn_on_flips(rank_cmp: pd.DataFrame, threshold: float = 1.5):
    """Print a warning for any model whose seed-42 rank differs from the
    multi-seed mean rank by more than `threshold` positions."""
    flips = rank_cmp[abs(rank_cmp["rank_change"]) >= threshold]
    if flips.empty:
        print("\nRank stability: no flips ≥ "
              f"{threshold} positions between seed=42 and the 4-seed mean.")
        return
    print(f"\n⚠️  Rank flips ≥ {threshold} positions detected (single-seed vs 4-seed mean):")
    print(flips.to_string(index=False))


def main():
    print(">>> Task 3: Multi-seed validation on heart, pima (seeds 0, 1, 7)")
    print(f"Reduced grids: catboost {REDUCED_GRIDS['catboost']}")
    print(f"               random_forest {REDUCED_GRIDS['random_forest']}")
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    for s in NEW_SEEDS:
        for ds in DATASETS:
            for m in MODELS:
                run_one(m, ds, s)
    print(f"\nAll multi-seed runs done in {time.perf_counter() - t_total:.1f}s")

    print("\n--- Aggregating ---")
    for ds in DATASETS:
        long_df = collect(ds)
        out_csv = TABLES_DIR / f"multi_seed_{ds}.csv"
        # round numeric columns to 4 dp
        rounded = long_df.copy()
        for c in ["auroc", "auroc_std_folds", "f1", "brier", "total_time_sec"]:
            rounded[c] = rounded[c].astype(float).round(4)
        rounded.to_csv(out_csv, index=False)
        print(f"\nLong table for {ds} (saved {out_csv}):")
        print(rounded.to_string(index=False))

    print("\n--- Rank comparison: seed=42 vs 4-seed mean ---")
    big_long = pd.concat([collect(ds) for ds in DATASETS], ignore_index=True)
    rank_cmp = rank_comparison(big_long)
    cmp_round = rank_cmp.copy()
    for c in ["auroc_seed42", "auroc_mean", "auroc_std", "mean_rank"]:
        cmp_round[c] = cmp_round[c].astype(float).round(4)
    cmp_csv = TABLES_DIR / "multi_seed_rank_comparison.csv"
    cmp_round.to_csv(cmp_csv, index=False)
    print(cmp_round.to_string(index=False))
    print(f"\nSaved: {cmp_csv}")

    warn_on_flips(rank_cmp, threshold=1.5)

    print("\nTASK 3 DONE")


if __name__ == "__main__":
    main()
