"""Task 3: Multi-seed validation on heart, pima (seeds 0, 1, 7).

Why: AUROC std across folds is ~0.05; single-seed (=42) ranks may be fragile.
We re-run the 2 borderline datasets at 3 additional seeds and check whether
the model ordering remains stable. The 3 saturated datasets (breast_cancer,
ckd) are skipped — multi-seed can't change their conclusions. Liver is a
3-way tie at single-seed; also skipped per task spec.

To stay well under the HPC 4-hour session limit, CatBoost and Random Forest
use a *reduced* hyperparameter grid (subset around the seed-42 winners).
LR / SVM / MLP / XGBoost / TabPFN keep the full grid.

Resume-safe: each (dataset, model, seed) writes to its own JSON; if the file
exists the run is skipped. So if the SLURM job is killed at the 4-hour wall
limit, simply re-submitting the same script picks up where it left off.

Outputs:
    results/raw_multi_seed/{dataset}_{model}_seed{s}.json   (per-run details)
    results/multi_seed_heart.csv     (per-fold long: seed, model, fold, auroc, accuracy, f1_weighted, brier)
    results/multi_seed_pima.csv      (same)
    results/multi_seed_summary.csv   (per-(dataset, model): 4-seed mean ± std AUROC vs seed=42)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.nested_cv import run_nested_cv  # noqa: E402

OUT_RAW = ROOT / "results" / "raw_multi_seed"
RESULTS_DIR = ROOT / "results"
SEED42_RAW = ROOT / "results" / "raw"

NEW_SEEDS = [0, 1, 7]
ALL_SEEDS = [42] + NEW_SEEDS
DATASETS = ["heart", "pima"]
MODELS = [
    "logistic_regression", "svm", "random_forest", "mlp",
    "xgboost", "catboost", "tabpfn",
]

# Reduced grids centered on seed=42 most-frequent winners.
# CatBoost full grid: 27 combos; reduced: 8 combos (~3.4x speed-up).
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

# Walltime estimate (seconds) per (dataset, model) for one seed, used only for
# the up-front runtime budget print. Numbers come from seed=42 actuals; CB/RF
# are scaled down by the reduction factor.
EST_SEC = {
    ("heart", "logistic_regression"): 3,
    ("heart", "svm"): 6,
    ("heart", "random_forest"): 19,    # 113 / 6
    ("heart", "mlp"): 9,
    ("heart", "xgboost"): 37,
    ("heart", "catboost"): 149,        # 506 / 3.4
    ("heart", "tabpfn"): 17,
    ("pima",  "logistic_regression"): 4,
    ("pima",  "svm"): 13,
    ("pima",  "random_forest"): 22,    # 133 / 6
    ("pima",  "mlp"): 18,
    ("pima",  "xgboost"): 70,
    ("pima",  "catboost"): 220,        # 747 / 3.4
    ("pima",  "tabpfn"): 12,
}
RUNTIME_CEILING_SEC = 5 * 3600


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def grid_for(model: str) -> tuple[dict | None, str]:
    if model in REDUCED_GRIDS:
        return REDUCED_GRIDS[model], "reduced"
    return None, "full"


def estimate_runtime() -> int:
    """Total seconds for the new seeds (skipped runs not yet known here)."""
    total = 0
    for s in NEW_SEEDS:
        for ds in DATASETS:
            for m in MODELS:
                total += EST_SEC[(ds, m)]
    return total


def run_one(model: str, dataset: str, seed: int):
    out_path = OUT_RAW / f"{dataset}_{model}_seed{seed}.json"
    if out_path.exists():
        print(f"[{stamp()}] [skip] {dataset} x {model} seed={seed} (already done)", flush=True)
        return
    grid, label = grid_for(model)
    print(f"[{stamp()}] === {dataset} x {model} seed={seed} grid={label} ===", flush=True)
    t0 = time.perf_counter()
    try:
        run_nested_cv(
            model_name=model, dataset_name=dataset, seed=seed,
            outer_folds=10, inner_folds=5,
            output_dir=OUT_RAW,
            param_grid_override=grid,
            grid_label=label,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        d = json.load(open(out_path))
        a = d["aggregate"]["roc_auc"]
        print(f"[{stamp()}]   done in {dt:.1f}s  AUROC={a['mean']:.4f}+-{a['std']:.4f}",
              flush=True)
    except Exception as e:
        print(f"[{stamp()}]   FAILED: {type(e).__name__}: {e}", flush=True)


def per_fold_rows(json_path: Path) -> list[dict]:
    """Expand one nested-CV JSON to a list of per-fold dicts with the columns
    requested by the user: seed, model, fold, auroc, accuracy, f1_weighted, brier.
    f1_weighted is recomputed from the saved y_true / y_pred (the JSON only
    stores binary F1 on the positive class)."""
    d = json.load(open(json_path))
    rows = []
    for fr in d["fold_results"]:
        y_true = np.asarray(fr["y_true"])
        y_pred = np.asarray(fr["y_pred"])
        rows.append({
            "seed": d["seed"],
            "model": d["model"],
            "fold": fr["fold"],
            "auroc": fr["metrics"]["roc_auc"],
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "brier": fr["metrics"]["brier_score"],
        })
    return rows


def collect_per_fold(dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    for m in MODELS:
        # seed 42 from results/raw/
        f42 = SEED42_RAW / f"{dataset}_{m}_seed42.json"
        if f42.exists():
            rows.extend(per_fold_rows(f42))
        # new seeds from results/raw_multi_seed/
        for s in NEW_SEEDS:
            f = OUT_RAW / f"{dataset}_{m}_seed{s}.json"
            if f.exists():
                rows.extend(per_fold_rows(f))
    df = pd.DataFrame(rows)
    for c in ("auroc", "accuracy", "f1_weighted", "brier"):
        df[c] = df[c].astype(float).round(4)
    return df


def build_summary(per_fold: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each (dataset, model): mean ± std AUROC across the 4 seeds (where each
    seed contributes its mean-of-folds AUROC), and the seed=42 single value."""
    rows = []
    for ds, df in per_fold.items():
        # mean AUROC per (model, seed) — average over folds first
        per_seed = (df.groupby(["model", "seed"], observed=True)["auroc"]
                      .mean().reset_index())
        for m in MODELS:
            sub = per_seed[per_seed["model"] == m]
            if sub.empty:
                continue
            v_all = sub["auroc"].values
            v_42 = sub.loc[sub["seed"] == 42, "auroc"]
            seed42 = float(v_42.iloc[0]) if len(v_42) else np.nan
            rows.append({
                "dataset": ds,
                "model": m,
                "n_seeds": int(len(v_all)),
                "auroc_seed42": round(seed42, 4),
                "auroc_4seed_mean": round(float(np.mean(v_all)), 4),
                "auroc_4seed_std": round(float(np.std(v_all, ddof=1))
                                          if len(v_all) > 1 else np.nan, 4),
                "delta_vs_seed42": round(float(np.mean(v_all)) - seed42, 4)
                                    if not np.isnan(seed42) else np.nan,
            })
    out = pd.DataFrame(rows)
    # rank-in-dataset by 4-seed mean AUROC (1 = best)
    out["rank_4seed"] = out.groupby("dataset", observed=True)["auroc_4seed_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    out["rank_seed42"] = out.groupby("dataset", observed=True)["auroc_seed42"].rank(
        ascending=False, method="min"
    ).astype(int)
    out["rank_change"] = out["rank_seed42"] - out["rank_4seed"]
    return out.sort_values(["dataset", "rank_4seed"]).reset_index(drop=True)


def main():
    print("STARTING MULTI-SEED", flush=True)
    print(f"[{stamp()}] datasets={DATASETS}  new_seeds={NEW_SEEDS}", flush=True)
    print(f"[{stamp()}] reduced grids: catboost={REDUCED_GRIDS['catboost']}", flush=True)
    print(f"[{stamp()}]                random_forest={REDUCED_GRIDS['random_forest']}", flush=True)

    est = estimate_runtime()
    print(f"[{stamp()}] estimated total runtime (assuming nothing already done): "
          f"{est} sec = {est/60:.1f} min = {est/3600:.2f} h", flush=True)
    if est > RUNTIME_CEILING_SEC:
        print(f"[{stamp()}] estimate exceeds {RUNTIME_CEILING_SEC/3600:.1f}h ceiling — "
              f"aborting and waiting for confirmation.", flush=True)
        sys.exit(2)

    OUT_RAW.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    for s in NEW_SEEDS:
        for ds in DATASETS:
            for m in MODELS:
                run_one(m, ds, s)
    print(f"[{stamp()}] All multi-seed runs done in "
          f"{time.perf_counter() - t_total:.1f}s", flush=True)

    print(f"[{stamp()}] --- Building per-fold tables ---", flush=True)
    per_fold = {}
    for ds in DATASETS:
        df = collect_per_fold(ds)
        out_csv = RESULTS_DIR / f"multi_seed_{ds}.csv"
        df.to_csv(out_csv, index=False)
        per_fold[ds] = df
        print(f"[{stamp()}] saved {out_csv}  rows={len(df)}", flush=True)

    print(f"[{stamp()}] --- Building summary ---", flush=True)
    summary = build_summary(per_fold)
    summary_csv = RESULTS_DIR / "multi_seed_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[{stamp()}] saved {summary_csv}", flush=True)
    print(summary.to_string(index=False), flush=True)

    print("MULTI-SEED DONE", flush=True)


if __name__ == "__main__":
    main()
