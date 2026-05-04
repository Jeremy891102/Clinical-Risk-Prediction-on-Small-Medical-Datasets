"""Summarize nested-CV results across all (model, dataset, seed) JSON files.

Usage:
    python scripts/summarize_results.py                      # all seeds
    python scripts/summarize_results.py --seed 42            # one seed
    python scripts/summarize_results.py --results-dir results/raw
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

DATASET_ORDER = ["heart", "pima", "breast_cancer", "liver", "ckd"]
MODEL_ORDER = [
    "logistic_regression", "svm", "random_forest", "mlp",
    "xgboost", "catboost", "tabpfn",
]


def load_rows(results_dir: Path, seed: int | None):
    pattern = f"*_seed{seed}.json" if seed is not None else "*_seed*.json"
    rows = []
    for f in sorted(glob.glob(str(results_dir / pattern))):
        d = json.load(open(f))
        a = d["aggregate"]
        rows.append({
            "dataset": d["dataset"],
            "model": d["model"],
            "seed": d["seed"],
            "auroc_mean": a["roc_auc"]["mean"], "auroc_std": a["roc_auc"]["std"],
            "f1_mean": a["f1"]["mean"], "f1_std": a["f1"]["std"],
            "acc_mean": a["accuracy"]["mean"], "acc_std": a["accuracy"]["std"],
            "brier_mean": a["brier_score"]["mean"], "brier_std": a["brier_score"]["std"],
            "time_sec": d["total_time_sec"],
            "outer_folds": d["outer_folds"],
            "inner_folds": d["inner_folds"],
        })
    rows.sort(key=lambda r: (
        DATASET_ORDER.index(r["dataset"]) if r["dataset"] in DATASET_ORDER else 99,
        MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
        r["seed"],
    ))
    return rows


def print_table(rows, expected_total: int):
    header = (
        f'{"dataset":<14} {"model":<22} {"seed":>5} '
        f'{"AUROC":>14} {"F1":>14} {"Acc":>14} {"Brier":>14} {"time(s)":>9}'
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f'{r["dataset"]:<14} {r["model"]:<22} {r["seed"]:>5} '
            f'{r["auroc_mean"]:.3f}±{r["auroc_std"]:.3f}    '
            f'{r["f1_mean"]:.3f}±{r["f1_std"]:.3f}    '
            f'{r["acc_mean"]:.3f}±{r["acc_std"]:.3f}    '
            f'{r["brier_mean"]:.3f}±{r["brier_std"]:.3f}    '
            f'{r["time_sec"]:>8.1f}'
        )
    print()
    print(f"Total files: {len(rows)}/{expected_total}")
    if rows:
        seeds = sorted({r["seed"] for r in rows})
        print(f"Seeds present: {seeds}")
    missing = expected_total - len(rows)
    if missing > 0:
        present = {(r["dataset"], r["model"], r["seed"]) for r in rows}
        seeds = sorted({r["seed"] for r in rows}) or [None]
        print(f"\nMissing ({missing}):")
        for s in seeds:
            for ds in DATASET_ORDER:
                for md in MODEL_ORDER:
                    if (ds, md, s) not in present:
                        print(f"  - {ds} × {md} (seed {s})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--seed", type=int, default=None,
                    help="filter to a single seed; default = all seeds")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"No such directory: {results_dir}")

    rows = load_rows(results_dir, args.seed)
    n_seeds = len({r["seed"] for r in rows}) or 1
    expected = len(DATASET_ORDER) * len(MODEL_ORDER) * (1 if args.seed is not None else n_seeds)
    print_table(rows, expected_total=expected)


if __name__ == "__main__":
    main()
