"""Run all 7 models on all 5 datasets and save results."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.registry import DATASET_REGISTRY
from src.models.registry import MODEL_REGISTRY
from src.evaluation.pipeline import run_single_experiment


def main():
    results = []
    datasets = list(DATASET_REGISTRY.keys())
    models = list(MODEL_REGISTRY.keys())

    for dataset_name in datasets:
        for model_name in models:
            print(f"\n--- {model_name} on {dataset_name} ---")
            try:
                model = MODEL_REGISTRY[model_name]()
                result = run_single_experiment(model, dataset_name, seed=42)
                result["status"] = "ok"
                result["error_message"] = ""
                results.append(result)
                print(f"  AUC={result['roc_auc']:.3f}  F1={result['f1']:.3f}  "
                      f"time={result['fit_time_sec']:.2f}s")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": 42,
                    "status": "failed",
                    "error_message": str(e),
                })

    df = pd.DataFrame(results)
    out_path = Path("results/phase2_smoke_test.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Pivot table of AUROC
    ok = df[df["status"] == "ok"]
    if not ok.empty:
        pivot = ok.pivot_table(values="roc_auc", index="dataset", columns="model")
        print("\n=== AUROC by Dataset x Model ===")
        print(pivot.to_string(float_format="%.3f"))

    failed = df[df["status"] == "failed"]
    if not failed.empty:
        print(f"\n=== {len(failed)} Failed Runs ===")
        print(failed[["dataset", "model", "error_message"]].to_string(index=False))


if __name__ == "__main__":
    main()
