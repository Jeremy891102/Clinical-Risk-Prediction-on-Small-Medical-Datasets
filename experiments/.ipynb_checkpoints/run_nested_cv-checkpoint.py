"""Run nested CV. Examples:
    python experiments/run_nested_cv.py --model all --dataset all
    python experiments/run_nested_cv.py --model logistic_regression --dataset heart
    python experiments/run_nested_cv.py --model xgboost --dataset all --seed 42
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.registry import DATASET_REGISTRY
from src.evaluation.nested_cv import run_nested_cv
from src.models.registry import MODEL_REGISTRY


def _resolve(arg: str, registry: dict) -> list[str]:
    if arg == "all":
        return list(registry.keys())
    if arg not in registry:
        raise SystemExit(f"Unknown name: {arg}. Available: {list(registry.keys())} or 'all'")
    return [arg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name or 'all'")
    ap.add_argument("--dataset", required=True, help="dataset name or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer-folds", type=int, default=10)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--output-dir", default="results/raw")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-run even if output file already exists")
    args = ap.parse_args()

    models = _resolve(args.model, MODEL_REGISTRY)
    datasets = _resolve(args.dataset, DATASET_REGISTRY)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[tuple[str, str, str, str]] = []
    for ds in datasets:
        for m in models:
            tag = f"{ds} × {m} (seed {args.seed})"
            out_path = out_dir / f"{ds}_{m}_seed{args.seed}.json"
            if out_path.exists() and not args.no_skip_existing:
                print(f"\n[skip] {tag} — {out_path} exists")
                summary.append((ds, m, "skipped", "—"))
                continue
            print(f"\n=== {tag} ===")
            try:
                res = run_nested_cv(
                    model_name=m, dataset_name=ds, seed=args.seed,
                    outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                    output_dir=out_dir,
                )
                a = res["aggregate"]["roc_auc"]
                summary.append((ds, m, "ok",
                                f"AUROC={a['mean']:.3f}±{a['std']:.3f}  "
                                f"({res['total_time_sec']:.1f}s)"))
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                summary.append((ds, m, "failed", str(e)[:80]))

    print("\n=== Summary ===")
    for row in summary:
        print(f"  {row[0]:>14}  {row[1]:>20}  {row[2]:>8}  {row[3]}")


if __name__ == "__main__":
    main()
