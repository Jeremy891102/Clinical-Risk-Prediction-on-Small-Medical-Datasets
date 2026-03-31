"""
run.py — Main entry point for the project.

Usage:
    python run.py --step all       # run everything
    python run.py --step eval      # just run evaluation
    python run.py --step analysis  # just run analysis
    python run.py --step figures   # just generate figures
"""

import argparse
from src.data_loader import load_all_datasets
from src.evaluate import evaluate_all, save_results, run_xgboost_default
from src.analysis import (
    run_ablation, error_analysis, feature_importance,
    statistical_test, efficiency_analysis
)
from src.visualize import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="DS-GA 1003 Final Project Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "eval", "analysis", "figures"],
                        help="Which step to run")
    args = parser.parse_args()

    if args.step in ["all", "eval"]:
        print("\n" + "=" * 50)
        print("  STEP 1: Loading datasets")
        print("=" * 50)
        datasets = load_all_datasets()

        print("\n" + "=" * 50)
        print("  STEP 2: Evaluating all models")
        print("=" * 50)
        results_df, predictions_dict = evaluate_all(datasets)
        save_results(results_df, predictions_dict)

        print("\n  Running XGBoost default (no tuning) for fair comparison...")
        run_xgboost_default(datasets)

    if args.step in ["all", "analysis"]:
        print("\n" + "=" * 50)
        print("  STEP 3: Running analysis")
        print("=" * 50)
        datasets = load_all_datasets()

        # Load saved results
        import pandas as pd
        import pickle
        results_df = pd.read_csv("results/tables/main_results.csv")
        with open("results/tables/predictions.pkl", "rb") as f:
            predictions_dict = pickle.load(f)

        print("  Running ablation study...")
        run_ablation(datasets)

        print("  Running error analysis...")
        error_analysis(datasets, predictions_dict)

        print("  Running feature importance...")
        feature_importance(datasets)

        print("  Running statistical tests...")
        statistical_test(results_df)

        print("  Running efficiency analysis...")
        efficiency_analysis(results_df)

    if args.step in ["all", "figures"]:
        print("\n" + "=" * 50)
        print("  STEP 4: Generating figures")
        print("=" * 50)
        generate_all_figures()

    print("\n  Done!")


if __name__ == "__main__":
    main()
