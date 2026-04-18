"""Smoke test: load all datasets and run DummyClassifier on each."""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.dummy import DummyClassifier

from src.data.registry import DATASET_REGISTRY
from src.evaluation.pipeline import run_single_experiment


def main():
    print("=" * 60)
    print("Loading all datasets...")
    print("=" * 60)

    results = []
    for name in DATASET_REGISTRY:
        print()
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        result = run_single_experiment(dummy, name, seed=42)
        results.append(result)

    print()
    print("=" * 60)
    print("Results (DummyClassifier baseline)")
    print("=" * 60)
    header = f"{'Dataset':<25} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Brier':>6} {'N_train':>7} {'N_test':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<25} "
            f"{r['accuracy']:>6.3f} "
            f"{r['f1']:>6.3f} "
            f"{r['roc_auc']:>6.3f} "
            f"{r['brier_score']:>6.3f} "
            f"{r['n_train']:>7} "
            f"{r['n_test']:>6}"
        )
    print()
    print("Smoke test passed!")


if __name__ == "__main__":
    main()
