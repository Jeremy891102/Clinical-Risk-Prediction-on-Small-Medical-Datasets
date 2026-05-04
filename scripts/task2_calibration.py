"""Task 2: Reliability diagrams + ECE for heart, pima.

Each sample appears in exactly one outer-fold test set, so we can stitch the
fold-level y_proba/test_indices into out-of-fold (OOF) probabilities for the
full dataset, then compute calibration metrics without any re-fitting.

Inputs:  results/raw/{heart,pima}_{model}_seed42.json
Outputs:
    results/figures/reliability_heart.pdf
    results/figures/reliability_pima.pdf
    results/tables/ece_per_model.csv  (dataset, model, ECE, Brier, n_samples)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

DATASETS = ["heart", "pima"]
MODELS = [
    "logistic_regression", "svm", "random_forest", "mlp",
    "xgboost", "catboost", "tabpfn",
]
N_BINS = 10
SEED = 42

MODEL_COLORS = {
    "logistic_regression": "#1f77b4",
    "svm": "#ff7f0e",
    "random_forest": "#2ca02c",
    "mlp": "#d62728",
    "xgboost": "#9467bd",
    "catboost": "#8c564b",
    "tabpfn": "#000000",
}


def load_oof(dataset: str, model: str, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stitch fold-level y_proba into OOF arrays. Returns (y_true, y_proba1)."""
    path = RAW_DIR / f"{dataset}_{model}_seed{seed}.json"
    d = json.load(open(path))
    n_total = sum(len(f["test_indices"]) for f in d["fold_results"])
    y_true = np.full(n_total, -1, dtype=int)
    y_proba = np.full(n_total, np.nan, dtype=float)
    for f in d["fold_results"]:
        idx = np.asarray(f["test_indices"])
        y_true[idx] = np.asarray(f["y_true"], dtype=int)
        y_proba[idx] = np.asarray(f["y_proba"], dtype=float)
    assert (y_true != -1).all(), f"Some samples not in any test fold for {dataset}×{model}"
    assert not np.isnan(y_proba).any()
    return y_true, y_proba


def reliability_bins(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS):
    """Equal-width binning of [0,1]. Returns per-bin (mean_pred, frac_pos, count)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index for each sample (0..n_bins-1). Clip the rightmost edge into the last bin.
    idx = np.clip(np.digitize(y_proba, edges, right=False) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_pred[b] = float(y_proba[mask].mean())
            frac_pos[b] = float(y_true[mask].mean())
    return mean_pred, frac_pos, counts


def ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error (Naeini et al., 2015) — equal-width bins."""
    mean_pred, frac_pos, counts = reliability_bins(y_true, y_proba, n_bins)
    n = counts.sum()
    valid = counts > 0
    return float(np.sum(counts[valid] / n * np.abs(mean_pred[valid] - frac_pos[valid])))


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def plot_reliability(dataset: str, ece_table: list[dict], out_path: Path):
    """One figure with 7 model curves + diagonal reference."""
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, ls="--", label="perfectly calibrated")

    rows_for_dataset = [r for r in ece_table if r["dataset"] == dataset]
    rows_for_dataset.sort(key=lambda r: r["ECE"])  # legend ordered by calibration

    for r in rows_for_dataset:
        mp, fp, ct = r["_bins"]
        valid = ct > 0
        label = (f"{r['model'].replace('_',' ')}  "
                 f"ECE={r['ECE']:.3f}  Brier={r['Brier']:.3f}")
        ax.plot(mp[valid], fp[valid], color=MODEL_COLORS[r["model"]],
                marker="o", lw=1.6, ms=5, label=label)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability (in bin)")
    ax.set_ylabel("Fraction of positives (in bin)")
    ax.set_title(f"Reliability Diagram — {dataset}  (10 equal-width bins, OOF)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print(">>> Task 2: Reliability diagrams + ECE for heart, pima")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table = []
    for ds in DATASETS:
        for m in MODELS:
            y_true, y_proba = load_oof(ds, m)
            e = ece(y_true, y_proba)
            b = brier(y_true, y_proba)
            bins = reliability_bins(y_true, y_proba)
            table.append({
                "dataset": ds, "model": m,
                "ECE": e, "Brier": b, "n_samples": int(len(y_true)),
                "_bins": bins,
            })
            print(f"  {ds:>6} × {m:<22}  ECE={e:.4f}  Brier={b:.4f}  n={len(y_true)}")

    # write CSV (drop the bin tuple)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in table])
    df = df.round(4)
    out_csv = TABLES_DIR / "ece_per_model.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))

    for ds in DATASETS:
        out_pdf = FIGURES_DIR / f"reliability_{ds}.pdf"
        plot_reliability(ds, table, out_pdf)
        print(f"Saved: {out_pdf}")

    # quick summary
    print("\nBest-calibrated model per dataset:")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("ECE")
        print(f"  {ds:>6}: {sub.iloc[0]['model']} (ECE={sub.iloc[0]['ECE']:.4f}); "
              f"worst: {sub.iloc[-1]['model']} (ECE={sub.iloc[-1]['ECE']:.4f})")

    print("\nTASK 2 DONE")


if __name__ == "__main__":
    main()
