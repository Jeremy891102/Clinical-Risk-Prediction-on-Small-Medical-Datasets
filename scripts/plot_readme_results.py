"""Render README figures from archived fold scores; never train or fetch data.

Run from any directory: python scripts/plot_readme_results.py
Requires only NumPy and Matplotlib from requirements.txt.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "readme"
DATASETS = ["heart", "pima", "breast_cancer", "liver", "ckd"]
DATASET_LABELS = ["Heart", "Pima", "Breast cancer", "Liver", "CKD"]
MODELS = ["logistic_regression", "svm", "random_forest", "mlp",
          "xgboost", "catboost", "tabpfn"]
LABELS = ["Logistic regression", "SVM", "Random forest", "MLP",
          "XGBoost", "CatBoost", "TabPFN"]
TEAL, INK, MUTED = "#087f8c", "#182b40", "#526577"


def load_run(dataset: str, model: str, seed: int) -> dict:
    folder = "raw" if seed == 42 else "raw_multi_seed"
    path = ROOT / "results" / folder / f"{dataset}_{model}_seed{seed}.json"
    run = json.loads(path.read_text())
    assert (run["dataset"], run["model"], run["seed"]) == (dataset, model, seed), path
    folds = run["fold_results"]
    assert len(folds) == 10 and {f["fold"] for f in folds} == set(range(10)), path
    for metric in ("roc_auc", "brier_score"):
        values = np.array([f["metrics"][metric] for f in folds])
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all(), path
        assert np.isclose(values.mean(), run["aggregate"][metric]["mean"], atol=1e-12), path
    return run


def scorecard() -> None:
    sizes = {r["dataset"]: int(r["n_samples"]) for r in csv.DictReader(
        (ROOT / "results/tables/dataset_comparison.csv").open())}
    runs = {(d, m): load_run(d, m, 42) for d in DATASETS for m in MODELS}
    for (d, _), run in runs.items():
        assert sum(f["n_test"] for f in run["fold_results"]) == sizes[d]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(left=.15, right=.97, top=.77, bottom=.20, wspace=.45)
    fig.text(.025, .945, "Can a foundation model improve small-data prediction?",
             fontsize=19, weight="bold", color=INK)
    fig.text(.025, .89, "7 models  /  5 datasets  /  10 outer folds  /  seed 42",
             fontsize=12, color=MUTED)
    configs = [("roc_auc", "AUROC  ↑ higher is better", "Blues", .65, 1., 3),
               ("brier_score", "Brier score  ↓ lower is better", "Blues_r", 0., .22, 4)]
    for ax, (metric, title, cmap, vmin, vmax, decimals) in zip(axes, configs):
        values = np.array([[runs[d, m]["aggregate"][metric]["mean"]
                            for d in DATASETS] for m in MODELS])
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14)
        ax.set_xticks(range(5), [f"{label}\nn={sizes[d]}" for d, label in zip(DATASETS, DATASET_LABELS)], fontsize=9)
        ax.set_yticks(range(7), LABELS, fontsize=10)
        ax.tick_params(length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for j in range(5):
            best = values[:, j].max() if metric == "roc_auc" else values[:, j].min()
            for i in range(7):
                value = values[i, j]
                winner = np.isclose(value, best, rtol=0, atol=1e-12)
                rgba = im.cmap(im.norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, f"{value:.{decimals}f}", ha="center", va="center",
                        fontsize=10, color="white" if luminance < .5 else INK,
                        weight="bold" if winner else "normal")
                if winner:
                    ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False,
                                           edgecolor=TEAL, linewidth=2))
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.04, pad=.16)
        cb.ax.tick_params(labelsize=8)
    fig.text(.025, .06, "Outlined cells = best mean in each dataset (ties included). Values are outer-fold means, not pooled scores.",
             fontsize=10, color=MUTED)
    fig.text(.025, .02, "Baselines: 5-fold inner tuning. TabPFN: default configuration, no inner search. Source: results/raw/*_seed42.json",
             fontsize=9, color=MUTED)
    fig.savefig(OUT / "benchmark.png", dpi=180, facecolor="white")
    plt.close(fig)


def stability() -> None:
    seeds = [0, 1, 7, 42]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(left=.16, right=.98, top=.77, bottom=.24, wspace=.60)
    fig.text(.025, .945, "How stable is the result across four seeds?", fontsize=19, weight="bold", color=INK)
    fig.text(.025, .89, "Large dots = mean  /  bars = ±1 seed SD  /  small dots = individual seed means",
             fontsize=11, color=MUTED)
    saved = {(r["dataset"], r["model"]): r for r in csv.DictReader(
        (ROOT / "results/multi_seed_summary.csv").open())}
    for ax, dataset, title in zip(axes, ["heart", "pima"], ["Heart", "Pima"]):
        for i, model in enumerate(MODELS):
            values = np.array([load_run(dataset, model, s)["aggregate"]["roc_auc"]["mean"] for s in seeds])
            mean, std = values.mean(), values.std(ddof=1)
            # Cross-check against the existing rounded summary, not only plotted data.
            row = saved[dataset, model]
            assert int(row["n_seeds"]) == 4
            assert abs(mean - float(row["auroc_4seed_mean"])) <= .00011
            assert abs(std - float(row["auroc_4seed_std"])) <= .00011
            color = TEAL if model == "tabpfn" else "#71869a"
            ax.scatter(values, i + np.linspace(-.13, .13, 4), s=22, color=color, alpha=.55, zorder=3)
            ax.errorbar(mean, i, xerr=std, fmt="o", color=color, markersize=7, capsize=4, zorder=4)
        ax.set_yticks(range(7), LABELS, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
        ax.set_xlabel("AUROC →", fontsize=11)
        ax.grid(axis="x", color="#e3e9ee", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    fig.text(.025, .12, "Seeds: 0, 1, 7, 42. RF / CatBoost use reduced tuning grids for seeds 0, 1, 7; seed 42 uses full grids.", fontsize=10, color=MUTED)
    fig.text(.025, .075, "This is a robustness check with mixed search budgets; bars are descriptive SD, not confidence intervals.", fontsize=10, color=MUTED)
    fig.text(.025, .03, "Source: results/raw/ and results/raw_multi_seed/; each point averages 10 outer folds. Panel x-axis ranges differ.", fontsize=9, color=MUTED)
    fig.savefig(OUT / "stability.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": INK})
    OUT.mkdir(parents=True, exist_ok=True)
    scorecard()
    stability()
    print("Verified archived scores and wrote benchmark.png + stability.png")
