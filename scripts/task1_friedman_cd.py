"""Task 1: Friedman test + post-hoc Nemenyi + Critical Difference diagrams.

For both AUROC (higher=better) and Brier (lower=better, ranks inverted).

Inputs:  results/raw/{dataset}_{model}_seed42.json (35 files)
Outputs:
    results/tables/friedman_nemenyi_auroc.csv  -- pairwise Nemenyi p-values
    results/tables/friedman_nemenyi_brier.csv  -- pairwise Nemenyi p-values
    results/tables/avg_ranks.csv               -- mean ranks per model for both metrics
    results/figures/cd_diagram_auroc.pdf       -- CD plot (AUROC)
    results/figures/cd_diagram_brier.pdf       -- CD plot (Brier)

Standard Demšar (2006) recipe:
  1. Per-dataset rank classifiers (1 = best).
  2. Friedman chi-square on rank matrix.
  3. If significant, Nemenyi post-hoc:
        z = |R_i - R_j| / sqrt(k(k+1) / (6N))
        p = sf_studentized_range(z * sqrt(2), k, inf)
  4. CD = q_α * sqrt(k(k+1) / (6N)),  q_α = studentized_range.ppf(1-α, k, inf)/sqrt(2).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
RAW_DIR = ROOT / "results" / "raw"
ALPHA = 0.05

DATASET_ORDER = ["heart", "pima", "breast_cancer", "liver", "ckd"]
MODEL_ORDER = [
    "logistic_regression", "svm", "random_forest", "mlp",
    "xgboost", "catboost", "tabpfn",
]


def load_summary(seed: int = 42) -> pd.DataFrame:
    rows = []
    for f in sorted(RAW_DIR.glob(f"*_seed{seed}.json")):
        d = json.load(open(f))
        a = d["aggregate"]
        rows.append({
            "dataset": d["dataset"], "model": d["model"], "seed": d["seed"],
            "auroc": a["roc_auc"]["mean"],
            "f1": a["f1"]["mean"],
            "brier": a["brier_score"]["mean"],
            "accuracy": a["accuracy"]["mean"],
        })
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def rank_pivot(df: pd.DataFrame, metric: str, higher_is_better: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw_pivot, rank_pivot). 1 = best. Average ranks for ties."""
    pivot = df.pivot(index="dataset", columns="model", values=metric)
    pivot = pivot.loc[DATASET_ORDER, MODEL_ORDER]
    if higher_is_better:
        # rank in descending order: best score → rank 1
        ranks = pivot.apply(lambda r: rankdata(-r.values, method="average"), axis=1, result_type="expand")
    else:
        ranks = pivot.apply(lambda r: rankdata(r.values, method="average"), axis=1, result_type="expand")
    ranks.columns = pivot.columns
    return pivot, ranks


def friedman_nemenyi(ranks: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Run Friedman + Nemenyi post-hoc on a rank matrix (rows=datasets, cols=models)."""
    k = ranks.shape[1]
    N = ranks.shape[0]

    # Friedman test: scipy expects per-classifier sequences (one column per group)
    chi2, p = friedmanchisquare(*[ranks.iloc[:, i].values for i in range(k)])

    mean_ranks = ranks.mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6 * N))

    # Nemenyi pairwise p-values: studentized range with k means, infinite df
    pairwise_p = pd.DataFrame(
        np.ones((k, k)), index=ranks.columns, columns=ranks.columns
    )
    for i, m1 in enumerate(ranks.columns):
        for j, m2 in enumerate(ranks.columns):
            if i >= j:
                continue
            diff = abs(mean_ranks[m1] - mean_ranks[m2])
            z = diff / se
            # p = P(Q > z*sqrt(2)) where Q ~ studentized_range(k, inf)
            p_ij = float(studentized_range.sf(z * np.sqrt(2), k, np.inf))
            pairwise_p.loc[m1, m2] = p_ij
            pairwise_p.loc[m2, m1] = p_ij

    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    CD = float(q_alpha * se)

    return {
        "chi2": float(chi2), "p": float(p), "df": k - 1,
        "k": k, "N": N,
        "mean_ranks": mean_ranks,
        "pairwise_p": pairwise_p,
        "CD": CD,
        "alpha": alpha,
    }


def cd_diagram(mean_ranks: pd.Series, CD: float, title: str, out_path: Path):
    """Demšar-style CD diagram.

    Models drawn at their mean-rank position on a horizontal axis.
    Models connected by a thick horizontal line are NOT significantly different
    (their rank difference is ≤ CD).
    """
    ranks = mean_ranks.sort_values()
    names = list(ranks.index)
    values = ranks.values
    k = len(values)
    rank_min = float(np.floor(values.min()))
    rank_max = float(np.ceil(values.max()))

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.18)

    # Demšar convention: best (lowest rank) on the left → matplotlib default, no invert.
    ax.set_xlim(rank_min - 0.3, rank_max + 0.3)
    ax.set_ylim(-0.5, 1.6)

    # Show top axis only
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    tick_step = 1 if (rank_max - rank_min) <= 8 else 2
    ax.set_xticks(np.arange(rank_min, rank_max + 0.001, tick_step))
    ax.set_xlabel("Average rank (lower = better)")
    ax.tick_params(axis="x", which="major", length=4)

    # vertical ticks at each model's rank
    for r in values:
        ax.plot([r, r], [0.0, 0.05], color="black", lw=1.0)

    # label models with elbow lines: half on left below, half on right below.
    # Use Demšar convention: alternate ascending columns left then right.
    # Simpler approach used here: split at midpoint by sorted index.
    half = (k + 1) // 2
    label_y_top = 0.3
    label_y_step = 0.20
    for idx, (name, r) in enumerate(zip(names, values)):
        if idx < half:
            # left side — text on the far left, elbow goes down then left
            text_x = rank_min - 0.25
            text_y = label_y_top + label_y_step * (half - 1 - idx)
            ha = "right"
        else:
            # right side
            text_x = rank_max + 0.25
            text_y = label_y_top + label_y_step * (idx - half)
            ha = "left"
        # elbow: vertical down from tick, then horizontal to label
        ax.plot([r, r], [0.0, text_y], color="black", lw=0.7)
        ax.plot([r, text_x], [text_y, text_y], color="black", lw=0.7)
        ax.text(text_x + (-0.03 if ha == "right" else 0.03), text_y,
                name.replace("_", " "),
                ha=ha, va="center", fontsize=10)

    # CD bar at top
    cd_y = 1.25
    cd_x_start = rank_min
    cd_x_end = cd_x_start + CD
    ax.plot([cd_x_start, cd_x_end], [cd_y, cd_y], color="black", lw=1.5)
    ax.plot([cd_x_start, cd_x_start], [cd_y - 0.06, cd_y + 0.06], color="black", lw=1.5)
    ax.plot([cd_x_end, cd_x_end], [cd_y - 0.06, cd_y + 0.06], color="black", lw=1.5)
    ax.text((cd_x_start + cd_x_end) / 2, cd_y + 0.12, f"CD = {CD:.3f}",
            ha="center", va="bottom", fontsize=10)

    # Group bars: find maximal cliques of models with consecutive rank differences ≤ CD.
    # Iterate over sorted ranks; for each starting i, extend j while ranks[j] - ranks[i] ≤ CD.
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and (values[j + 1] - values[i]) <= CD + 1e-9:
            j += 1
        if j > i:
            groups.append((i, j))
    # Remove subsumed groups (only keep maximal)
    maximal = []
    for g in groups:
        if not any(o[0] <= g[0] and g[1] <= o[1] and o != g for o in groups):
            maximal.append(g)

    bar_y = -0.15
    for (i, j) in maximal:
        x0, x1 = values[i], values[j]
        ax.plot([x0 - 0.05, x1 + 0.05], [bar_y, bar_y], color="black", lw=4.0,
                solid_capstyle="butt")
        bar_y -= 0.10

    ax.set_title(title, pad=24)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print(">>> Task 1: Friedman + Nemenyi + CD diagrams")
    df = load_summary(seed=42)
    print(f"Loaded {len(df)} rows from results/raw/*_seed42.json "
          f"({df['dataset'].nunique()} datasets × {df['model'].nunique()} models)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    avg_ranks = {}

    for metric, higher in [("auroc", True), ("brier", False)]:
        direction = "higher = better" if higher else "lower = better"
        print(f"\n--- Metric: {metric} ({direction}) ---")
        pivot, ranks = rank_pivot(df, metric, higher_is_better=higher)
        print("Per-dataset values:")
        print(pivot.round(4).to_string())
        print("\nPer-dataset ranks (1 = best):")
        print(ranks.round(2).to_string())

        res = friedman_nemenyi(ranks, alpha=ALPHA)
        print(f"\nFriedman: chi2 = {res['chi2']:.4f}, df = {res['df']}, "
              f"p = {res['p']:.4f}  (k={res['k']} models, N={res['N']} datasets)")
        if res["p"] < ALPHA:
            print(f"  → significant at α={ALPHA}; running Nemenyi post-hoc")
        else:
            print(f"  → NOT significant at α={ALPHA}; Nemenyi reported for completeness")

        print(f"\nMean ranks (sorted):")
        for m, r in res["mean_ranks"].sort_values().items():
            print(f"  {m:>22}: {r:.3f}")
        print(f"\nCritical difference (α={ALPHA}, k={res['k']}, N={res['N']}): CD = {res['CD']:.3f}")

        # Save Nemenyi pairwise p-value matrix
        out_csv = TABLES_DIR /f"friedman_nemenyi_{metric}.csv"
        # Add a header section with summary stats
        header_lines = [
            f"# Friedman + Nemenyi post-hoc on {metric} ({direction})",
            f"# k = {res['k']} models, N = {res['N']} datasets",
            f"# Friedman: chi2 = {res['chi2']:.4f}, df = {res['df']}, p = {res['p']:.4f}",
            f"# CD (alpha={ALPHA}) = {res['CD']:.4f}",
            f"# Pairwise Nemenyi p-values (lower = more different)",
        ]
        with open(out_csv, "w") as fh:
            fh.write("\n".join(header_lines) + "\n")
            res["pairwise_p"].round(4).to_csv(fh)
        print(f"Saved: {out_csv}")

        # Save CD diagram
        out_pdf = FIGURES_DIR / f"cd_diagram_{metric}.pdf"
        title = f"Critical Difference Diagram — {metric.upper()} ({direction})"
        cd_diagram(res["mean_ranks"], res["CD"], title, out_pdf)
        print(f"Saved: {out_pdf}")

        avg_ranks[metric] = res["mean_ranks"]

    # combined ranks CSV
    ranks_df = pd.DataFrame(avg_ranks)
    ranks_df.index.name = "model"
    ranks_csv = TABLES_DIR /"avg_ranks.csv"
    ranks_df.round(4).to_csv(ranks_csv)
    print(f"\nSaved combined avg ranks: {ranks_csv}")
    print(ranks_df.round(3).to_string())

    print("\nTASK 1 DONE")


if __name__ == "__main__":
    main()
