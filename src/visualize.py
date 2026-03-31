"""
visualize.py — Generate all tables and figures for the paper.
Owner: Person B (Week 4-5)

All figures saved as PDF to FIGURES_DIR for LaTeX \includegraphics.
All tables saved as CSV to TABLES_DIR (manually format in LaTeX).

TODO LIST:
  [B8]  fig_main_results_table()      — the big comparison table
  [B9]  fig_ablation_curves()         — line plot: accuracy vs train fraction
  [B10] fig_error_overlap()           — heatmap or Venn: who misclassifies what
  [B11] fig_feature_importance()      — grouped bar chart per dataset
  [B12] fig_critical_difference()     — CD diagram from Nemenyi test
  [B13] fig_efficiency()              — accuracy vs time scatter/bar
  [B14] fig_dataset_stats()           — summary stats table for Data section
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, TABLES_DIR

# ── Style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def fig_main_results_table(summary_df):
    """
    The main results table — most important figure in the paper.

    TODO [B8]:
    Input: summary_df with columns [dataset, model, accuracy_mean, accuracy_std,
           roc_auc_mean, roc_auc_std, f1_mean, f1_std]
    
    Create a heatmap-style table:
    - Rows: models (7 models)
    - Columns: datasets (5 datasets) × metrics (accuracy, roc_auc)
    - Cell value: "mean ± std"
    - Bold the best value in each column
    - Add a "Mean Rank" column on the right

    Save as: FIGURES_DIR/main_results.pdf
    Also save the raw table: TABLES_DIR/main_results_formatted.csv

    HINT: use seaborn heatmap or just create a clean matplotlib table.
    A simple approach: save as CSV and format manually in LaTeX.
    """
    raise NotImplementedError("TODO [B8]")


def fig_ablation_curves(ablation_df):
    """
    Line plot showing accuracy vs training data fraction.

    TODO [B9]:
    - One subplot per dataset (5 subplots, 1×5 or 2×3 grid)
    - X-axis: train fraction (10%, 25%, 50%, 100%)
    - Y-axis: accuracy (or roc_auc)
    - One line per model, different colors
    - Add error bars (std across seeds)
    - Highlight the "crossover point" where XGBoost overtakes TabPFN

    Save as: FIGURES_DIR/ablation_curves.pdf
    """
    raise NotImplementedError("TODO [B9]")


def fig_error_overlap(error_df):
    """
    Visualize which samples are misclassified by which models.

    TODO [B10]:
    Option A — Heatmap:
    - Rows: samples (sorted by # models that misclassify them)
    - Columns: models
    - Color: correct (green) / wrong (red)
    - Shows "hard cases" as rows that are all red

    Option B — Bar chart:
    - For each dataset: stacked bar showing
      "only TabPFN wrong", "only XGBoost wrong", "both wrong", "both correct"

    Save as: FIGURES_DIR/error_overlap.pdf
    """
    raise NotImplementedError("TODO [B10]")


def fig_feature_importance(fi_df):
    """
    Feature importance comparison across models.

    TODO [B11]:
    - One subplot per dataset
    - Grouped horizontal bar chart: features on y-axis, importance on x-axis
    - Different bars for each model (3-4 models, not all 7 — pick LR, RF, XGBoost, TabPFN)
    - Highlight features that match known medical risk factors

    Save as: FIGURES_DIR/feature_importance.pdf
    """
    raise NotImplementedError("TODO [B11]")


def fig_critical_difference(ranks_df):
    """
    Critical difference diagram from Nemenyi test.

    TODO [B12]:
    - X-axis: average rank (1 = best, 7 = worst)
    - Models listed on the axis
    - Horizontal bars connecting models that are NOT significantly different
    - This is the standard visualization from Demsar (2006)

    HINT: use the `autorank` package or draw manually:
        pip install autorank
        from autorank import autorank, plot_stats
    
    Alternative: draw manually with matplotlib lines.

    Save as: FIGURES_DIR/critical_difference.pdf
    """
    raise NotImplementedError("TODO [B12]")


def fig_efficiency(efficiency_df):
    """
    Accuracy vs computation time trade-off.

    TODO [B13]:
    - Scatter plot: x-axis = total time (log scale), y-axis = mean accuracy
    - One point per model, labeled
    - Or: grouped bar chart with two bars per model (accuracy, time)
    
    Key story: TabPFN gets high accuracy in seconds, XGBoost needs minutes of tuning.

    Save as: FIGURES_DIR/efficiency.pdf
    """
    raise NotImplementedError("TODO [B13]")


def fig_dataset_stats(datasets):
    """
    Dataset summary table for the Data section.

    TODO [B14]:
    - Load all datasets, compute stats using get_dataset_stats()
    - Create a clean table:
      Dataset | Samples | Features | Num | Cat | Classes | Majority% | Missing%
    - Save as: TABLES_DIR/dataset_stats.csv

    This goes directly into the paper's Data section.
    """
    raise NotImplementedError("TODO [B14]")


def generate_all_figures():
    """
    Load all results CSVs and generate all figures.
    Call this after evaluate.py and analysis.py have run.
    """
    print("Loading results...")
    # TODO: load CSVs from TABLES_DIR
    # summary_df = pd.read_csv(...)
    # ablation_df = pd.read_csv(...)
    # etc.

    print("Generating figures...")
    # fig_main_results_table(summary_df)
    # fig_ablation_curves(ablation_df)
    # fig_error_overlap(error_df)
    # fig_feature_importance(fi_df)
    # fig_critical_difference(ranks_df)
    # fig_efficiency(efficiency_df)
    # fig_dataset_stats(datasets)

    print(f"All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_figures()
