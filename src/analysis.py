"""
analysis.py — All analysis for the paper's Analysis section (20% of grade).
Owner: Person A (ablation) + Person B (error analysis, feature importance)

This is the KEY to getting an A. The rubric specifically asks:
  "does the paper describe which datapoints models misclassify?"

TODO LIST:
  [A16] Implement run_ablation()            — Person A, ~2 hrs
  [B5]  Implement error_analysis()          — Person B, ~2 hrs
  [B6]  Implement feature_importance()      — Person B, ~1 hr
  [B7]  Implement statistical_test()        — Person B, ~1 hr
  [A17] Implement efficiency_analysis()     — Person A, ~30 min
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from scipy.stats import friedmanchisquare
from src.config import TRAIN_FRACTIONS, RANDOM_SEEDS, TABLES_DIR


def run_ablation(datasets):
    """
    Sample size ablation: train with 10%, 25%, 50%, 100% of data.

    TODO [A16]:
    For each dataset:
      For each model:
        For each fraction in TRAIN_FRACTIONS:
          - Subsample the training set (keep test set the same!)
          - Use StratifiedKFold as usual for outer CV
          - In each fold, after splitting train/test:
              train_sub = train[:int(len(train) * fraction)]
              (make sure to stratify the subsample too!)
          - Record accuracy and roc_auc for each fraction
    
    Save to TABLES_DIR/ablation_results.csv
    Columns: dataset, model, fraction, seed, fold, accuracy, roc_auc

    HINT: use sklearn.model_selection.train_test_split with stratify
    to subsample the training set while keeping class balance.

    This answers: "at what training set size does XGBoost overtake TabPFN?"
    """
    raise NotImplementedError("TODO [A16]: implement run_ablation()")


def error_analysis(datasets, predictions_dict):
    """
    Analyze which samples each model misclassifies.

    TODO [B5]:
    For each dataset:
      1. Get per-sample predictions from predictions_dict
         (these come from evaluate.py — each sample was in exactly 1 test fold)
      2. For each model, compute: correct (True/False) for each sample
      3. Build a DataFrame: rows = samples, columns = models, values = correct/wrong
      4. Compute overlap analysis:
         - How many samples do ALL models get wrong? (hard cases)
         - How many samples does ONLY TabPFN get wrong? (TabPFN weaknesses)
         - How many samples does ONLY XGBoost get wrong? (XGBoost weaknesses)
         - Venn diagram data: TabPFN-wrong ∩ XGBoost-wrong
      5. For the "hard cases" (all models wrong):
         - Extract their features
         - Compute mean feature values vs correctly classified samples
         - Are they near decision boundaries? (e.g., borderline lab values)
      6. Save to TABLES_DIR/error_analysis.csv

    This DIRECTLY answers the rubric: "which datapoints models misclassify"
    """
    raise NotImplementedError("TODO [B5]: implement error_analysis()")


def feature_importance(datasets):
    """
    Compare feature importance across models.

    TODO [B6]:
    For each dataset:
      1. Train each model on the full training set
      2. Use sklearn.inspection.permutation_importance() with n_repeats=10
         (this works for ALL models including TabPFN — model-agnostic)
      3. Record top-5 most important features per model
      4. Compare: do all models agree on what's important?
      5. Cross-reference with medical domain knowledge:
         - Heart: chest pain type, max heart rate, ST depression are known risk factors
         - Diabetes: glucose level, BMI, age are key predictors
         - Breast cancer: worst radius, worst concave points are discriminative
         - Liver: SGPT, SGOT, total bilirubin are liver function markers
         - Kidney: hemoglobin, albumin, serum creatinine are CKD indicators
      6. Save to TABLES_DIR/feature_importance.csv

    This answers: "do the models' decisions align with medical knowledge?"
    """
    raise NotImplementedError("TODO [B6]: implement feature_importance()")


def statistical_test(results_df):
    """
    Friedman test + Nemenyi post-hoc test.

    TODO [B7]:
    1. From results_df, compute mean metric per (dataset, model)
       → pivot table: rows=datasets, columns=models, values=mean accuracy (or roc_auc)
    2. Run Friedman test:
       from scipy.stats import friedmanchisquare
       stat, p = friedmanchisquare(*[col for col in pivot.T.values])
       If p < 0.05 → there are significant differences
    3. Run Nemenyi post-hoc test:
       pip install scikit-posthocs
       import scikit_posthocs as sp
       nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
    4. Save p-values table to TABLES_DIR/statistical_tests.csv
    5. Compute average ranks for critical difference diagram
       Save to TABLES_DIR/avg_ranks.csv

    Reference: Demsar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets"
    """
    raise NotImplementedError("TODO [B7]: implement statistical_test()")


def efficiency_analysis(results_df):
    """
    Compare training + inference time across models.

    TODO [A17]:
    1. From results_df, compute mean train_time and predict_time per model
    2. Compute total_time = train_time + predict_time
    3. Create a table: model, mean_train_time, mean_predict_time, mean_total_time
    4. Save to TABLES_DIR/efficiency_results.csv

    Key insight to highlight: TabPFN predict_time ≈ seconds vs XGBoost tuning ≈ minutes
    """
    raise NotImplementedError("TODO [A17]: implement efficiency_analysis()")
