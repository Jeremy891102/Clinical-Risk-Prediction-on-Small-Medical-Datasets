"""
evaluate.py — Nested CV evaluation pipeline.
Owner: Person B (Week 1-2)

Core logic:
  For each dataset:
    For each model:
      For each seed:
        Run 10-fold stratified outer CV
          For tunable models: 5-fold inner CV with RandomizedSearchCV
          For TabPFN: just fit and predict (no tuning)
        Record: accuracy, roc_auc, f1_weighted, train_time, predict_time

TODO LIST:
  [B1] Implement evaluate_single()    — ~30 min (the core function)
  [B2] Implement evaluate_all()       — ~20 min (loop over datasets × models × seeds)
  [B3] Implement save_results()       — ~10 min
  [B4] Implement run_xgboost_default() — ~10 min (XGBoost with no tuning, for fair comparison)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.config import (
    N_OUTER_FOLDS, N_INNER_FOLDS, N_RANDOM_SEARCH_ITER,
    RANDOM_SEEDS, METRICS, TABLES_DIR
)
from src.models import MODELS, PARAM_GRIDS
from src.data_loader import preprocess


def evaluate_single(model_name, X, y, seed):
    """
    Evaluate one model on one dataset with one seed.
    Returns a list of dicts, one per outer fold.

    TODO [B1]:
    1. Create StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=seed)
    2. For each fold:
       a. Split into X_train, X_test, y_train, y_test
       b. Get model factory and model_type from MODELS[model_name]
       c. Preprocess X_train and X_test using preprocess(model_type=...)
       d. If model needs tuning (MODELS[model_name][1] == True):
          - Create fresh model instance
          - Run RandomizedSearchCV with:
              inner CV = StratifiedKFold(N_INNER_FOLDS)
              param_distributions = PARAM_GRIDS[model_name]
              n_iter = N_RANDOM_SEARCH_ITER
              scoring = 'roc_auc'
          - best_model = search.best_estimator_
       e. If model does NOT need tuning (TabPFN):
          - Just create and fit the model
       f. Time the fit: start = time.time(); model.fit(...); train_time = time.time() - start
       g. Time the predict: predict_time
       h. Compute metrics:
          - accuracy = accuracy_score(y_test, y_pred)
          - roc_auc = roc_auc_score(y_test, y_proba)
            NOTE: for binary, use y_proba[:, 1]
            CAREFUL: some models might fail on predict_proba, wrap in try/except
          - f1 = f1_score(y_test, y_pred, average='weighted')
       i. Store fold results + the y_test indices and y_pred
          (we need per-sample predictions for error analysis later!)
    3. Return list of fold result dicts

    HINT for storing per-sample predictions:
        fold_result = {
            "fold": fold_idx,
            "accuracy": acc,
            "roc_auc": auc,
            "f1_weighted": f1,
            "train_time": train_time,
            "predict_time": predict_time,
            "test_indices": test_idx.tolist(),
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }
    """
    raise NotImplementedError("TODO [B1]: implement evaluate_single()")


def evaluate_all(datasets):
    """
    Run evaluate_single() for all datasets × models × seeds.

    TODO [B2]:
    - datasets: dict of {name: (X, y)} from data_loader.load_all_datasets()
    - Loop: for dataset_name → for model_name → for seed
    - Aggregate fold results into a big DataFrame
    - Columns: dataset, model, seed, fold, accuracy, roc_auc, f1_weighted,
               train_time, predict_time
    - Save per-sample predictions separately (for error analysis)
    - Print progress: "Evaluating {model} on {dataset} (seed={seed})..."
    - Return: (results_df, predictions_dict)

    HINT: predictions_dict structure for error analysis:
        predictions_dict[dataset_name][model_name] = {
            "test_indices": [...],  # all test sample indices across folds
            "y_true": [...],
            "y_pred": [...],
        }

    NOTE: This is the most time-consuming step (~30-60 min compute time).
          Consider adding a checkpoint: save intermediate results after each dataset.
    """
    raise NotImplementedError("TODO [B2]: implement evaluate_all()")


def save_results(results_df, predictions_dict):
    """
    TODO [B3]:
    - Save results_df to TABLES_DIR/main_results.csv
    - Save predictions_dict to TABLES_DIR/predictions.pkl (use pickle or json)
    - Also create a summary table: mean ± std per (dataset, model) across seeds+folds
      Save to TABLES_DIR/summary_results.csv
    """
    raise NotImplementedError("TODO [B3]: implement save_results()")


def run_xgboost_default(datasets):
    """
    Run XGBoost with DEFAULT hyperparameters (no tuning) on all datasets.
    This is for fair comparison with TabPFN (both zero config).

    TODO [B4]:
    - Same CV protocol as evaluate_single but skip RandomizedSearchCV
    - Save results to TABLES_DIR/xgboost_default_results.csv
    """
    raise NotImplementedError("TODO [B4]: implement run_xgboost_default()")
