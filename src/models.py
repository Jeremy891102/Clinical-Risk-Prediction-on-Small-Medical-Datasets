"""
models.py — Define all 7 models with sklearn-compatible API.
Owner: Person A (Week 1)

Every model must have .fit(X, y) and .predict(X) and .predict_proba(X).

TODO LIST:
  [A8]  Implement get_logistic_regression()    — ~5 min
  [A9]  Implement get_svm()                    — ~10 min
  [A10] Implement get_random_forest()          — ~5 min
  [A11] Implement get_mlp()                    — ~10 min
  [A12] Implement get_xgboost()                — ~5 min
  [A13] Implement get_catboost()               — ~10 min
  [A14] Implement get_tabpfn()                 — ~10 min
  [A15] Define PARAM_GRIDS for each model      — ~20 min
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# TODO [A14]: uncomment when tabpfn is installed
# from tabpfn import TabPFNClassifier


# ── Model Factories ───────────────────────────────────
# Each returns a fresh (unfitted) sklearn-compatible estimator.

def get_logistic_regression():
    """
    TODO [A8]: L2-regularized logistic regression.
    - Use solver='lbfgs', max_iter=1000
    - Set random_state for reproducibility
    """
    raise NotImplementedError("TODO [A8]")


def get_svm():
    """
    TODO [A9]: SVM with RBF kernel.
    - MUST set probability=True (needed for predict_proba → ROC AUC)
    - Set random_state for reproducibility
    - NOTE: SVM is slow on larger datasets, this is expected
    """
    raise NotImplementedError("TODO [A9]")


def get_random_forest():
    """
    TODO [A10]: Random Forest.
    - Use n_estimators=100 as default
    - Set random_state for reproducibility
    """
    raise NotImplementedError("TODO [A10]")


def get_mlp():
    """
    TODO [A11]: Multi-layer Perceptron with 2 hidden layers.
    - hidden_layer_sizes=(64, 32) is a good default
    - max_iter=500, early_stopping=True
    - Set random_state for reproducibility
    - NOTE: MLP needs scaled input — handled in preprocess()
    """
    raise NotImplementedError("TODO [A11]")


def get_xgboost():
    """
    TODO [A12]: XGBoost classifier.
    - use_label_encoder=False, eval_metric='logloss'
    - Set random_state for reproducibility
    - verbosity=0 to suppress output
    """
    raise NotImplementedError("TODO [A12]")


def get_catboost():
    """
    TODO [A13]: CatBoost classifier.
    - verbose=0 to suppress output
    - Set random_seed for reproducibility
    - CatBoost handles categoricals natively — pass cat_features in .fit()
    """
    raise NotImplementedError("TODO [A13]")


def get_tabpfn():
    """
    TODO [A14]: TabPFN 2.5.
    - Install: pip install tabpfn
    - TabPFNClassifier() with default params — that's the whole point (zero-shot)
    - No hyperparameter tuning needed
    - Requires GPU for fast inference, falls back to CPU (slower)
    - NOTE: check TabPFN docs for current API, it may have changed
    """
    raise NotImplementedError("TODO [A14]")


# ── Hyperparameter Search Spaces ──────────────────────
# Used by RandomizedSearchCV in evaluate.py
# TabPFN is NOT here because it doesn't need tuning.

PARAM_GRIDS = {
    "logistic_regression": {
        # TODO [A15]: define search space
        # "C": [0.001, 0.01, 0.1, 1, 10, 100],
        # "penalty": ["l2"],
    },
    "svm": {
        # TODO [A15]: define search space
        # "C": [0.1, 1, 10, 100],
        # "gamma": ["scale", "auto", 0.01, 0.1],
    },
    "random_forest": {
        # TODO [A15]: define search space
        # "n_estimators": [50, 100, 200, 500],
        # "max_depth": [None, 5, 10, 20],
        # "min_samples_split": [2, 5, 10],
    },
    "mlp": {
        # TODO [A15]: define search space
        # "hidden_layer_sizes": [(32,), (64,32), (128,64)],
        # "learning_rate_init": [0.001, 0.01],
        # "alpha": [0.0001, 0.001, 0.01],
    },
    "xgboost": {
        # TODO [A15]: define search space
        # "n_estimators": [50, 100, 200],
        # "max_depth": [3, 5, 7],
        # "learning_rate": [0.01, 0.1, 0.3],
        # "subsample": [0.8, 1.0],
    },
    "catboost": {
        # TODO [A15]: define search space
        # "iterations": [100, 200, 500],
        # "depth": [4, 6, 8],
        # "learning_rate": [0.01, 0.1, 0.3],
    },
}


# ── Registry ──────────────────────────────────────────
# Maps model name → (factory_function, needs_tuning, model_type)
# model_type is used by preprocess() to decide how to handle data

MODELS = {
    "logistic_regression": (get_logistic_regression, True,  "linear"),
    "svm":                 (get_svm,                 True,  "linear"),
    "random_forest":       (get_random_forest,       True,  "tree"),
    "mlp":                 (get_mlp,                 True,  "mlp"),
    "xgboost":             (get_xgboost,             True,  "tree"),
    "catboost":            (get_catboost,            True,  "tree"),
    "tabpfn":              (get_tabpfn,              False, "tabpfn"),
}
