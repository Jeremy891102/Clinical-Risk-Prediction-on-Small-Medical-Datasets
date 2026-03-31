"""
data_loader.py — Download and preprocess all 5 medical datasets.
Owner: Person A (Week 1)

Each loader returns (X: pd.DataFrame, y: pd.Series) ready for sklearn.

TODO LIST:
  [A1] Implement load_heart()       — ~15 min
  [A2] Implement load_diabetes()    — ~10 min
  [A3] Implement load_breast()      — ~5 min (sklearn built-in)
  [A4] Implement load_liver()       — ~15 min
  [A5] Implement load_kidney()      — ~20 min (most preprocessing needed)
  [A6] Implement preprocess()       — ~20 min (imputation + encoding)
  [A7] Implement get_dataset_stats() — ~15 min (for paper Data section table)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder


def load_heart():
    """
    Heart Disease Cleveland — UCI ID 45
    303 samples, 13 features, binary classification

    TODO [A1]:
    - Use `from ucimlrepo import fetch_ucirepo` with id=45
    - Target column is 'num': values 0-4
    - Binarize target: 0 = no disease, 1-4 = disease → (y > 0).astype(int)
    - Return (X, y)
    """
    raise NotImplementedError("TODO [A1]: implement load_heart()")


def load_diabetes():
    """
    Pima Indians Diabetes — OpenML ID 37
    768 samples, 8 features, binary classification

    TODO [A2]:
    - Use `import openml; openml.datasets.get_dataset(37)`
    - Call .get_data(target=dataset.default_target_attribute)
    - Encode target: 'tested_negative' → 0, 'tested_positive' → 1
    - NOTE: columns like blood pressure, skin thickness, insulin have 0 values
      that are actually missing. Consider replacing 0 with NaN for those columns.
      Columns with suspicious zeros: 'plas', 'pres', 'skin', 'insu', 'mass'
    - Return (X, y)
    """
    raise NotImplementedError("TODO [A2]: implement load_diabetes()")


def load_breast():
    """
    Breast Cancer Wisconsin — sklearn built-in
    569 samples, 30 features, binary classification

    TODO [A3]:
    - Use sklearn.datasets.load_breast_cancer()
    - target: 0 = malignant, 1 = benign
    - Return (X as DataFrame, y as Series)
    """
    raise NotImplementedError("TODO [A3]: implement load_breast()")


def load_liver():
    """
    Indian Liver Patient (ILPD) — UCI ID 225
    583 samples, 10 features, binary classification

    TODO [A4]:
    - Use fetch_ucirepo(id=225)
    - Target 'Selector': 1 = liver disease, 2 = no disease
    - Remap target: 1 → 1 (disease), 2 → 0 (no disease)
    - 'Gender' column is categorical ('Male'/'Female'), need to encode
    - Has 4 missing values in 'A/G Ratio'
    - NOTE: heavy class imbalance (416 vs 167), document this
    - Return (X, y)
    """
    raise NotImplementedError("TODO [A4]: implement load_liver()")


def load_kidney():
    """
    Chronic Kidney Disease — UCI ID 336
    400 samples, 24 features (14 numeric + 10 categorical), binary classification

    TODO [A5]:
    - Use fetch_ucirepo(id=336)
    - Target 'class': strip whitespace! ('ckd\\t' → 'ckd')
    - Encode target: 'ckd' → 1, 'notckd' → 0
    - Has MANY missing values (10.5% of all cells)
      Top missing: rbc (38%), rbcc (33%), wbcc (27%), pot (22%), sod (22%)
    - Has 10 categorical columns that need encoding
    - Return (X, y)
    """
    raise NotImplementedError("TODO [A5]: implement load_kidney()")


def preprocess(X, y, model_type="tree"):
    """
    Preprocess features based on model type.

    TODO [A6]:
    - If model_type == "tree" (RF, XGBoost, CatBoost) or "tabpfn":
        - These handle NaN and categorical natively (mostly)
        - For XGBoost/RF: encode categoricals with LabelEncoder
        - For CatBoost: pass categorical column indices
        - For TabPFN: handles everything natively
    - If model_type == "linear" (LogReg, SVM) or "mlp":
        - Impute missing: median for numeric, mode for categorical
        - Encode categoricals: LabelEncoder or OneHotEncoder
        - StandardScaler for numeric features
    - Return (X_processed, y)

    HINT: use sklearn Pipeline to keep it clean:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
    """
    raise NotImplementedError("TODO [A6]: implement preprocess()")


def get_dataset_stats(X, y, name):
    """
    Return a dict of stats for the paper's Data section table.

    TODO [A7]:
    - n_samples, n_features
    - n_numerical, n_categorical
    - n_classes
    - class_distribution (e.g., {0: 164, 1: 139})
    - majority_class_pct
    - missing_pct (total missing cells / total cells)
    - Return as dict
    """
    raise NotImplementedError("TODO [A7]: implement get_dataset_stats()")


# ── Convenience function ──────────────────────────────

LOADERS = {
    "heart":    load_heart,
    "diabetes": load_diabetes,
    "breast":   load_breast,
    "liver":    load_liver,
    "kidney":   load_kidney,
}


def load_all_datasets():
    """Load all 5 datasets, return dict of {name: (X, y)}."""
    datasets = {}
    for name, loader in LOADERS.items():
        print(f"Loading {name}...")
        X, y = loader()
        stats = get_dataset_stats(X, y, name)
        print(f"  {stats['n_samples']} samples, {stats['n_features']} features, "
              f"{stats['n_classes']} classes, {stats['missing_pct']:.1f}% missing")
        datasets[name] = (X, y)
    return datasets


if __name__ == "__main__":
    load_all_datasets()
