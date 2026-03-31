"""
config.py — All project constants in one place.
Owner: Person A (Week 1)

Nothing to implement here, just fill in your names and adjust if needed.
"""

# ── Team ──────────────────────────────────────────────
TEAM = {
    "person_a": "YOUR_NAME",
    "person_b": "YOUR_NAME",
}

# ── Reproducibility ───────────────────────────────────
RANDOM_SEEDS = [42, 123, 456]   # run CV with 3 different seeds, report mean ± std
N_OUTER_FOLDS = 10              # 10-fold stratified CV
N_INNER_FOLDS = 5               # 5-fold inner CV for hyperparameter tuning
N_RANDOM_SEARCH_ITER = 50       # number of random search iterations per model

# ── Datasets ──────────────────────────────────────────
# (name, source, id/loader, description)
DATASETS = {
    "heart":   {"source": "uci",     "id": 45,  "target": "num",      "desc": "Heart Disease Cleveland"},
    "diabetes":{"source": "openml",  "id": 37,  "target": "default",  "desc": "Pima Indians Diabetes"},
    "breast":  {"source": "sklearn", "id": None, "target": None,      "desc": "Breast Cancer Wisconsin"},
    "liver":   {"source": "uci",     "id": 225, "target": "Selector", "desc": "Indian Liver Patient"},
    "kidney":  {"source": "uci",     "id": 336, "target": "class",    "desc": "Chronic Kidney Disease"},
}

# ── Metrics ───────────────────────────────────────────
METRICS = ["accuracy", "roc_auc", "f1_weighted"]

# ── Ablation ──────────────────────────────────────────
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.0]

# ── Paths ─────────────────────────────────────────────
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
