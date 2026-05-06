"""Task 7: Permutation importance across 5 datasets x 7 models (option C).

Option C: TabPFN runs only on heart, pima, liver (the AUROC-discriminating
datasets). It is skipped on breast_cancer and ckd because those datasets
are already saturated (most models exceed 0.99 AUROC at fraction=1.0)
and TabPFN's permutation cost there is huge for negligible analytical
value. The other 6 models run on all 5 datasets.

n_repeats=10 throughout (this script is intended to run on GPU; on CPU
it would take ~17h which is why option C was chosen).

For each (dataset, model): seed=42 10-fold StratifiedKFold; in each fold
fit a pipeline (preprocessor + classifier with the seed=42 best HP) on
the training fold and run sklearn.inspection.permutation_importance on
the test fold (scoring='roc_auc'). Importance per feature is averaged
across the 10 folds (mean and std).

Note on feature naming: permutation_importance is called on the raw X
DataFrame (the Pipeline's first step is a ColumnTransformer that handles
imputation / scaling / one-hot encoding). This means importance is
attributed at the *original* feature level — categorical features have
their entire OneHot block permuted together, which is the standard
interpretation.

Outputs:
    results/permutation_importance.csv     (long: dataset, model, feature, importance_mean, importance_std, rank, n_repeats)
    results/feature_consensus.csv          (per dataset: n_models_top3 / n_models — voting denominator varies per dataset)
    figures/perm_importance_<dataset>.pdf  (5 files; one per dataset, model subplots, top-5 features each)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import get_dataset  # noqa: E402
from src.models.registry import get_model  # noqa: E402

# ----- Config -----
DATASETS = ["heart", "pima", "breast_cancer", "liver", "ckd"]
MODELS = ["logistic_regression", "svm", "random_forest", "mlp",
          "xgboost", "catboost", "tabpfn"]
SEED = 42
K_FOLDS = 10
N_REPEATS = 10

# Option C: skip TabPFN on saturated datasets.
SKIP_BY_DATASET = {
    "breast_cancer": {"tabpfn"},
    "ckd": {"tabpfn"},
}

TOP_K_CONSENSUS = 3
TOP_K_FIG = 5

SEED42_RAW = ROOT / "results" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PERM_CSV = RESULTS_DIR / "permutation_importance.csv"
CONSENSUS_CSV = RESULTS_DIR / "feature_consensus.csv"

DATASET_NFEATURES = {
    "heart": 13, "pima": 8, "breast_cancer": 30, "liver": 10, "ckd": 24,
}


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_skipped(dataset: str, model: str) -> bool:
    return model in SKIP_BY_DATASET.get(dataset, set())


def models_for(dataset: str) -> list[str]:
    return [m for m in MODELS if not is_skipped(dataset, m)]


# ----- Helpers (preprocessor + best-HP loading; same as task6) -----
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def mode_best_params(json_path: Path) -> dict:
    d = json.load(open(json_path))
    if not d.get("fold_results"):
        return {}
    keys = []
    for fr in d["fold_results"]:
        keys.append(tuple(sorted(
            (k, _hashable(v)) for k, v in fr["best_params"].items()
        )))
    most_common, _ = Counter(keys).most_common(1)[0]
    out = {}
    for k, v in most_common:
        out[k.replace("classifier__", "", 1)] = v
    return out


def load_best_hp(dataset: str, model: str) -> dict | None:
    if model == "tabpfn":
        return {}
    f = SEED42_RAW / f"{dataset}_{model}_seed42.json"
    if not f.exists():
        return None
    return mode_best_params(f)


# ----- Per-(dataset, model) run -----
def run_one(dataset: str, model: str, hp: dict, n_repeats: int) -> list[dict]:
    X, y, _ = get_dataset(dataset)
    y = np.asarray(y)
    feature_names = list(X.columns)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    fold_means = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = y[tr]
        y_te = y[te]

        wrapper = get_model(model, random_state=SEED)
        estimator = wrapper.get_estimator()
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(X_tr)),
            ("classifier", estimator),
        ])
        if hp:
            pipe.set_params(**{f"classifier__{k}": v for k, v in hp.items()})

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y_tr)
                result = permutation_importance(
                    pipe, X_te, y_te,
                    n_repeats=n_repeats,
                    scoring="roc_auc",
                    random_state=SEED,
                    n_jobs=1,
                )
        except Exception as e:
            print(f"[{stamp()}]   fold {fold} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            continue

        fold_mean = result.importances.mean(axis=1)
        fold_means.append(fold_mean)

    if not fold_means:
        return []

    arr = np.vstack(fold_means)
    imp_mean = arr.mean(axis=0)
    imp_std = (arr.std(axis=0, ddof=1) if arr.shape[0] > 1
               else np.zeros_like(imp_mean))
    rank = (-imp_mean).argsort().argsort() + 1

    rows = []
    for i, fname in enumerate(feature_names):
        rows.append({
            "dataset": dataset,
            "model": model,
            "feature": fname,
            "importance_mean": float(imp_mean[i]),
            "importance_std": float(imp_std[i]),
            "rank": int(rank[i]),
            "n_repeats": int(n_repeats),
        })
    return rows


# ----- Consensus -----
def build_consensus(perm_df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, feature): n_models_top3 = how many models have this
    feature in their top-3. n_models = the number of models that actually
    ran on this dataset (5 or 7 for option C). The 'fraction' column makes
    it easy to compare across datasets with different denominators."""
    rows = []
    for ds in DATASETS:
        sub = perm_df[perm_df["dataset"] == ds]
        if sub.empty:
            continue
        ds_models = models_for(ds)
        n_models = len(ds_models)
        feat_models = defaultdict(list)
        for m in ds_models:
            mss = (sub[sub["model"] == m]
                   .sort_values("rank")
                   .head(TOP_K_CONSENSUS))
            for f in mss["feature"]:
                feat_models[f].append(m)
        for f in sub["feature"].unique():
            count = len(feat_models.get(f, []))
            rows.append({
                "dataset": ds,
                "feature": f,
                f"n_models_top{TOP_K_CONSENSUS}": count,
                "n_models": n_models,
                "fraction": round(count / n_models, 3) if n_models else 0.0,
                "models": ",".join(feat_models.get(f, [])),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["dataset",
                           f"n_models_top{TOP_K_CONSENSUS}",
                           "feature"],
                          ascending=[True, False, True]).reset_index(drop=True)


# ----- Figures -----
def make_figures(perm_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        sub = perm_df[perm_df["dataset"] == ds]
        if sub.empty:
            continue
        ds_models = models_for(ds)
        fig, axes = plt.subplots(1, len(ds_models),
                                 figsize=(4 * len(ds_models), 4.5))
        if len(ds_models) == 1:
            axes = [axes]
        for ax, m in zip(axes, ds_models):
            mss = sub[sub["model"] == m].sort_values("importance_mean")
            top = mss.tail(TOP_K_FIG)
            if top.empty:
                ax.set_title(f"{m}\n(no data)")
                continue
            ax.barh(top["feature"], top["importance_mean"],
                    xerr=top["importance_std"].fillna(0.0),
                    color="steelblue", error_kw={"capsize": 2})
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_title(m, fontsize=10)
            ax.set_xlabel("Δ AUROC (perm)")
            ax.tick_params(axis="y", labelsize=8)
        skip_note = ""
        if SKIP_BY_DATASET.get(ds):
            skip_note = (f"  (skipped: {','.join(sorted(SKIP_BY_DATASET[ds]))}"
                         f" — saturated dataset)")
        fig.suptitle(f"Permutation importance — {ds}{skip_note}", y=1.02)
        fig.tight_layout()
        out = FIGURES_DIR / f"perm_importance_{ds}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[{stamp()}] saved {out}", flush=True)


# ----- Main -----
def main():
    print("STARTING PERMUTATION IMPORTANCE (option C)", flush=True)
    print(f"[{stamp()}] datasets={DATASETS}", flush=True)
    print(f"[{stamp()}] models={MODELS}", flush=True)
    print(f"[{stamp()}] seed={SEED}  k_folds={K_FOLDS}  "
          f"n_repeats={N_REPEATS}", flush=True)
    for ds, sk in SKIP_BY_DATASET.items():
        print(f"[{stamp()}] SKIP on {ds}: {sorted(sk)} (saturated)",
              flush=True)

    # Device check (informational only — TabPFN auto-detects)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[{stamp()}] CUDA available: "
                  f"{torch.cuda.get_device_name(0)} "
                  f"(devices: {torch.cuda.device_count()})", flush=True)
        else:
            print(f"[{stamp()}] CUDA not available — TabPFN will fall back "
                  f"to CPU (very slow). Abort and resubmit on a GPU node.",
                  flush=True)
    except ImportError:
        print(f"[{stamp()}] torch not installed", flush=True)

    best_hp: dict[tuple[str, str], dict] = {}
    missing = []
    for ds in DATASETS:
        for m in models_for(ds):
            hp = load_best_hp(ds, m)
            if hp is None:
                missing.append((ds, m))
                continue
            best_hp[(ds, m)] = hp
    if missing:
        print(f"[{stamp()}] WARNING: missing seed=42 results for: {missing}",
              flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    t_total = time.perf_counter()

    for ds in DATASETS:
        for m in MODELS:
            if is_skipped(ds, m):
                print(f"[{stamp()}] SKIP {ds} / {m} (option C)", flush=True)
                continue
            if (ds, m) not in best_hp:
                continue
            hp = best_hp[(ds, m)]
            print(f"[{stamp()}] === {ds} / {m} (n_repeats={N_REPEATS}) ===",
                  flush=True)
            t0 = time.perf_counter()
            rows = run_one(ds, m, hp, N_REPEATS)
            dt = time.perf_counter() - t0
            all_rows.extend(rows)
            print(f"[{stamp()}] DONE {ds} / {m} in {dt:.1f}s "
                  f"({len(rows)} features)", flush=True)

    perm_df = pd.DataFrame(all_rows)
    if not perm_df.empty:
        for c in ("importance_mean", "importance_std"):
            perm_df[c] = perm_df[c].astype(float).round(6)
    perm_df.to_csv(PERM_CSV, index=False)
    print(f"[{stamp()}] saved {PERM_CSV}  rows={len(perm_df)}", flush=True)

    consensus = build_consensus(perm_df)
    consensus.to_csv(CONSENSUS_CSV, index=False)
    print(f"[{stamp()}] saved {CONSENSUS_CSV}  rows={len(consensus)}",
          flush=True)

    make_figures(perm_df)

    print(f"[{stamp()}] total runtime: "
          f"{time.perf_counter() - t_total:.1f}s", flush=True)
    print("PERMUTATION IMPORTANCE DONE", flush=True)


if __name__ == "__main__":
    main()
