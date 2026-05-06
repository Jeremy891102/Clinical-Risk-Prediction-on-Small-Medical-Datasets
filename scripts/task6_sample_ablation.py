"""Task 6: Sample size ablation across 5 datasets x 7 models.

For each (dataset, model), evaluate on stratified subsamples of size
{10%, 25%, 50%, 100%} of the data using the best HP found by the
seed=42 nested-CV main experiment (mode of best_params across the 10
outer folds; TabPFN runs default zero-shot — no HP). 10-fold stratified
CV (no nested CV); drop to 5-fold when 10-fold would give a min test-fold
size below DOWNGRADE_THRESHOLD. Skip a (dataset, fraction) when 5-fold
itself would give a min test-fold size below SKIP_MIN_FOLD_SIZE.

Note on the skip threshold: the spec said "skip if min fold size < 20".
Applied literally, every 10% subset on every dataset would be skipped
(largest 10% subset is pima at 77 samples -> 5-fold gives 15 per fold),
which defeats the ablation. We set SKIP_MIN_FOLD_SIZE=5 (still requires
both classes to be representable) and DOWNGRADE_THRESHOLD=20 (matches
the user's intent for switching from 10-fold to 5-fold). Adjust the
constants below to recover the literal interpretation.

Each (dataset, model, fraction) is averaged over 3 seeds (0, 1, 42).

Outputs:
    results/sample_ablation_raw.csv       (per-fold rows)
    results/sample_ablation_summary.csv   (per-(dataset, model, fraction) mean +/- std)
    figures/sample_ablation.pdf           (5 subplots: AUROC vs fraction, 7 lines)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.registry import get_dataset  # noqa: E402
from src.models.registry import get_model  # noqa: E402

# ----- Config -----
DATASETS = ["heart", "pima", "breast_cancer", "liver", "ckd"]
MODELS = ["logistic_regression", "svm", "random_forest", "mlp",
          "xgboost", "catboost", "tabpfn"]
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 42]

DEFAULT_K = 10
FALLBACK_K = 5
DOWNGRADE_THRESHOLD = 20  # if 10-fold min test fold < this, drop to 5-fold
SKIP_MIN_FOLD_SIZE = 5    # skip if even 5-fold min test fold < this

MIN_RAM_GB = 3.0  # warn (don't abort) if available RAM below this
RUNTIME_CEILING_HOURS = 4.0

SEED42_RAW = ROOT / "results" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RAW_CSV = RESULTS_DIR / "sample_ablation_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "sample_ablation_summary.csv"
FIG_PDF = FIGURES_DIR / "sample_ablation.pdf"

# Cached dataset sizes (so the runtime estimate doesn't need to load them).
DATASET_SIZES = {
    "heart": 303,
    "pima": 768,
    "breast_cancer": 569,
    "liver": 583,
    "ckd": 400,
}

# Estimated seconds for ONE 10-fold sample-ablation run (single HP, no inner
# search) at fraction=1.0. Calibrated from seed-42 nested-CV totals divided
# by (5 * |grid|) since nested has 10*5*|grid| fits and ablation has 10.
EST_SEC_FULL = {
    "logistic_regression": 1.0,
    "svm": 3.0,
    "random_forest": 5.0,
    "mlp": 6.0,
    "xgboost": 5.0,
    "catboost": 25.0,
    "tabpfn": 25.0,  # CPU per-fold ~2.5s
}


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    """Return the most frequent best_params across the outer folds, with
    the 'classifier__' prefix stripped. Lists are converted to tuples."""
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
    """Returns {} for tabpfn (zero-shot, no HP); None if seed=42 result missing."""
    if model == "tabpfn":
        return {}
    f = SEED42_RAW / f"{dataset}_{model}_seed42.json"
    if not f.exists():
        return None
    return mode_best_params(f)


def choose_k(n_subset: int) -> tuple[int | None, str]:
    """Pick k (10 by default, 5 fallback, None = skip) and a human note."""
    if n_subset // DEFAULT_K >= DOWNGRADE_THRESHOLD:
        return DEFAULT_K, "10-fold"
    if n_subset // FALLBACK_K >= SKIP_MIN_FOLD_SIZE:
        return FALLBACK_K, (f"5-fold (10-fold would give min test fold "
                            f"< {DOWNGRADE_THRESHOLD})")
    return None, f"skip (5-fold min test fold < {SKIP_MIN_FOLD_SIZE})"


def stratified_subsample(X, y, fraction: float, seed: int):
    if fraction >= 1.0:
        return X, np.asarray(y)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=seed
    )
    return X_sub, np.asarray(y_sub)


def fit_and_score(model_name: str, hp: dict, X_tr, y_tr, X_te, y_te, seed: int):
    wrapper = get_model(model_name, random_state=seed)
    estimator = wrapper.get_estimator()
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_tr)),
        ("classifier", estimator),
    ])
    if hp:
        pipe.set_params(**{f"classifier__{k}": v for k, v in hp.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_weighted": float(f1_score(y_te, y_pred, average="weighted")),
    }


def run_one(dataset, model, fraction, seed, hp, X_full, y_full):
    X_sub, y_sub = stratified_subsample(X_full, y_full, fraction, seed)
    n_sub = len(X_sub)
    k, note = choose_k(n_sub)
    if k is None:
        return [], k, n_sub, note

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_sub, y_sub)):
        X_tr = X_sub.iloc[tr_idx] if hasattr(X_sub, "iloc") else X_sub[tr_idx]
        X_te = X_sub.iloc[te_idx] if hasattr(X_sub, "iloc") else X_sub[te_idx]
        y_tr_v = y_sub[tr_idx]
        y_te_v = y_sub[te_idx]
        try:
            metrics = fit_and_score(model, hp, X_tr, y_tr_v, X_te, y_te_v, seed)
        except Exception as e:
            print(f"[{stamp()}]   fold {fold} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            continue
        rows.append({
            "dataset": dataset,
            "model": model,
            "fraction": fraction,
            "seed": seed,
            "fold": fold,
            "k_folds": k,
            "n_subset": n_sub,
            **metrics,
        })
    return rows, k, n_sub, note


def estimate_runtime() -> float:
    total = 0.0
    for ds in DATASETS:
        n_total = DATASET_SIZES[ds]
        for model in MODELS:
            base = EST_SEC_FULL[model]
            for frac in FRACTIONS:
                n_sub = int(n_total * frac)
                k, _ = choose_k(n_sub)
                if k is None:
                    continue
                # Compute scales sublinearly with n for tree/MLP fits; use
                # max(0.2, frac) as a rough lower floor for setup overhead.
                scale = max(0.2, frac)
                k_ratio = k / DEFAULT_K
                total += base * scale * k_ratio * len(SEEDS)
    return total


def check_ram() -> float | None:
    try:
        import psutil
    except ImportError:
        print(f"[{stamp()}] psutil not installed; skipping RAM check", flush=True)
        return None
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"[{stamp()}] available RAM: {avail_gb:.2f} GB "
          f"(total {psutil.virtual_memory().total / 1024**3:.1f} GB)", flush=True)
    if avail_gb < MIN_RAM_GB:
        print(f"[{stamp()}] WARNING: only {avail_gb:.2f} GB free, "
              f"< {MIN_RAM_GB} GB recommended. Continuing — close other apps "
              f"if you see OOM crashes.", flush=True)
    return avail_gb


def make_figure(summary_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    cmap = plt.get_cmap("tab10")
    for ax, ds in zip(axes, DATASETS):
        sub = summary_df[summary_df["dataset"] == ds]
        for i, m in enumerate(MODELS):
            row = sub[sub["model"] == m].sort_values("fraction")
            if row.empty:
                continue
            ax.errorbar(
                row["fraction"], row["mean_auroc"],
                yerr=row["std_auroc"].fillna(0.0),
                marker="o", label=m, capsize=2, lw=1.2, ms=4,
                color=cmap(i),
            )
        ax.set_title(ds)
        ax.set_xlabel("training fraction")
        ax.set_ylabel("AUROC")
        ax.set_xticks(FRACTIONS)
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="lower right", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"[{stamp()}] saved {FIG_PDF}", flush=True)


def main():
    print("STARTING SAMPLE ABLATION", flush=True)
    print(f"[{stamp()}] datasets={DATASETS}", flush=True)
    print(f"[{stamp()}] models={MODELS}", flush=True)
    print(f"[{stamp()}] fractions={FRACTIONS}  seeds={SEEDS}", flush=True)

    check_ram()

    est = estimate_runtime()
    print(f"[{stamp()}] estimated total runtime: {est:.0f} s "
          f"= {est/60:.1f} min = {est/3600:.2f} h", flush=True)
    if est > RUNTIME_CEILING_HOURS * 3600:
        print(f"[{stamp()}] estimate > {RUNTIME_CEILING_HOURS} h. "
              f"Suggestions to shrink (pick one and edit the constants):",
              flush=True)
        print(f"    - SEEDS=[42]            -> ~{est/3:.0f} s "
              f"(1 seed instead of 3)", flush=True)
        print(f"    - SEEDS=[0, 42]         -> ~{est*2/3:.0f} s "
              f"(2 seeds)", flush=True)
        print(f"    - drop 'tabpfn' from MODELS  -> -~17%", flush=True)
        print(f"    - drop 'catboost' from MODELS -> -~30%", flush=True)
        print(f"[{stamp()}] aborting; rerun after editing.", flush=True)
        sys.exit(2)

    best_hp = {}
    missing = []
    for ds in DATASETS:
        for m in MODELS:
            hp = load_best_hp(ds, m)
            if hp is None:
                missing.append((ds, m))
                continue
            best_hp[(ds, m)] = hp
    if missing:
        print(f"[{stamp()}] WARNING: missing seed=42 results for: {missing}",
              flush=True)

    print(f"[{stamp()}] best HPs (mode across 10 outer folds):", flush=True)
    for (ds, m), hp in best_hp.items():
        if hp:
            print(f"    {ds:>15} / {m:<22} : {hp}", flush=True)
        else:
            print(f"    {ds:>15} / {m:<22} : (zero-shot / no HP)", flush=True)

    skipped = []
    for ds in DATASETS:
        n_total = DATASET_SIZES[ds]
        for f in FRACTIONS:
            n_sub = int(n_total * f)
            k, note = choose_k(n_sub)
            if k is None:
                skipped.append((ds, f, n_sub, note))
    if skipped:
        print(f"[{stamp()}] SKIP combos (insufficient subset size):",
              flush=True)
        for s in skipped:
            print(f"    - dataset={s[0]} fraction={s[1]} n_sub={s[2]}: {s[3]}",
                  flush=True)

    print(f"[{stamp()}] loading datasets...", flush=True)
    data_cache = {}
    for ds in DATASETS:
        X, y, _ = get_dataset(ds)
        data_cache[ds] = (X, np.asarray(y))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    t_total = time.perf_counter()
    for ds in DATASETS:
        X_full, y_full = data_cache[ds]
        for m in MODELS:
            if (ds, m) not in best_hp:
                continue
            hp = best_hp[(ds, m)]
            for frac in FRACTIONS:
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    rows, k, n_sub, note = run_one(
                        ds, m, frac, seed, hp, X_full, y_full
                    )
                    dt = time.perf_counter() - t0
                    if not rows:
                        print(f"[{stamp()}]   {ds}/{m}/frac={frac}/seed={seed}"
                              f": SKIPPED ({note})", flush=True)
                        continue
                    all_rows.extend(rows)
                    auroc_mean = float(np.mean([r["auroc"] for r in rows]))
                    print(f"[{stamp()}]   {ds}/{m}/frac={frac}/seed={seed}"
                          f": k={k} n_sub={n_sub} AUROC={auroc_mean:.4f}  "
                          f"({dt:.1f}s)", flush=True)

    raw_df = pd.DataFrame(all_rows)
    if not raw_df.empty:
        for c in ("auroc", "accuracy", "f1_weighted"):
            raw_df[c] = raw_df[c].round(6)
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"[{stamp()}] saved {RAW_CSV}  rows={len(raw_df)}", flush=True)

    if not raw_df.empty:
        agg = (raw_df.groupby(["dataset", "model", "fraction"], observed=True)
                     .agg(mean_auroc=("auroc", "mean"),
                          std_auroc=("auroc", "std"),
                          mean_accuracy=("accuracy", "mean"),
                          mean_f1_weighted=("f1_weighted", "mean"),
                          n_seeds=("seed", "nunique"),
                          n_folds_total=("auroc", "count"),
                          k_folds=("k_folds", "first"),
                          n_subset=("n_subset", "first"))
                     .reset_index())
        for c in ("mean_auroc", "std_auroc",
                  "mean_accuracy", "mean_f1_weighted"):
            agg[c] = agg[c].astype(float).round(4)
        agg.to_csv(SUMMARY_CSV, index=False)
        print(f"[{stamp()}] saved {SUMMARY_CSV}", flush=True)
        make_figure(agg)
    else:
        print(f"[{stamp()}] NO DATA — summary/figure skipped", flush=True)

    print(f"[{stamp()}] total runtime: "
          f"{time.perf_counter() - t_total:.1f}s", flush=True)
    print("SAMPLE ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
