# Clinical Risk Prediction Benchmarking

Benchmark of 7 tabular ML methods (Logistic Regression, SVM, Random Forest,
MLP, XGBoost, CatBoost, TabPFN) on 5 small clinical datasets for DS-GA 1003.

## Main finding

TabPFN is the strongest overall method in this benchmark, but the report avoids
claiming a universal win. In the seed-42 nested-CV results, TabPFN ranks first
on Heart, Pima, and Breast Cancer by AUROC, ties on the saturated CKD dataset,
and is within 0.002 AUROC of CatBoost on Liver. It ranks first on all five
datasets by Brier score. The Brier-score advantage is significant under the
Friedman test (p = 0.022), while the AUROC advantage is consistent but marginal
(p = 0.064).

Calibration is metric-dependent: TabPFN has the best Brier scores, but it is
not uniformly best by ECE (SVM is best on Heart; XGBoost is best on Pima). The
sample-size ablation shows the clearest small-data gain on Pima, while
classical baselines remain competitive or better on some extreme 10% subsamples
such as Heart and Liver.

## Setup

Recommended: Python 3.11 or 3.12. TabPFN v2.5 weights require a valid
Prior Labs license/API token supplied outside the repo.

```bash
pip install -r requirements.txt
```

## Run

```bash
# Smoke test: load all datasets and run DummyClassifier
python experiments/run_smoke_test.py

# Main nested-CV runner
python experiments/run_nested_cv.py --model logistic_regression --dataset heart
python experiments/run_nested_cv.py --model all --dataset all --seed 42

# Unit tests
pytest tests/
```

The actively maintained pipeline lives under `src/data/`, `src/models/`, and
`src/evaluation/`. Analysis/figure generation scripts are in `scripts/task*.py`.
The top-level `run.py` and legacy `src/*.py` modules are early planning stubs.

On macOS, XGBoost may need OpenMP (`libomp.dylib`) available at runtime.

## Report

The final report is maintained separately in Overleaf. Its reproducibility notes
document the main entry point, seeds, nested-CV protocol, and TabPFN
license/checkpoint requirements.

## Reproducibility notes

- Main entry point: `experiments/run_nested_cv.py`
- Main seed: `42`
- Stability seeds: `0`, `1`, `7`, and `42` on Heart and Pima
- Main protocol: stratified 10-fold outer CV with stratified 5-fold inner
  `GridSearchCV` for tuned baselines
- TabPFN protocol: zero-shot/default configuration, no inner hyperparameter
  search
- TabPFN checkpoint: pure-synthetic TabPFN v2.5 default checkpoint pinned in
  `src/models/tabpfn_model.py` via `ModelVersion.V2_5`
- TabPFN license: v2.5 weights require a Prior Labs license/API token supplied
  outside the repository, e.g. `TABPFN_TOKEN` or
  `tabpfn.browser_auth.save_token`
- No TabPFN license token or downloaded checkpoint should be committed

For license-free local testing, run the non-TabPFN test subset:

```bash
pytest tests/ -k "not tabpfn"
```

Current local verification (2026-05-08): the license-free test subset passed
with `39 passed, 2 deselected` after allowing network access for UCI dataset
fetching. The full test suite is not expected to be fully reproducible without a
valid TabPFN v2.5 license/API token and access to the corresponding model
weights.
