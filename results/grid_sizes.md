# Hyperparameter Grid Sizes

Nested CV: outer=10-fold × inner=5-fold (GridSearchCV).
`Total fits` = grid size × inner folds × outer folds (per dataset).

| Model | Grid size | Total fits | Hyperparameters |
|---|---:|---:|---|
| logistic_regression | 5 | 250 | `C`: 5 values; `penalty`: 1 values |
| svm | 16 | 800 | `C`: 4 values; `gamma`: 4 values; `kernel`: 1 values |
| random_forest | 24 | 1200 | `n_estimators`: 3 values; `max_depth`: 4 values; `min_samples_split`: 2 values |
| mlp | 24 | 1200 | `hidden_layer_sizes`: 4 values; `alpha`: 3 values; `learning_rate_init`: 2 values |
| xgboost | 54 | 2700 | `n_estimators`: 3 values; `max_depth`: 3 values; `learning_rate`: 3 values; `subsample`: 2 values |
| catboost | 27 | 1350 | `iterations`: 3 values; `depth`: 3 values; `learning_rate`: 3 values |
| tabpfn | 0 (no tuning) | — | — |

## Detailed grids

### logistic_regression
- `C` (5): `[0.01, 0.1, 1.0, 10.0, 100.0]`
- `penalty` (1): `['l2']`

### svm
- `C` (4): `[0.1, 1.0, 10.0, 100.0]`
- `gamma` (4): `['scale', 0.01, 0.1, 1.0]`
- `kernel` (1): `['rbf']`

### random_forest
- `n_estimators` (3): `[100, 300, 500]`
- `max_depth` (4): `[None, 5, 10, 20]`
- `min_samples_split` (2): `[2, 5]`

### mlp
- `hidden_layer_sizes` (4): `[(32,), (64,), (64, 32), (128, 64)]`
- `alpha` (3): `[0.0001, 0.001, 0.01]`
- `learning_rate_init` (2): `[0.001, 0.01]`

### xgboost
- `n_estimators` (3): `[100, 300, 500]`
- `max_depth` (3): `[3, 6, 10]`
- `learning_rate` (3): `[0.01, 0.1, 0.3]`
- `subsample` (2): `[0.8, 1.0]`

### catboost
- `iterations` (3): `[100, 300, 500]`
- `depth` (3): `[4, 6, 8]`
- `learning_rate` (3): `[0.03, 0.1, 0.3]`

### tabpfn
No hyperparameter tuning (foundation model uses default settings).
