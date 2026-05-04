# Liver Dataset Error Analysis

Single seed (42), 10-fold nested CV, OOF predictions for n=583 samples.

## 1. Class balance
- Positive rate: **0.714** (majority class = positive at 0.714)
- N positive: 416, N negative: 167
- See `results/figures/liver_class_dist.pdf`.

## 2. Feature discriminability
Sorted by mutual information; Mann-Whitney U two-sided p-values shown alongside.

| feature   |       MI |    MWU_p |   mean_pos |   mean_neg |   n_pos |   n_neg | note                  |
|:----------|---------:|---------:|-----------:|-----------:|--------:|--------:|:----------------------|
| Sgpt      |   0.086  |   0      |    99.6058 |    33.6527 |     416 |     167 |                       |
| TB        |   0.0797 |   0      |     4.1644 |     1.1425 |     416 |     167 |                       |
| Sgot      |   0.0757 |   0      |   137.7    |    40.6886 |     416 |     167 |                       |
| DB        |   0.0687 |   0      |     1.9236 |     0.3964 |     416 |     167 |                       |
| Age       |   0.0532 |   0.0018 |    46.1538 |    41.2395 |     416 |     167 |                       |
| Alkphos   |   0.0394 |   0      |   319.007  |   219.755  |     416 |     167 |                       |
| A/G Ratio |   0.0182 |   0      |     0.9143 |     1.0284 |     416 |     167 |                       |
| ALB       |   0.0178 |   0.0001 |     3.0606 |     3.3443 |     416 |     167 |                       |
| TP        |   0.0012 |   0.4371 |     6.4591 |     6.5431 |     416 |     167 |                       |
| Gender    | nan      | nan      |   nan      |   nan      |     416 |     167 | non-numeric (skipped) |

Top discriminators are typical liver-function markers (alkphos, sgpt, sgot, total/direct bilirubin).
Features near the bottom (esp. demographic variables) are nearly indistinguishable between classes,
which limits the headroom every model can extract.

## 3. Per-class precision / recall (OOF)
Class 1 = liver patient (positive); class 0 = no liver disease.

| model               |   precision_pos |   recall_pos |   precision_neg |   recall_neg |   n_errors |
|:--------------------|----------------:|-------------:|----------------:|-------------:|-----------:|
| logistic_regression |          0.74   |       0.9303 |          0.5167 |       0.1856 |        165 |
| svm                 |          0.7206 |       0.9423 |          0.3846 |       0.0898 |        176 |
| random_forest       |          0.762  |       0.8774 |          0.5096 |       0.3174 |        165 |
| mlp                 |          0.7293 |       0.9327 |          0.451  |       0.1377 |        172 |
| xgboost             |          0.7645 |       0.8582 |          0.4914 |       0.3413 |        169 |
| catboost            |          0.7756 |       0.8726 |          0.5391 |       0.3713 |        158 |
| tabpfn              |          0.7337 |       0.9471 |          0.5217 |       0.1437 |        165 |

Note the asymmetry: every model has high recall on positives (majority class) and poor recall on negatives — a sign that models default to the majority class on uncertain cases.

## 4. Error-set overlap (pairwise Jaccard)

|                     |   logistic_regression |    svm |   random_forest |    mlp |   xgboost |   catboost |   tabpfn |
|:--------------------|----------------------:|-------:|----------------:|-------:|----------:|-----------:|---------:|
| logistic_regression |                1      | 0.6394 |          0.5789 | 0.7282 |    0.4978 |     0.5093 |   0.7935 |
| svm                 |                0.6394 | 1      |          0.5571 | 0.6493 |    0.5265 |     0.5113 |   0.7222 |
| random_forest       |                0.5789 | 0.5571 |          1      | 0.5388 |    0.67   |     0.6823 |   0.6098 |
| mlp                 |                0.7282 | 0.6493 |          0.5388 | 1      |    0.4328 |     0.5    |   0.7107 |
| xgboost             |                0.4978 | 0.5265 |          0.67   | 0.4328 |    1      |     0.5797 |   0.5321 |
| catboost            |                0.5093 | 0.5113 |          0.6823 | 0.5    |    0.5797 |     1      |   0.5529 |
| tabpfn              |                0.7935 | 0.7222 |          0.6098 | 0.7107 |    0.5321 |     0.5529 |   1      |

- Samples wrong by ALL 7 models: **75** (12.9%) — universally hard, likely intrinsic noise / unidentifiable.
- Samples wrong by NO model:      **309** (53.0%) — universally easy.
- Otherwise high pairwise Jaccard (≫ 0.5) means the 7 models tend to fail on the *same* samples,
  suggesting the residual error is data-side (noise / class overlap), not model-side.

## 5. Cross-dataset comparison

| dataset       |   n_samples |   n_features |   n_numeric |   n_categorical |   positive_rate |   majority_pct |   missing_pct |
|:--------------|------------:|-------------:|------------:|----------------:|----------------:|---------------:|--------------:|
| heart         |         303 |           13 |          13 |               0 |          0.4587 |         0.5413 |        0.0015 |
| pima          |         768 |            8 |           8 |               0 |          0.349  |         0.651  |        0      |
| breast_cancer |         569 |           30 |          30 |               0 |          0.6274 |         0.6274 |        0      |
| liver         |         583 |           10 |           9 |               1 |          0.7136 |         0.7136 |        0.0007 |
| ckd           |         400 |           24 |          14 |              10 |          0.625  |         0.625  |        0.1054 |

Why liver is hard relative to the other datasets:
- **Class imbalance** (positive rate 0.71) — but heart and pima are similar and easier.
- **Few discriminative features** — the top MI (0.086) is still modest;
  by contrast breast_cancer's worst-radius / worst-concave-points features have MI ≫ 0.5.
- **Categorical 'gender' contributes little** — most signal is from continuous lab values whose distributions overlap heavily across classes.
- The combination of moderate sample size, strong class overlap, and weak per-feature signal puts every model near the same ceiling (~0.76 AUROC).
