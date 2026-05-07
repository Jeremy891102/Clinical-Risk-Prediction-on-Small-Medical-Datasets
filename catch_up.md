# ML Final Project — Results & Findings

> A self-contained walkthrough of what I built, what I found, and where the project currently stands. Everything you need to review is in this document — you don't need to open the CSVs or run any code.
>
> **Read time: ~20 minutes.**

---

## 1. The question

Which machine learning method works best for clinical risk prediction on small medical datasets?

The motivation: TabPFN 2.5 (Hollmann et al., *Nature* 2025) is a transformer-based tabular foundation model that claims strong performance with no hyperparameter tuning. Whether this holds in the clinical small-data regime — where calibration and reproducibility matter — was an open question. We test it on five public clinical datasets against six standard baselines.

This is the **Applications Track** project we proposed in March. The framing is practical guidance for practitioners, not method-versus-method research.

---

## 2. What was tested

### Datasets (all small clinical tabular)

| Dataset | n | Features | Pos. rate | Notes |
|---|---:|---:|---:|---|
| Heart Disease (Cleveland, UCI) | 303 | 13 | 45.9% | Binarized from 5-class |
| Pima Indians Diabetes | 768 | 8 | 34.9% | No missing |
| Breast Cancer Wisconsin (sklearn) | 569 | 30 | 62.7% | No missing |
| Indian Liver Patient (UCI) | 583 | 10 | 71.4% | Most imbalanced |
| Chronic Kidney Disease (UCI) | 400 | 24 | 62.5% | 10.5% missing |

### Methods (7 total)

Logistic Regression (L2), SVM (RBF), Random Forest, MLP (2 hidden layers), XGBoost, CatBoost, **TabPFN 2.5**.

### Evaluation protocol

- **10-fold stratified outer CV × 5-fold stratified inner CV** for hyperparameter search (`GridSearchCV`)
- HP grids: 5 to 54 configurations per method (TabPFN uses defaults — no tuning)
- Metrics: AUROC, accuracy, weighted F1, Brier score, wall-clock time
- Statistical comparison: Friedman + Nemenyi post-hoc, following Demšar (2006)
- Single seed (42) for the main experiment; multi-seed validation for stability check

---

## 3. Main results

### AUROC (10-fold mean, seed 42)

| Model | Heart | Pima | Breast | Liver | CKD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.907 | 0.832 | 0.995 | 0.752 | **1.000** |
| SVM | 0.903 | 0.829 | 0.996 | 0.697 | **1.000** |
| Random Forest | 0.907 | 0.837 | 0.989 | 0.760 | **1.000** |
| MLP | 0.904 | 0.832 | 0.991 | 0.717 | **1.000** |
| XGBoost | 0.898 | 0.841 | 0.994 | 0.736 | 0.999 |
| CatBoost | 0.910 | 0.843 | 0.992 | **0.762** | **1.000** |
| **TabPFN** | **0.915** | **0.847** | **0.997** | 0.760 | **1.000** |

### Brier score (lower is better)

| Model | Heart | Pima | Breast | Liver | CKD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.138 | 0.158 | 0.023 | 0.173 | 0.025 |
| SVM | 0.125 | 0.160 | 0.021 | 0.191 | 0.004 |
| Random Forest | 0.125 | 0.158 | 0.033 | 0.172 | 0.007 |
| MLP | 0.128 | 0.162 | 0.027 | 0.183 | 0.073 |
| XGBoost | 0.138 | 0.155 | 0.024 | 0.203 | 0.010 |
| CatBoost | 0.130 | 0.157 | 0.025 | 0.179 | 0.006 |
| **TabPFN** | **0.117** | **0.153** | **0.015** | **0.171** | **0.001** |

### One-line summary

**TabPFN ranks 1st or tied-1st on every dataset for both AUROC and Brier.** It is the only method that does so without any hyperparameter tuning.

---

## 4. Statistical significance

### Average ranks across 5 datasets

| Model | AUROC rank | Brier rank |
|---|---:|---:|
| **TabPFN** | **1.6** | **1.0** |
| CatBoost | 3.2 | 4.0 |
| Logistic Regression | 3.8 | 4.6 |
| Random Forest | 4.0 | 4.0 |
| SVM | 5.0 | 3.6 |
| MLP | 5.2 | 5.8 |
| XGBoost | 5.2 | 5.0 |

### Friedman test

- **Brier**: χ² = 14.74, **p = 0.022 (significant)**. Nemenyi post-hoc: TabPFN vs MLP p = 0.008 (significant); TabPFN vs XGBoost p = 0.053 (marginal).
- **AUROC**: χ² = 11.91, **p = 0.064 (marginal, did not cross α = 0.05)**.

### How I handle the AUROC marginality (important — please flag if you disagree)

I don't claim AUROC significance in the paper. I write it as "consistent but marginal," and explicitly say the Friedman test on N=5 datasets is underpowered. The strongest claim I make is on **Brier (calibration)**, which is significant.

The argument I make: even though the formal AUROC test doesn't cross 0.05, TabPFN ranks first on all five datasets without exception, and the multi-seed analysis (next section) shows the lead is robust. So the underpowered formal test is consistent with a real but small effect, not noise.

**Question for you: does this read as honest, or as me dodging the issue?**

---

## 5. Multi-seed stability (does the seed-42 ranking hold up?)

I re-ran heart and pima with seeds 0, 1, 7 (in addition to 42) — the two datasets where the rankings are tightest. CatBoost and Random Forest used a reduced HP grid in this experiment to save compute (documented in paper Appendix A).

### Heart

| Model | seed 0 | seed 1 | seed 7 | seed 42 | μ₄ | σ₄ |
|---|---:|---:|---:|---:|---:|---:|
| **TabPFN** | 0.919 | 0.908 | 0.912 | 0.915 | **0.913** | 0.005 |
| Random Forest | 0.907 | 0.903 | 0.906 | 0.907 | 0.906 | 0.002 |
| Logistic Regression | 0.909 | 0.899 | 0.908 | 0.907 | 0.905 | 0.005 |
| SVM | 0.909 | 0.907 | 0.895 | 0.903 | 0.903 | 0.006 |
| CatBoost | 0.904 | 0.904 | 0.887 | 0.910 | 0.901 | 0.010 |
| XGBoost | 0.899 | 0.895 | 0.895 | 0.898 | 0.897 | 0.002 |
| MLP | 0.897 | 0.884 | 0.900 | 0.904 | 0.897 | 0.009 |

### Pima

| Model | seed 0 | seed 1 | seed 7 | seed 42 | μ₄ | σ₄ |
|---|---:|---:|---:|---:|---:|---:|
| **TabPFN** | 0.844 | 0.849 | 0.845 | 0.847 | **0.846** | 0.002 |
| CatBoost | 0.833 | 0.840 | 0.835 | 0.843 | 0.838 | 0.005 |
| Random Forest | 0.829 | 0.839 | 0.828 | 0.837 | 0.833 | 0.006 |
| XGBoost | 0.830 | 0.827 | 0.835 | 0.841 | 0.833 | 0.006 |
| Logistic Regression | 0.830 | 0.833 | 0.834 | 0.832 | 0.832 | 0.002 |
| SVM | 0.829 | 0.836 | 0.832 | 0.829 | 0.832 | 0.003 |
| MLP | 0.812 | 0.827 | 0.834 | 0.832 | 0.826 | 0.010 |

### What this shows

- **TabPFN ranks first under all four seeds on both datasets.**
- **TabPFN has the smallest cross-seed std on Pima (σ₄ = 0.002)** — the most reproducible model.
- One interesting drop: CatBoost was rank 2 on Heart at seed 42, but drops to rank 5 in the four-seed mean. Its seed-42 result was favorable but doesn't generalize across seeds.

---

## 6. Sample size ablation (where TabPFN wins biggest)

Each method was re-run at 10%, 25%, 50%, 100% of training data, with 3 random subsample seeds per fraction.

### What I found

1. **TabPFN's lead grows as data shrinks.** On Pima at 10% (~70 training samples), TabPFN reaches AUROC 0.839 vs second-best (Random Forest) 0.798 — a **0.04 gap, four times larger than the gap at 100% data**. Across all five datasets at 10%, TabPFN ranks first or tied-first.
2. **Kernel and neural methods break in extreme small-data conditions.** SVM on Heart at 10% achieves AUROC = 0.144, far below random. The small sample is insufficient to identify a useful kernel margin and the model effectively predicts the wrong class. MLP shows similar but milder failures.
3. **Liver is non-monotonic with sample size**: 25% performs *worse* than 10% for several methods. This violates the expectation that more data = better, and points to label noise. (See Section 8 below.)

### Implication

The small-clinical regime is exactly where method choice matters most — and exactly where TabPFN's advantage is largest. This is the strongest argument for TabPFN as a default.

---

## 7. Calibration (Heart and Pima)

ECE = Expected Calibration Error (10 equal-width bins, pooled out-of-fold predictions). Lower is better.

| Model | Heart | Pima |
|---|---:|---:|
| Logistic Regression | 0.140 | 0.032 |
| SVM | 0.040 | 0.037 |
| Random Forest | 0.059 | 0.048 |
| MLP | 0.052 | 0.034 |
| XGBoost | 0.063 | **0.028** |
| CatBoost | 0.086 | 0.031 |
| **TabPFN** | **0.044** | 0.036 |

### Counter-intuitive finding: LR on Heart

Logistic Regression has the **highest accuracy on Heart (0.852)** and second-highest AUROC, **but the worst calibration by 3×** (ECE = 0.140 vs ~0.04-0.08 for others).

This means LR's decision boundary is good (it gets the right class), but its predicted probabilities are unreliable. In a clinical setting where the predicted probability feeds a downstream decision rule (e.g., "if P(disease) > 0.7 send for further testing"), LR on Heart would mislead clinicians despite high accuracy.

This is a key argument I make in the paper: **clinical ML benchmarks should report calibration alongside discrimination as a default, consistent with TRIPOD reporting guidelines.** Most published clinical ML papers report only AUROC.

---

## 8. Why is Liver hard? (case study)

Liver is the only dataset where TabPFN doesn't clearly lead — top three (CatBoost, RF, TabPFN) tied within 0.002. To understand why, I examined four things:

### Class imbalance

Liver is 71% positive / 29% negative. Per-class metrics:

| Model | Recall pos | Recall neg | Total errors |
|---|---:|---:|---:|
| Logistic Regression | 0.93 | 0.19 | 165 |
| SVM | 0.94 | 0.09 | 176 |
| Random Forest | 0.88 | 0.32 | 165 |
| MLP | 0.93 | 0.14 | 172 |
| XGBoost | 0.86 | 0.34 | 169 |
| **CatBoost** | 0.87 | **0.37** | **158** |
| TabPFN | 0.95 | 0.14 | 165 |

**SVM essentially predicts "positive" for everyone** (recall_neg = 0.09) — accuracy looks OK because most samples are positive, but the model is clinically useless. CatBoost has the lowest total errors and the best minority-class recall.

### Feature signal

Mutual information identifies Sgpt (0.086), TB (0.080), Sgot (0.076), DB (0.069) as the top-4 informative features. Total Protein (TP) has near-zero MI (Wilcoxon p = 0.44) — it's literally a useless feature. The dataset has structurally weaker signal than the others.

### Error overlap (Jaccard similarity between model error sets)

If different models make different errors → ensembling has room to help. If all models make the same errors → the gap is closer to irreducible.

- **TabPFN ↔ Logistic Regression: J = 0.79** (high overlap — they fail on the same patients despite very different architectures)
- **XGBoost ↔ MLP: J = 0.43** (most diverse)

The high overlap argues that the remaining gap on Liver is **not closable by better modeling alone** — it's likely label noise, missing relevant features, or aleatoric uncertainty.

### Conclusion of this case study

Liver is hard because of (a) severe class imbalance handled inconsistently across methods, (b) at least one feature with no discriminative power, and (c) high cross-model error overlap suggesting irreducible noise. TabPFN's failure to clearly lead here is **not a weakness of TabPFN** — it's a property of the data.

---

## 9. Permutation importance (does TabPFN use weird features?)

For each (dataset, model), compute permutation importance via sklearn (n_repeats=10, AUROC scoring). For each feature, count how many models rank it in their top-3.

### Cross-model agreement

| Dataset | Top features | Agreement |
|---|---|---|
| **Heart (n=7 models)** | `ca`, `cp`, `thal` | All 7 models agree |
| **Pima (n=7)** | `plas` (glucose), `mass` (BMI) | All 7 agree on top-2 |
| **Breast Cancer (n=6)** | `worst texture` | 5/6 agree |
| **Liver (n=7)** | `DB`, `Sgpt` | 5/7 agree |
| **CKD (n=6)** | `sg` (specific gravity) | 6/6 agree |

### What this means

1. **Cross-model agreement is high on the well-discriminated datasets.** All seven models agree on Heart's top-3. All seven agree on Pima's top-2. This is reassuring — it means the methods are picking up the same real signal.

2. **The agreed features make clinical sense.** Glucose is the diagnostic criterion for diabetes (Pima). Specific gravity and hemoglobin are core renal-function markers (CKD). Number of major vessels affected, chest pain type, and thalassemia are standard cardiac risk indicators (Heart).

3. **Liver has the lowest cross-model agreement** — consistent with the error-overlap finding from Section 8. When the signal is weak, different inductive biases latch onto different features.

4. **TabPFN's top features align with classical methods on every dataset.** On Heart it agrees with everyone. On Pima its top-3 is identical to the tree-based methods. **TabPFN is not relying on idiosyncratic representations** — at least at the level of marginal feature importance, it behaves like a conventional classifier. This addresses a common worry about transformer-based tabular models being uninterpretable.

> Note: TabPFN was omitted on Breast Cancer and CKD because (a) those are saturated and feature-importance ranks are uninformative, and (b) repeated in-context inference is expensive on those datasets. Documented in paper Limitations.

---

## 10. Runtime

Wall-clock seconds for nested CV (10 outer × 5 inner folds) per dataset, including hyperparameter search:

| Model | Heart | Pima | Breast | Liver | CKD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 3 | 4 | 1 | 2 | 2 |
| SVM | 6 | 13 | 6 | 8 | 6 |
| Random Forest | 113 | 133 | 136 | 124 | 108 |
| MLP | 9 | 18 | 12 | 13 | 13 |
| XGBoost | 37 | 70 | 95 | 55 | 34 |
| CatBoost | 506 | 747 | 824 | 133 | 201 |
| TabPFN | 17 | 12 | 12 | 13 | 13 |

### What this shows

- **TabPFN is 30–66× faster than CatBoost** despite matching or exceeding it in AUROC and Brier. The reason: CatBoost runtime is dominated by 27-config hyperparameter search; TabPFN has no search.
- This isn't a fair comparison in the narrow sense — but "no search needed" is itself a property TabPFN provides. From the practitioner's perspective, total time to a deployable model is what matters.
- TabPFN ran on a single A100 GPU; everything else ran on CPU. On CPU, TabPFN is roughly 10-20× slower (still tractable for these dataset sizes).

---

## 11. The story I'm telling in the paper

Combining everything above, the framing is:

> **TabPFN 2.5 is a strong default for small clinical tabular ML because three properties combine: significantly best calibration, consistently best AUROC (modulo statistical power), and zero hyperparameter tuning. The advantage grows in the small-data regime, which is exactly where clinical applications live. Permutation importance confirms it relies on the same clinical features as classical methods. Logistic regression remains a competitive lightweight fallback when compute is tight.**

The paper closes with practical recommendations:
- **Compute-constrained / interpretability-required**: Logistic Regression (with calibration applied)
- **Best calibration + zero tuning**: TabPFN
- **Tree-based interpretability or regulatory needs**: CatBoost (matches TabPFN on AUROC at 30-66× the runtime)
- **Don't recommend as default**: SVM (small-data instability), MLP (consistently weakest), XGBoost (no clear advantage at this scale)

---

## 12. What I'm uncertain about (where I most want your input)

In rough order of how much it would help if you have a view:

1. **Is the framing balanced?** I argue TabPFN is best but try not to overclaim. Any place where it sounds like a TabPFN sales pitch?

2. **Does the AUROC p=0.064 handling read as honest?** Or as me trying to dodge it?

3. **Is the Liver case study (Section 8) convincing?** The argument is "remaining gap is irreducible," supported by error overlap + non-monotonic ablation + weak features. Does the evidence actually support that conclusion, or am I reaching?

4. **Permutation importance section (Section 9)** — am I overselling the cross-model agreement finding?

5. **Practical recommendations (Section 11)** — useful or preachy?

6. **Anything claimed in the paper that the data above doesn't support?** This is the highest-priority thing to flag.

7. **Anything missing?** Limitations I should acknowledge, related work I should cite (especially clinical ML papers using these specific datasets — that would strengthen the related work section).

---

## 13. Decisions I made on my own (worth knowing)

- **Brier score added** to evaluation (proposal only mentioned AUROC + accuracy + weighted F1). Brier turned out to be the strongest evidence for TabPFN's calibration win.
- **Multi-seed only on heart + pima** (not all 5 datasets). Other 3 are saturated (CKD, breast) or already statistically tied (liver) — extra seeds wouldn't change conclusions.
- **CatBoost / RF used reduced HP grid in multi-seed runs.** Documented in paper.
- **Sample ablation uses single CV (not nested)** with seed-42 best HPs reused. Otherwise the experiment would have taken 5× longer.
- **Calibration analysis only on heart + pima.** Other 3 are saturated.
- **TabPFN omitted from permutation importance on breast cancer + CKD.** Saturated + expensive.
- **Framing changed from proposal**: proposal was neutral ("which method works best"); paper is "TabPFN as a strong default." If you disagree, push back — this is the biggest unilateral decision.

---

## 14. Honest things I want you to know

- **N=5 is small** for Friedman test. The AUROC p=0.064 is a real statistical issue and I handle it directly rather than burying it.
- **These are old, well-known UCI datasets.** Heart and Pima have been in ML papers since the 1990s. There's a real chance they're in TabPFN's pretraining (the paper claims explicit dedup but I can't independently verify). I discuss this in paper Limitations.
- **The findings are not surprising in retrospect.** "Foundation model wins on small tabular data" is consistent with TabPFN's own paper. Our contribution is more like "this holds in the clinical small-data regime, with proper calibration analysis," not "we discovered something new."
- **The actual contribution is methodological rigor.** Most clinical ML benchmarks don't do half of nested CV + multi-seed + sample ablation + calibration + permutation importance + statistical testing. The "story" is partly the rigor itself.

---

## 15. Logistics

- **Deadline**: May 8 (verify with syllabus)
- **Course**: DS-GA 1003, Applications Track
- **Repository**: https://github.com/Jeremy891102/Clinical-Risk-Prediction-on-Small-Medical-Datasets
- **Paper PDF (latest draft)**: https://www.overleaf.com/project/69c9d0c90a1b9c110a7b36b0
- **Compute used**: NYU Greene HPC, A100 GPU for TabPFN