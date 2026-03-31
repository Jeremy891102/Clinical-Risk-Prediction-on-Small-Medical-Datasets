# TODO TRACKER
# Check off items as you complete them: [ ] → [x]

## ═══════════════════════════════════════════════════
##  PERSON A — Data + Models + Writing
##  Estimated total: ~28 hours
## ═══════════════════════════════════════════════════

### Week 1 (4/1 - 4/6): Data & Models  ~8 hrs
- [ ] [A1]  data_loader.py  — load_heart()              (15 min)
- [ ] [A2]  data_loader.py  — load_diabetes()            (10 min)
- [ ] [A3]  data_loader.py  — load_breast()              (5 min)
- [ ] [A4]  data_loader.py  — load_liver()               (15 min)
- [ ] [A5]  data_loader.py  — load_kidney()              (20 min)
- [ ] [A6]  data_loader.py  — preprocess()               (1 hr)
- [ ] [A7]  data_loader.py  — get_dataset_stats()        (15 min)
- [ ] [A8]  models.py       — get_logistic_regression()  (5 min)
- [ ] [A9]  models.py       — get_svm()                  (10 min)
- [ ] [A10] models.py       — get_random_forest()        (5 min)
- [ ] [A11] models.py       — get_mlp()                  (10 min)
- [ ] [A12] models.py       — get_xgboost()              (5 min)
- [ ] [A13] models.py       — get_catboost()             (10 min)
- [ ] [A14] models.py       — get_tabpfn()               (10 min)
- [ ] [A15] models.py       — PARAM_GRIDS (all models)   (20 min)
      >>> CHECKPOINT: run sanity check notebook on 1 dataset

### Week 2 (4/7 - 4/13): Debug + Support  ~4 hrs
- [ ] Fix any bugs found by Person B during full eval
- [ ] Help debug preprocessing issues per dataset
- [ ] Verify TabPFN runs correctly on GPU (NYU HPC)

### Week 3 (4/14 - 4/20): Ablation  ~5 hrs
- [ ] [A16] analysis.py — run_ablation()                 (2 hrs)
- [ ] [A17] analysis.py — efficiency_analysis()          (30 min)
      >>> CHECKPOINT: all analysis CSVs generated

### Week 4 (4/21 - 4/27): Write Paper  ~6 hrs
- [ ] Write Section 1: Introduction                      (1.5 hrs)
- [ ] Write Section 2: Related Work                      (2 hrs)
- [ ] Write Section 3: Data                              (1 hr)
      Include dataset_stats table from [B14]
- [ ] Go to TA office hours for check-in                 (30 min)

### Week 5 (4/28 - 5/8): Analysis + Polish  ~5 hrs
- [ ] Write Section 6: Analysis                          (2.5 hrs)
      This is 20% of grade — spend time here!
- [ ] Write Section 7: Conclusion                        (1 hr)
- [ ] Proofread entire paper                             (1.5 hrs)


## ═══════════════════════════════════════════════════
##  PERSON B — Evaluation + Analysis + Figures
##  Estimated total: ~32 hours
## ═══════════════════════════════════════════════════

### Week 1 (4/1 - 4/6): Evaluation Pipeline  ~7 hrs
- [ ] [B1] evaluate.py — evaluate_single()               (2 hrs)
           This is the hardest function. Take your time.
- [ ] [B2] evaluate.py — evaluate_all()                  (1.5 hrs)
- [ ] [B3] evaluate.py — save_results()                  (30 min)
- [ ] [B4] evaluate.py — run_xgboost_default()           (30 min)
      >>> CHECKPOINT: pipeline runs on 1 dataset × 2 models

### Week 2 (4/7 - 4/13): Run Full Evaluation  ~5 hrs
- [ ] Run evaluate_all() on ALL 5 datasets × 7 models × 3 seeds
      (expect ~30-60 min compute time, but debugging takes longer)
- [ ] Verify results are reasonable (check vs prior work numbers)
- [ ] Fix any issues with specific model × dataset combos
      >>> CHECKPOINT: main_results.csv and predictions.pkl exist

### Week 3 (4/14 - 4/20): Error + Feature Analysis  ~6 hrs
- [ ] [B5] analysis.py — error_analysis()                (2 hrs)
- [ ] [B6] analysis.py — feature_importance()            (1.5 hrs)
- [ ] [B7] analysis.py — statistical_test()              (1.5 hrs)
      >>> CHECKPOINT: all analysis CSVs generated

### Week 4 (4/21 - 4/27): Figures + Write  ~8 hrs
- [ ] [B8]  visualize.py — fig_main_results_table()      (1 hr)
- [ ] [B9]  visualize.py — fig_ablation_curves()         (1 hr)
- [ ] [B10] visualize.py — fig_error_overlap()           (1 hr)
- [ ] [B11] visualize.py — fig_feature_importance()      (45 min)
- [ ] [B12] visualize.py — fig_critical_difference()     (45 min)
- [ ] [B13] visualize.py — fig_efficiency()              (30 min)
- [ ] [B14] visualize.py — fig_dataset_stats()           (30 min)
- [ ] Write Section 4: Methods                           (1.5 hrs)
- [ ] Write Section 5: Results                           (1.5 hrs)

### Week 5 (4/28 - 5/8): Polish  ~6 hrs
- [ ] Finalize all figures (adjust colors, labels, sizing)
- [ ] Format LaTeX tables from CSVs
- [ ] Cross-check: does every rubric item have a section?
- [ ] Proofread, fix typos, ensure references are correct
- [ ] Final compile + submit on Gradescope


## ═══════════════════════════════════════════════════
##  RUBRIC ↔ CODE MAPPING
## ═══════════════════════════════════════════════════
##
##  Rubric Item        Weight   Code / Section
##  ──────────────────────────────────────────────
##  Proposal            10%     (already submitted)
##  Problem selection   10%     Paper Section 1 (Intro)
##  Related work        20%     Paper Section 2
##  Data                10%     Paper Section 3 + [B14]
##  Methods             10%     Paper Section 4 + models.py
##  Results             20%     Paper Section 5 + [B8-B13]
##  Analysis            20%     Paper Section 6 + [A16,B5,B6,B7]
## ═══════════════════════════════════════════════════
