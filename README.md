## Comparing Classical and Modern ML Methods for Clinical Risk Prediction on Small Medical Datasets

**Team:** [Person A] & [Person B]  
**Track:** Applications  
**Proposal due:** March 31, 2026  
**Paper due:** May 8, 2026  

---

## Project Structure

```
ds-ga-1003-final/
├── README.md                  ← you are here
├── requirements.txt           ← pip install -r requirements.txt
│
├── src/                       ← all source code
│   ├── config.py              ← dataset/model configs, random seeds, hyperparams
│   ├── data_loader.py         ← download + preprocess all 5 datasets
│   ├── models.py              ← define all 7 models with sklearn API
│   ├── evaluate.py            ← nested CV pipeline, metrics, timing
│   ├── analysis.py            ← error analysis, feature importance, ablation
│   └── visualize.py           ← generate all tables + figures for paper
│
├── notebooks/                 ← exploration only, NOT final code
│   ├── 01_eda.ipynb           ← explore datasets, check distributions
│   └── 02_sanity_check.ipynb  ← quick test: 1 dataset, 2 models, does it run?
│
├── results/                   ← all outputs (gitignored except .gitkeep)
│   ├── tables/                ← CSV files of results
│   └── figures/               ← PNG/PDF figures for paper
│
├── paper/                     ← LaTeX source
│   ├── main.tex
│   └── references.bib
│
└── run.py                     ← main entry point: python run.py --step [all|eval|analysis|figures]
```

---

## Task Division (2 people × 5 hrs/week × 6 weeks = 60 hrs total)

### Person A: Data + Models + Writing
### Person B: Evaluation + Analysis + Figures

See detailed week-by-week plan at bottom of this README.

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd ds-ga-1003-final
pip install -r requirements.txt

# 2. Download all datasets (takes ~30 sec)
python -c "from src.data_loader import load_all_datasets; load_all_datasets()"

# 3. Run full pipeline
python run.py --step all

# 4. Run individual steps
python run.py --step eval      # run all models on all datasets
python run.py --step analysis  # ablation + error analysis + feature importance
python run.py --step figures   # generate all paper figures
```

---

## Week-by-Week Plan

| Week | Dates | Person A | Person B | Deliverable |
|------|-------|----------|----------|-------------|
| 0 | 3/29-3/31 | Submit proposal | Setup repo + sanity check | Proposal on Gradescope |
| 1 | 4/1-4/6 | `data_loader.py` + `models.py` | `evaluate.py` skeleton | Pipeline runs end-to-end on 1 dataset |
| 2 | 4/7-4/13 | Tune hyperparams, fix bugs | Run full eval on all 5 datasets | `results/tables/main_results.csv` |
| 3 | 4/14-4/20 | `analysis.py` (ablation) | `analysis.py` (error + feature imp) | All analysis CSVs done |
| 4 | 4/21-4/27 | Write Intro + Related Work + Data | Write Methods + Results + make figures | Draft v1 |
| 5 | 4/28-5/8 | Write Analysis section | Polish figures + proofread | Submit paper |

---

## ⚠️ IMPORTANT: LLM Policy
Applications track: **NO AI coding assistants** (Cursor, Claude Code, Copilot).  
All code must be written by team members. Paper must NOT be written by LLMs.
