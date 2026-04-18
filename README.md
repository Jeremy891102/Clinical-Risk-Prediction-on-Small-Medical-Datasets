# Clinical Risk Prediction Benchmarking

Comparing 7 ML methods (Logistic Regression, SVM, Random Forest, MLP, XGBoost, CatBoost, TabPFN) on 5 small clinical datasets for DS-GA 1003.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Smoke test: load all datasets and run DummyClassifier
python experiments/run_smoke_test.py

# Unit tests
pytest tests/
```
