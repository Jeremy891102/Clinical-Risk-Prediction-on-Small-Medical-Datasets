from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVMModel
from src.models.random_forest import RandomForestModel
from src.models.mlp import MLPModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.tabpfn_model import TabPFNModel

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "svm": SVMModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "tabpfn": TabPFNModel,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
