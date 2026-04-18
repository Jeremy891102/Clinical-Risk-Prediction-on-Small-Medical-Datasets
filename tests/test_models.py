import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.registry import MODEL_REGISTRY

X_synth, y_synth = make_classification(n_samples=100, n_features=10, random_state=0)


def _get_params():
    params = []
    for name in MODEL_REGISTRY:
        if name == "tabpfn":
            params.append(pytest.param(name, marks=pytest.mark.skipif(
                not _tabpfn_available(), reason="tabpfn not installed")))
        else:
            params.append(name)
    return params


def _tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(params=_get_params())
def model_name(request):
    return request.param


def test_model_fit_predict(model_name):
    model = MODEL_REGISTRY[model_name]()
    model.fit(X_synth, y_synth)

    preds = model.predict(X_synth)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = model.predict_proba(X_synth)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
