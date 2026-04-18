import pandas as pd
import pytest

from src.data.loaders import load_heart, load_pima, load_breast_cancer, load_liver, load_ckd


@pytest.fixture(params=[
    load_heart,
    load_pima,
    load_breast_cancer,
    load_liver,
    load_ckd,
], ids=["heart", "pima", "breast_cancer", "liver", "ckd"])
def loaded_dataset(request):
    return request.param()


def test_returns_three_items(loaded_dataset):
    assert len(loaded_dataset) == 3


def test_x_is_dataframe(loaded_dataset):
    X, y, meta = loaded_dataset
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] > 0
    assert X.shape[1] > 0


def test_y_is_binary_int_series(loaded_dataset):
    X, y, meta = loaded_dataset
    assert isinstance(y, pd.Series)
    assert y.dtype == int
    assert set(y.unique()).issubset({0, 1})


def test_meta_n_samples(loaded_dataset):
    X, y, meta = loaded_dataset
    assert meta["n_samples"] == len(X) == len(y)


def test_y_no_nans(loaded_dataset):
    X, y, meta = loaded_dataset
    assert y.isna().sum() == 0
