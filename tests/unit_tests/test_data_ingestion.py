import pytest

# assuming these exist in main.py
from src.main import (  # assuming correct import path
    fetch_housing_data,
    final_rmse,
    housing,
    housing_labels,
    housing_prepared,
    housing_tr,
    lin_rmse,
    load_housing_data,
    tree_rmse,
)


def test_data_ingestion():
    assert housing.shape == (16512, 10)
    assert housing_labels.shape == (16512,)


def test_data_preparation():
    assert housing_tr.shape == (16512, 13)
    assert housing_prepared.shape == (16512, 13)


def test_model_training():
    assert lin_rmse == 68628.19819848923
    assert tree_rmse == 0.0


def test_model_evaluation():
    assert final_rmse == 47730.22690385927
