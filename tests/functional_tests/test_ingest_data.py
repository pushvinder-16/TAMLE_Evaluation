import pytest

from src.ingest_data import load_data


def test_load_data():
    dataset = load_data()
    assert isinstance(dataset, pd.DataFrame), "Loaded data should be in DataFrame format"
    assert dataset.shape[0] > 0, "Dataset should not be empty"
