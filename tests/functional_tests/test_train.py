import pytest

from src.train import train_model


def test_train_model():
    best_model = train_model()
    assert best_model is not None, "Model training should return a model, got None"
