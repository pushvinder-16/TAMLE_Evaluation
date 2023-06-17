import pytest

from src.score import score_model


def test_score_model():
    score = score_model()
    assert isinstance(score, (int, float)), "Score should be a number"
    assert 0 <= score <= 100, "Score should be between 0 and 100"
