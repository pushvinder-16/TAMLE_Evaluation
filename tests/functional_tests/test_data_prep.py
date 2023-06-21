import pytest

from housepred.data_prep import income_cat_proportions


def test_income_cat_proportions():
    proportions = income_cat_proportions(housing)
    assert proportions.sum() == 1, "The proportions should sum up to 1"
    assert isinstance(proportions, pd.Series), "Output should be a pandas Series"
