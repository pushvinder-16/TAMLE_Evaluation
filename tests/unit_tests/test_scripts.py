import os
import pandas as pd
import pytest

from housepred.ingest_data import load_housing_data


def test_load_data():
    data = load_housing_data(os.path.join("data", "raw"))
    assert isinstance(data, pd.DataFrame), "Output should be a pandas DataFrame."
    assert not data.empty, "DataFrame should not be empty."


# run the tests
if __name__ == "__main__":
    pytest.main()
