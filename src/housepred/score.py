import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle

# from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def load_data(path):
    """Load data from csv file.

    Parameters
    ----------
    path : str
        Path to csv file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing data.
    """
    return pd.read_csv(path)


def score_main(input_dir, model_path):
    """Main function for scoring that takes input and model directories and tests the model.

    Parameters
    ----------
    input_dir : str
        Input directory.
    model_path : str
        Model directory.
    """
    # Load the data
    strat_test_set = load_data(os.path.join(input_dir, "strat_test_set.csv"))
    # imputer = SimpleImputer(strategy="median")

    strat_test_set.drop("income_cat", axis=1, inplace=True)

    # load the imputer
    with open(os.path.join(model_path, "imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)

    # Add additional attributes
    housing = strat_test_set.copy()
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    y_test = strat_test_set["median_house_value"].copy()

    # Data Imputation

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    X_test_prepared = housing_prepared.copy()

    # Load the model
    with open(os.path.join(model_path, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)

    # Score the model
    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)
    logging.info(f"Final RMSE: {final_rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input directory.")
    parser.add_argument("--model_path", type=str, help="Path of the trained model.")
    args = parser.parse_args()
    log_dir = "/home/pushvinder/mle_training/logs/"
    logging.basicConfig(
        filename=os.path.join(log_dir, "score.log"),
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
    )
    logging.info("Started scoring.")
    score_main(args.input_dir, args.model_path)
    logging.info("Finished scoring.")
