import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    # generate numpy style doctrings for this function
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


def income_cat_proportions(data):
    """Calculate income category proportions.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing data.

    Returns
    -------
    pd.Series
        Series containing income category proportions.
    """
    return data["income_cat"].value_counts() / len(data)


num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)


def data_prep_main(input_dir, output_dir):
    """Main function for data preparation that takes input and output directories and creates train and test sets.

    Parameters
    ----------
    input_dir : str
        Input directory.
    output_dir : str
        Output directory.
    """
    housing = load_data(os.path.join(input_dir, "housing.csv"))

    # Create an income category attribute
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Split the data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Random split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    strat_train_set.to_csv(os.path.join(output_dir, "strat_train_set.csv"), index=False)
    strat_test_set.to_csv(os.path.join(output_dir, "strat_test_set.csv"), index=False)
    train_set.to_csv(os.path.join(output_dir, "train_set.csv"), index=False)
    test_set.to_csv(os.path.join(output_dir, "test_set.csv"), index=False)

    # Compare proportions
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Add additional attributes
    housing = strat_train_set.copy()
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing_labels = strat_train_set["median_house_value"].copy()

    # Data Imputation, Standard Scaling and One Hot Encoding using Pipeline
    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    # Save the prepared data and labels
    pd.DataFrame(housing_prepared).to_csv(
        os.path.join(output_dir, "housing_prepared.csv"), index=False
    )
    housing_labels.to_csv(os.path.join(output_dir, "housing_labels.csv"), index=False)

    # saving pipeline
    model_path = "/home/pushvinder/mle_training/artifacts/"
    with open(os.path.join(model_path, "full_pipeline.pkl"), "wb") as f:
        pickle.dump(full_pipeline, f)

    prep_info = {
        "num_samples": len(housing),
        "num_features": housing.shape[1],
        "missing_values": housing.isnull().sum().sum(),
        "output_dir": output_dir,
        "model_path": model_path,
    }

    return (output_dir, model_path, prep_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation script.")
    parser.add_argument("--input_dir", type=str, help="Input directory.")
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    args = parser.parse_args()
    log_dir = "/home/pushvinder/mle_training/logs/"
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_prep.log"),
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
    )
    logging.info("Started data preparation.")
    data_prep_main(args.input_dir, args.output_dir)
    logging.info("Finished data preparation.")
