import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def load_data(path):
    return pd.read_csv(path)


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def main(input_dir, output_dir):
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

    # Data Imputation
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    # Save the prepared data and labels
    housing_prepared.to_csv(os.path.join(output_dir, "housing_prepared.csv"), index=False)
    housing_labels.to_csv(os.path.join(output_dir, "housing_labels.csv"), index=False)

    # saving imputer
    model_path = "/home/pushvinder/mle_training/artifacts/"
    with open(os.path.join(model_path, "imputer.pkl"), "wb") as f:
        pickle.dump(imputer, f)


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
    main(args.input_dir, args.output_dir)
    logging.info("Finished data preparation.")
