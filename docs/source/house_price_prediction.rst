House Price Prediction
======================

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

- Linear regression
- Decision Tree
- Random Forest


Getting Started
===============

These instructions will guide you on how to run the package on your local machine for development and testing purposes.


Prerequisites
=============

You need Python 3.x and the following Python libraries installed:

- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- Scipy
- Six

You can install these packages using pip:

.. code-block:: bash

   pip install numpy pandas matplotlib scikit-learn scipy six 


Installing and Running the Package
==================================

Clone the repository to your local machine:

.. code-block:: bash

   git clone https://github.com/your_username/house-price-prediction.git

Navigate to the cloned repository:

.. code-block:: bash

   cd src

To excute the script:

- create a conda environment using the env.yml file using the command "conda env create -f env.yml"
- activate the environment using "conda activate mle_training"
- Run the code using "python <scriptname.py>"

Run the script inside src/housepred in the following order:

.. code-block:: bash

   python ingest_data.py --output_dir <output_dir>
   python data_prep.py --input_dir <input_dir> --output_dir <output_dir>
   python train.py --input_dir <input_dir> --output_dir <output_dir>
   python score.py --input_dir <input_dir> --model_path <model_path>


Usage
=====

The main script fetches the dataset from an online source, performs data cleaning and preparation, and uses various machine learning models to predict house prices.

Here is an overview of the operations performed:

1. Fetching and loading data.
2. Splitting data into training and testing sets.
3. Exploratory Data Analysis (EDA).
4. Handling missing values and categorical features.
5. Implementing models like Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
6. Using Grid Search and Randomized Search for hyperparameter tuning.
7. Evaluating models and making predictions.

The results of these operations, such as model performance metrics, are printed out to the console. You can modify the script according to your requirements to get more insights or to tweak the model's configuration.


Contributing
============

Please read ``CONTRIBUTING.md`` for details on our code of conduct, and the process for submitting pull requests.


Versioning
==========

We use `SemVer <http://semver.org/>`_ for versioning.


Authors
=======

* **Pushvinder Kumar**

See also the list of `contributors <https://github.com/pushvinderrohtagi/mle_training.git>`_ who participated in this project.


License
=======

This project is licensed under the MIT License - see the ``LICENSE.md`` file for details.


Acknowledgments
===============

This project uses the `California Housing Prices` dataset from the StatLib repository, provided by Aurélien Géron.

For a detailed walkthrough of the project, please refer the Tiger Analytics documentation.
