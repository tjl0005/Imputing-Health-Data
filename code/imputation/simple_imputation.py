"""
Imputing EHR data through mean, k-NN and MICE.
Note: Currently lacking hyperparameter optimisation and further reading required. - this is in progress
"""
import xgboost
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from code.constants import CATEGORIES, IMPUTATION_OUTPUT
import pandas as pd
import numpy as np


def mean_impute(data):
    """
    Impute specified columns using mean values.
    :param data: The dataset containing missing values.
    :return: Data with any missing values imputed with variable mean.
    """
    # Missing means they are nan and will be imputed using the mean
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    # Imputing reading measurement columns only
    data[CATEGORIES] = mean_imputer.fit_transform(data[CATEGORIES])

    return data


def knn_impute(data, k):
    """
    Impute numeric columns using k-NN to determine missing values.
    :param data: Dataset containing missing values
    :param k: The number of neighbours to approximate imputed value from
    :return: The dataset with numeric columns imputed using nearest neighbours
    """
    numerical_data = data[CATEGORIES]

    # Imputing the measurement columns with k-NN
    knn_imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
    imputed_data = knn_imputer.fit_transform(numerical_data)

    # Updating readings in original data with imputed values
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)
    data[CATEGORIES] = imputed_df

    return data


def mice_impute(data, estimator_choice="Linear Regression"):
    """
    Impute numeric columns through multiple imputation by chained equations. The estimator needs to be
    specified.
    :param data: Dataset containg missing values
    :param estimator_choice: Model to use for estimating missing values
    :return: THe dataset with missing values imputed using MICE
    """
    numerical_data = data[CATEGORIES]

    if estimator_choice == "Linear Regression":
        estimator = LinearRegression()
    else:
        estimator = xgboost.XGBRegressor()

    iterative_imputer = IterativeImputer(estimator=estimator, random_state=502, max_iter=10)
    imputed_data = iterative_imputer.fit_transform(numerical_data)

    # Updating readings in original data with imputed values
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)
    data[CATEGORIES] = imputed_df

    return data


def impute_and_save(data, imputation_type, file, k=5, estimator_choice="Linear Regression"):
    """
    Impute and save specified datasets through specified imputation method.
    :param file:
    :param data: Data with missing values to be imputed
    :param imputation_type: String specifying imputation type. "mean", "knn" or "mice"
    :param k: Number of neighbours to use in k-NN
    :param estimator_choice:
    """
    # Avoid issues when using different methods on the same data
    missing_data = data.copy()

    # 0 represents 0 missing values per row
    if file == "measurements_0":
        print("Cannot impute file with no missing values")
        return
    else:
        print("Imputing {} through {}".format(file, imputation_type))

    if imputation_type == "mean":
        imputed_data = mean_impute(missing_data)
    elif imputation_type == "knn":
        imputed_data = knn_impute(missing_data, k)
    elif imputation_type == "mice":
        imputed_data = mice_impute(missing_data, estimator_choice)
    else:
        print("Imputation not recognised")
        return

    output_file = IMPUTATION_OUTPUT + "{}_{}_imputed.csv".format(file, imputation_type)
    imputed_data.to_csv(output_file, index=False)
