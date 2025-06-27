"""
Imputing EHR data through mean, k-NN and MICE.
# Implement grid search
"""
import numpy as np
import pandas as pd
import xgboost
import miceforest as mf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from code.constants import MEASUREMENTS, IMPUTATION_OUTPUT

scaler = MinMaxScaler()

def mean_impute(data):
    """
    Impute specified columns using mean values. (Not an example of multiple imputation but used for comparison)
    :param data: The dataset containing missing values.
    :return: Data with any missing values imputed with variable mean.
    """
    # Missing means they are nan and will be imputed using the mean
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    # Imputing reading measurement columns only
    data[MEASUREMENTS] = mean_imputer.fit_transform(data[MEASUREMENTS])

    return data


def knn_impute(data, k):
    """
    Impute numeric columns using k-NN to determine missing values.
    :param data: Dataset containing missing values
    :param k: The number of neighbours to approximate imputed value from
    :return: The dataset with numeric columns imputed using nearest neighbours
    """
    numerical_data = data[MEASUREMENTS]
    scaled_data = scaler.fit_transform(numerical_data)

    # Imputing the measurement columns with k-NN
    knn_imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
    scaled_imputed_data = knn_imputer.fit_transform(scaled_data)

    imputed_data = scaler.inverse_transform(scaled_imputed_data)

    # Updating readings in original data with imputed values
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)
    data[MEASUREMENTS] = imputed_df

    return data


def mice_impute(data, estimator_choice="linear", max_iter=100):
    """
    Impute numeric columns through multiple imputation by chained equations. The estimator needs to be
    specified.
    :param data: Dataset containing missing values
    :param estimator_choice: Model to use for estimating missing values
    :return: THe dataset with missing values imputed using MICE
    """
    numerical_data = data[MEASUREMENTS]

    if estimator_choice == "linear":
        estimator = LinearRegression()
    elif estimator_choice == "rf":
        estimator = RandomForestRegressor(n_estimators=100)
    else:
        estimator = xgboost.XGBRegressor()

    iterative_imputer = IterativeImputer(estimator=estimator, random_state=502, max_iter=max_iter)
    imputed_data = iterative_imputer.fit_transform(numerical_data)

    # Updating readings in original data with imputed values
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)
    data[MEASUREMENTS] = imputed_df

    return data


def mice_forest_impute(data, num_datasets=2, max_iter=10, n_estimators=100):
    numerical_data = data[MEASUREMENTS]

    # Uses light-gbm and mean matching
    miceForest_imputer = mf.ImputationKernel(numerical_data, num_datasets=num_datasets, random_state=502)

    miceForest_imputer.mice(iterations=max_iter, n_estimators=n_estimators)

    imputed_data = miceForest_imputer.complete_data()

    if num_datasets > 1:
        print("Averaging {} imputed datasets for robust result".format(num_datasets))
        imputed_data = np.mean(imputed_data, axis=0)

    imputed_df = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)

    # Update the original DataFrame with the imputed values
    data[MEASUREMENTS] = imputed_df

    return data


def impute_and_save(data, imputation_type, file, k=5, estimator_choice="linear"):
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
    elif imputation_type == "mice_forest":
        imputed_data = mice_forest_impute(missing_data)
    else:
        print("Imputation not recognised")
        return None

    output_file = IMPUTATION_OUTPUT + "{}/{}_{}_imputed.csv".format(imputation_type, file, imputation_type)
    imputed_data.to_csv(output_file, index=False)

    return imputed_data