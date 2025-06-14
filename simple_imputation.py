"""
Imputing EHR data through mean, k-NN and MICE
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression


# Currently placeholder columns
MISSING_COLS = ["los", "anchor_age"]
FEATURES = MISSING_COLS

mcar = pd.read_csv("./data/missing/icu_mcar.csv")
mnar_central = pd.read_csv("./data/missing/icu_mnar_central.csv")
mnar_lower = pd.read_csv("./data/missing/icu_mnar_lower.csv")
mnar_upper = pd.read_csv("./data/missing/icu_mnar_upper.csv")


def mean_impute(data):
    """
    Impute specified columns using mean values
    :return: Data with specific columns imputed
    """
    # Missing means they are nan and will be imputed using the mean
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    # Imputed specified columns
    imputed_columns = mean_imputer.fit_transform(data[MISSING_COLS])

    # Converting imputed data into a dataframe so it can be merged with original data
    imputed_df = pd.DataFrame(imputed_columns, columns=MISSING_COLS, index=data.index)

    # Updating data with imputed values
    data[MISSING_COLS] = imputed_df[MISSING_COLS]

    return data


def knn_impute(data, k):
    """
    Impute specified columns using k-NN to determine missing values.
    :return:
    """
    # Imputing using numerical data only
    # Missing means they are nan and will be imputed using the mean
    knn_imputer = KNNImputer(n_neighbors=k)

    # Imputed specified columns
    imputed_columns = knn_imputer.fit_transform(data[FEATURES])

    # Converting imputed data into a dataframe so it can be merged with original data
    imputed_df = pd.DataFrame(imputed_columns, columns=data[FEATURES].columns, index=data[FEATURES].index)

    # Updating data with imputed values
    data[FEATURES] = imputed_df

    return data


def impute_and_save(data, missing_type, imputation_type, k=3):
    """
    Impute and save specified datasets through specified imputation method.
    :param data: Data with missing values to be imputed
    :param missing_type: String specifying the type of missing data, which will align with file name
    :param imputation_type: String specifying imputation type. "mean", "knn" or "mice"
    :param k: Number of neighbours to use in k-NN
    """
    if imputation_type == "mean":
        imputed_data = mean_impute(data)
    elif imputation_type == "knn":
        imputed_data = knn_impute(data, k)
    elif imputation_type == "mice":
        imputed_data = mice_impute(data)
    else:
        print("Imputation not recognised")
        return

    imputed_data.to_csv("./data/imputed/{}/{}_{}_imputed.csv".format(imputation_type, missing_type, imputation_type))


Imputing each type of missing data with each imputation method - default setups only
impute_and_save(mcar, "mcar", "mean")
impute_and_save(mcar, "mcar", "knn")

impute_and_save(mnar_central, "mcar", "mean")
impute_and_save(mnar_central, "mcar", "knn")

impute_and_save(mnar_lower, "mcar", "mean")
impute_and_save(mnar_lower, "mcar", "knn")

impute_and_save(mnar_upper, "mcar", "mean")
impute_and_save(mnar_upper, "mcar", "knn")