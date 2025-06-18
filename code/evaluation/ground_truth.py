"""
Using the data from feature selection with no missing values, which have been modified to be artificially missing data
that has been imputed, this file will evaluate the imputations using the ground truth through RMSE. This is not 
currently implemented in main.py.
NOTE: Will be updated once simple_imputation.py is set up for optimising imputation methods
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from code.constants import CATEGORIES, MISSING_DATA_DIR, IMPUTED_DATA_DIR, GROUND_TRUTH_FILE
from code.imputation.simple_imputation import impute_and_save

ground_truth_data =  pd.read_csv(os.path.join(MISSING_DATA_DIR, GROUND_TRUTH_FILE))


def impute_artificially_missing(imputation_type="knn"):
    """
    Reading the data inside the missing data directory (excluding ground truth) and impute it using the specified
    imputation type. This data will be saved to the imputed folder.
    :param imputation_type: String specifying which imputation method to apply to the data. Default is "knn".
    """
    # Read in artificially missing data in a loop
    for file in os.listdir(MISSING_DATA_DIR):
        if file.endswith(".csv") and file != GROUND_TRUTH_FILE:
            data = pd.read_csv(os.path.join(MISSING_DATA_DIR, file))
            file_name = os.path.splitext(file)[0]

            impute_and_save(data, imputation_type, file_name, IMPUTED_DATA_DIR)


def evaluate_ground_truth_imputed_file(file_name, output=False):
    """
    Given a file name containing imputed data originating from the ground truth evaluate the imputated values
    using RMSE for each of the measurements. The returned data will contain the RMSE for each category and the 
    average RMSE for the imputation.
    :param file_name: Name of the file for the imputed data, i.e. "measurements_0_knn.csv"
    :param output: A dataframe containing the RMSE for each of the measurement categories and the average
    :return: 
    """
    # Read the imputed data from the file
    imputed_data = pd.read_csv(os.path.join(IMPUTED_DATA_DIR, file_name))

    # Initialize a dictionary to store RMSE results
    rmse_results = {"file_name": file_name}

    # Individual RMSE scores per category
    rmse_values = []

    # Find RMSE for each category
    for category in CATEGORIES:
        category_rmse = np.sqrt(mean_squared_error(ground_truth_data[category], imputed_data[category]))
        rmse_results[category] = category_rmse
        rmse_values.append(category_rmse)

    # Adding average RMSE
    rmse_results["average_rmse"] = np.mean(rmse_values)

    if output:
        print(rmse_results)

    return rmse_results


def evaluate_all_ground_truth_imputations():
    """
    Go through all the ground truth data imputations (labelled "measurements_0_...") and evaluate them using RMSE.
    The evaluations will be saved as "ground_truth_comparisons.csv" in the evaluations folder and also returned as a
    Dataframe. It will contain a column representing the file and then remaining columns will specify the RMSE for
    each measurement
    """
    results = []

    # Going through all the imputed data files
    for file in os.listdir(IMPUTED_DATA_DIR):
        # Ensuring only getting ground truth data
        if file.endswith(".csv") and "measurements_0" in file:
            # Evaluating file with ground truth
            rmse_results = evaluate_ground_truth_imputed_file(file)

            results.append(rmse_results)

    # Saving results for all files as in one table
    results_df = pd.DataFrame(results)
    results_df.to_csv("../../data/evaluations/ground_truth_comparisons.csv", index=False)

    return results_df


# Example of single evaluation and full evaluation
# evaluate_ground_truth_imputed_file("measurements_0_mcar_knn_imputed.csv", True)
evaluate_all_ground_truth_imputations()
