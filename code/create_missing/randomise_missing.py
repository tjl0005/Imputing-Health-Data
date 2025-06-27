"""
Given a complete dataset, create artificially missing data either MCAR or MNAR. Descriptions of
how are in the docstrings for each function. The output is intended to be used in evaluating 
imputation through ground truth.
NOTE: Not implemented in main.py currently.
"""
import numpy as np
import pandas as pd
from code.constants import MEASUREMENTS

np.random.seed(50701)
# Measurements with no data missing - acts as a ground truth
complete_measurements = pd.read_csv("../../data/missing/resampled/measurements_0_downsample.csv")


def remove_given_indices(data, selected_indices, missing_percentage, n_cols, n_missing_per_row):
    """
    Given a dataset, remove specified indices, ensuring no more than `n_missing_per_row` missing values per row.
    :param data: Original data with complete values
    :param selected_indices: A list in the form of (row_index, col) to be removed
    :param missing_percentage: A float representing the percentage of data to be removed (e.g., 0.2 for 20%)
    :param n_cols: Number of columns to remove values from
    :param n_missing_per_row: Maximum number of missing values allowed per row
    :return: Modified dataset with artificially missing values
    """
    n_rows = data.shape[0]
    total_values = n_rows * n_cols
    # How many values can be missing given percentage
    n_missing = int(total_values * missing_percentage)

    # Tracking how many values are missing per row, not currently used but can be used to limit rows to having x values
    # row_missing_count = {row: data.iloc[row].isna().sum() for row in range(n_rows)}

    missing_pairs = []
    missing_data = data.copy()

    # Randomising order of deletion
    np.random.shuffle(selected_indices)

    # Removing specified number of rows and tracking how many are being deleted
    for row, col in selected_indices:
        if len(missing_pairs) < n_missing:
            missing_pairs.append((row, col))
            # row_missing_count[row] += 1

    # Remove values from dataset
    for row, col in missing_pairs:
        missing_data.at[row, col] = np.nan

    return len(missing_pairs), missing_data


def missing_completely_at_random(data, missing_percentage, columns, n_missing_per_row=5):
    """
    Given a dataset randomly remove values from specified columns.
    :param data: Dataframe containing specified columns
    :param missing_percentage: A float i.e. 0.2 representing the percentage of data to be removed
    :param columns: Columns to remove values from
    :return: Updated dataset with given percentage of random values missing
    """
    n_cols = len(columns)

    # Identify all values in the specified columns in the dataset
    column_pairs = [(row, col) for row in data.index for col in columns]

    # Remove any data from the identified values
    n_removed, data_removed = remove_given_indices(data, column_pairs, missing_percentage, n_cols, n_missing_per_row)
    print("{} randomly removed".format(n_removed))

    return data_removed


def missing_at_random_central(data, missing_percentage, columns, no_std, n_missing_per_row=5):
    """
    Given a dataset remove the given percentage of values within the desired standard deviations of the mean.
    :param data: Dataframe containing specified columns
    :param missing_percentage: A float i.e. 0.2 representing the percentage of data to be removed
    :param columns: Columns to remove values from
    :param no_std: The number of standard deviations in which data will be removed
    :return: Updated dataset with given percentage of values missing near the centre
    """
    n_cols = len(columns)

    # Values that are within specified standard deviations and can be randomly removed
    central_values = []

    # Getting values within range for each column
    for col in columns:
        # Mean and Standard Deviation for calculating bounds
        col_mean = data[col].mean()
        col_std = data[col].std()

        # Bounds from the number of standard deviations from the mean
        lower_bound = col_mean - no_std * col_std
        upper_bound = col_mean + no_std * col_std

        # Checking each value is within range and if so store entry to be randomly selected
        for row in data.index:
            if lower_bound <= data.at[row, col] <= upper_bound:
                central_values.append((row, col))

    # Remove desired percentage from subset of central datapoints
    n_removed, incomplete_data = remove_given_indices(data, central_values, missing_percentage, n_cols, n_missing_per_row)
    print("{} removed near center".format(n_removed))

    return incomplete_data


def missing_at_random_extremes(data, missing_percentage, columns, p, n_missing_per_row=5):
    """
    Given a dataset remove values randomly and within the extremes of the column values. The split is decided by "p", a
    higher value will mean completely random whereas a lower one means more will be removed near the mean.
    :param data: Dataframe containing specified columns
    :param missing_percentage: A float i.e. 0.2 representing the percentage of data to be removed
    :param columns: Columns to remove values from
    :param p: Ratio to split the amount of data removed randomly and the amount removed at either extreme.
    :return: Two updated datasets with given percentage of values missing randomly and at either extreme
    """
    n_cols = len(columns)

    # Splitting rate of randomness between completely at random and then from extremes
    random_missing_rate = missing_percentage * p
    not_random_missing_rate = missing_percentage - random_missing_rate

    print("\nRemoving {:.2f}% at random followed by {:.2f}% at extremes".format(random_missing_rate * 100,
                                                                                not_random_missing_rate * 100))

    # Values that are within specified standard deviations and can be randomly removed
    lower_extremes, upper_extremes = [], []

    # Getting values within range for each column
    for col in columns:
        # Calculating extremes using interquartile range
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Checking each value is within range and if so save reference
        for row in data.index:
            val = data.at[row, col]
            if lower_bound <= val:
                lower_extremes.append((row, col))
            elif val <= upper_bound:
                upper_extremes.append((row, col))

        # print("{} Bounds: {:.2f} and {:.2f}".format(col, lower_bound, upper_bound))

    # Removing data from lower and upper bounds
    lower_bound_n_removed, lower_bound_missing = remove_given_indices(data, lower_extremes, not_random_missing_rate,
                                                                      n_cols, n_missing_per_row)
    upper_bound_n_removed, upper_bound_missing = remove_given_indices(data, upper_extremes, not_random_missing_rate,
                                                                      n_cols, n_missing_per_row)

    print("{} removed at lower bound".format(lower_bound_n_removed))
    print("{} removed at upper bound".format(upper_bound_n_removed))

    # Remove given percent of values at random
    lower_bound_missing = missing_completely_at_random(lower_bound_missing, p, columns, n_missing_per_row)
    upper_bound_missing = missing_completely_at_random(upper_bound_missing, p, columns, n_missing_per_row)

    return lower_bound_missing, upper_bound_missing


def generate_missing_data(missing_percentage):
    mcar = missing_completely_at_random(complete_measurements, missing_percentage, MEASUREMENTS)
    mcar.to_csv("../../data/missing/artificial/measurements_{}_mcar.csv".format(missing_percentage), index=False, header=True)

    mnar_central = missing_at_random_central(complete_measurements, missing_percentage, MEASUREMENTS, 1)
    mnar_central.to_csv("../../data/missing/artificial/measurements_{}_mnar_central.csv".format(missing_percentage), index=False, header=True)

    mnar_lower, mnar_upper = missing_at_random_extremes(complete_measurements, missing_percentage, MEASUREMENTS, 0.5)
    mnar_lower.to_csv("../../data/missing/artificial/measurements_{}_mnar_lower.csv".format(missing_percentage), index=False, header=True)
    mnar_upper.to_csv("../../data/missing/artificial/measurements_{}_mnar_upper.csv".format(missing_percentage), index=False, header=True)


# Producing datasets with different percentages of missingness
generate_missing_data(0.2)
generate_missing_data(0.5)
generate_missing_data(0.7)