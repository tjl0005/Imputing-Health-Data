"""
Given the pre-processed data, create artificially missing data matching
"""
import pandas as pd
import numpy as np

np.random.seed(5072001)

# Columns to remove data from
MISSING_COLS = ["los", "anchor_age"]

icu_stays = pd.read_csv("./data/icu_stays.csv")

def remove_given_indices(data, selected_indices, missing_percentage, n_cols):
    """
    Given a dataset remove specified indexes.
    :param data: Original data with complete values
    :param selected_indices: A list in the form of (col, row_index) to be removed
    :param missing_percentage: A float i.e. 0.2 representing the percentage of data to be removed
    :param n_cols: Number of columns to remove values from
    :return: Modified dataset with artificially missing values
    """
    n_rows = data.shape[0]
    total_values = n_rows * n_cols
    n_missing = int(total_values * missing_percentage)

    # Desired number to remove may not exist so using minimum
    n_to_remove = min(len(selected_indices), n_missing)

    # Randomly selecting from subset of data and removing data
    missing_indices = np.random.choice(len(selected_indices), size=n_to_remove, replace=False)
    selected_pairs = [selected_indices[i] for i in missing_indices]

    for row, col in selected_pairs:
        data.at[row, col] = np.nan

    return n_to_remove, data


def missing_completely_at_random(data, missing_percentage, columns):
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
    n_removed, data_removed = remove_given_indices(data, column_pairs, missing_percentage, n_cols)
    print("{} randomly removed".format(n_removed))

    return data_removed


def missing_at_random_central(data, missing_percentage, columns, no_std):
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
    n_removed, incomplete_data = remove_given_indices(data, central_values, missing_percentage, n_cols)
    print("{} removed near center".format(n_removed))

    return incomplete_data


def missing_at_random_extremes(data, missing_percentage, columns, p):
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

    print("\nRemoving {:.2f}% at random followed by {:.2f}% at extremes".format(random_missing_rate * 100, not_random_missing_rate * 100))

    # Remove given percent of values at random
    data = missing_completely_at_random(data, p, columns)

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

        print("{} Bounds: {} and {}".format(col, lower_bound, upper_bound))

    # Removing data from lower and upper bounds
    lower_bound_n_removed, lower_bound_missing = remove_given_indices(data, lower_extremes, not_random_missing_rate, n_cols)
    upper_bound_n_removed, upper_bound_missing = remove_given_indices(data, upper_extremes, not_random_missing_rate, n_cols)

    print("{} removed at lower bound".format(lower_bound_n_removed))
    print("{} removed at upper bound".format(upper_bound_n_removed))

    return lower_bound_missing, upper_bound_missing


# Generating data with missing completely at random at 20%
mcar = missing_completely_at_random(icu_stays, 0.2, MISSING_COLS)
mcar.to_csv("./data/missing/icu_mcar.csv", index=False, header=True)

# Generating data with data missing near central values at 20%
mnar_central = missing_at_random_central(icu_stays, 0.2, MISSING_COLS, 1)
mnar_central.to_csv("./data/missing/icu_mnar_central.csv", index=False, header=True)

# Generating data with data missing near higher extremes at 20%
mnar_lower, mnar_upper = missing_at_random_extremes(icu_stays, 0.2, MISSING_COLS, 0.5)
mnar_lower.to_csv("./data/missing/icu_mnar_lower.csv", index=False, header=True)
mnar_upper.to_csv("./data/missing/icu_mnar_upper.csv", index=False, header=True)
