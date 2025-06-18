"""
Using the worst 24 hour measurement data this file is used to generate different versions of the dataset with different
levels of missingness ranging from 0 values per row to up to 5 values per row.
"""
import numpy as np
import pandas as pd
from code.constants import WORST_READINGS_FILE, CATEGORIES, NON_SCORED_COLUMNS


def pivot_readings():
    """
    Pivot measurements to columns while ensuring the retention of important non-measurement columns.
    :return: Pivoted dataframe.
    """
    readings_data = pd.read_csv(WORST_READINGS_FILE)

    # Pivoting measurements into columns
    pivoted_data = readings_data.pivot_table(index="subject_id", columns="measurement", values="value", aggfunc="first")

    # Need to remove repeated ids from original handling of measurements
    readings_data = readings_data[NON_SCORED_COLUMNS].drop_duplicates(subset="subject_id")

    # Retaining subject__id
    pivoted_data = pivoted_data.reset_index()

    # Merge pivoted data back with the original data
    final_data = pivoted_data.merge(readings_data[NON_SCORED_COLUMNS], on="subject_id", how="left")

    return final_data


def check_missing_scores(data):
    """
    Given the scored dataset print the number of rows with missing data for each score.
    :param data:
    """
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    print(missing_values)


def final_cleaning(reading_data, missing_limit=0):
    """
    Removes the temperature column as there is too much missing data to be meaningful. Also provides the option to
    remove rows where they are missing to many values. By default, all rows with missing data are removed.
    :param reading_data: Scored apache dataset
    :param missing_limit: A number representing the number of variables per row that can be missing
    :return: The cleaned dataset with missing data removed
    """
    initial_rows = reading_data.shape[0]

    # Dropping temperature as it is missing to many values
    reading_data = reading_data.drop(columns=["temperature", "glasgow coma score"])

    # Convert measurements to numeric and filter to be in range
    reading_data[CATEGORIES] = reading_data[CATEGORIES].apply(lambda col: pd.to_numeric(col, errors='coerce').where((col >= 0) & (col <= 9999), np.nan))

    # Limiting to rows with more than the specified number of values missing
    missing_per_row = reading_data.isna().sum(axis=1)
    reading_data = reading_data[missing_per_row <= missing_limit]

    # Checking how many rows were dropped
    final_rows = reading_data.shape[0]
    dropped_rows = initial_rows - final_rows
    print("Dropped {} rows due to rows missing at least {} values".format(dropped_rows, missing_limit))

    return reading_data
