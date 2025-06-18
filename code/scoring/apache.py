"""
Given worst readings from the past 24 hours this file is used to generate the Apache-II scores for all subjects.
"""
import numpy as np
from code.constants import CATEGORIES


def decide_oxygenation_metric(fi_o2, pa_co2, pa_o2):
    """
    Given the relevant elements the function will return the required metric and the value to determine the apache score
    for oxygenation. THe fi_o2 value determines whether the metric will be the alveolar-arterial oxygen difference or
    the partial pressure of arterial oxygen.
    :param fi_o2: Fraction of Inspired Oxygen as a percentage value. i.e. 50% -> 50.
    :param pa_co2: Partial Pressure of Arterial Carbon Dioxide.
    :param pa_o2: Partial Pressure of Arterial Oxygen.
    :return: A tuple of 2 values with a string containing the required metric and a numerical containing the metric
    value
    """
    # Apache score dependent on the bound
    if fi_o2 >= 50:
        # Calculating A-aDO2 using formula, FiO2 is stored as percentage not fraction hence division
        pao2 = (fi_o2 / 100) * 713
        a_a_do2 = pao2 - (pa_co2 / 0.8) - pa_o2
        return "A-aDO2", a_a_do2
    else:
        # Wrong unit used in original. Converting from mmHg to kPa
        pa_o2 = pa_o2 * 0.133
        return "PaO2", pa_o2


def calculate_single_scores(category, reading, pa_co2=None, pa_o2=None):
    """
    Given an apache category and reading for that category return the apache II score
    :param category: APACHE II category
    :param reading: Worst reading within 24 hours of the given category
    :param pa_o2: Default is none, required for FiO2 score - refers to
    :param pa_co2: Default is none, required for FiO2 score - refers to
    :return: Score based on the reading and category
    """
    # Equivalent to a missing reading
    if reading > 0 and reading > 999999:
        return np.nan
    # Defining ranges and scores for the categories
    elif category == "temperature":
        # Each sub-list is made up of the bounds per score, from highest lowest
        ranges = ([41, 50, 4], [39, 40.9, 3], [38.5, 38.9, 1], [36, 38.9, 0], [34, 35.9, 1], [32, 33.9, 2],
                  [30, 31.9, 3], [19, 29.9, 4])
    elif category == "mean arterial pressure":
        ranges = ([160, 1000, 4], [130, 159, 3], [110, 129, 2], [70, 109, 0], [50, 69, 2], [0, 49, 4])
    elif category == "heart rate":
        ranges = ([180, 300, 4], [140, 179, 3], [110, 139, 2], [70, 109, 0], [55, 69, 2], [40, 45, 3], [0, 49, 4])
    elif category == "respiratory rate":
        ranges = ([50, 100, 4], [35, 49, 3], [25, 34, 1], [12, 24, 0], [10, 11, 1], [6, 9, 2], [0, 5, 4])
    elif category == "FiO2":
        # Oxygenation requires check to identify correct metric and ranges
        metric, reading = decide_oxygenation_metric(reading, pa_co2, pa_o2)
        if metric == "A-aDO2":
            ranges = ([500, 1000, 4], [350, 499, 3], [200, 349, 2], [70, 200, 0])
        else:
            ranges = ([70, 100, 0], [61, 69, 1], [55, 60, 3], [0, 55, 4])
    elif category == "arterial pH":
        ranges = ([7.7, 9.0, 4], [7.6, 7.69, 3], [7.5, 7.59, 1], [7.33, 7.49, 0], [7.25, 7.32, 2], [7.15, 7.24, 3],
                  [5, 7.15, 4])
    elif category == "sodium":
        ranges = ([180, 300, 4], [160, 179, 3], [155, 159, 2], [150, 154, 1], [130, 149, 0], [120, 129, 2],
                  [111, 119, 3], [50, 110, 4])
    elif category == "postassium":
        ranges = ([7, 9, 4], [6, 6.9, 3], [5.5, 5.9, 1], [3.5, 5.4, 0], [3, 3.4, 1], [2.5, 2.9, 2], [2.5, 2, 4])
    elif category == "creatinine":
        ranges = ([3.5, 30, 4], [2, 3.4, 3], [1.5, 1.9, 2], [0.6, 1.4, 0], [0, 0.6, 2])
    elif category == "hematocrit":
        ranges = ([60, 100, 4], [50, 59.9, 2], [46, 49.9, 1], [30, 45.9, 0], [20, 29.9, 2], [0, 20, 4])
    elif category == "white blood cell":
        ranges = ([40, 100, 4], [20, 39.9, 2], [15, 19.9, 1], [3, 14.9, 0], [1, 2.9, 2], [0, 1, 4])
    elif category == "anchor_age":
        ranges = ([18, 44, 0], [45, 54, 2], [55, 64, 3], [65, 74, 5], [75, 120, 6])
    else:
        return np.nan

    # Go through each point range
    for value_range in ranges:
        # Check if the reading is within the current range
        if value_range[0] <= reading <= value_range[1]:
            # If reading within this range return the relevant score
            return int(value_range[2])

    # No metric value so no score can be applied
    return np.nan


def apply_scores(row):
    """
    Generate the scores for the passed row containing the relevant measurements for APACHE II
    :param row: the dataframe row containing the measurements to be scored
    :return: the row with the columns updated to contain their relevant APACHE II scores
    """
    # Iterate over the categories and update the corresponding column with its score
    for category in CATEGORIES:
        reading = row[category]

        # For FiO2, need the additional pa_co2 and pa_o2 values
        if category == "FiO2":
            pa_co2 = row["PCO2 (Arterial)"]
            pa_o2 = row["PO2 (Arterial)"]
            score = calculate_single_scores(category, reading, pa_co2=pa_co2, pa_o2=pa_o2)
        else:
            score = calculate_single_scores(category, reading)

        # Update the row with the calculated score
        row[category] = score

    return row


def save_score_data(data, file_name):
    """
    Create and save a new version of the provided data with the APACHE II measurements converted
    into their relevant scores.
    :param data: Dataset containing the relevant APACHE II measurements
    :param file_name: The filename to save the score data under
    """
    # Scoring the data and removing irrelevant columns
    data_with_scores = data.apply(apply_scores, axis=1).drop(["PCO2 (Arterial)", "PO2 (Arterial)"], axis=1)
    data_with_scores.to_csv("../data/scores/{}.csv".format(file_name, index=False))

