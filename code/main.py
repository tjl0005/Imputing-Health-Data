"""
This file is used to prepare the data and will not actually begin imputation testing. It is used to preprocess and
enforce the different levels of missingness and different missing mechanisms.

Please see the README for further guidance on next steps. i.e. ground truth optimisations and evaluation.
"""
import os

import pandas as pd

from code.create_missing.randomise_missing import create_artificially_missing_datasets
from constants import MISSING_LIMIT_INPUT, MAIN_MISSING_ARTIFICIAL_DIR
from code.create_missing.enforce_missingness import final_cleaning, check_missing_scores
from imputation.ml import impute_and_save
from preprocessing.exploration import initial_statistics, check_distributions
from preprocessing.pre_processing import process_into_full_icu_stays, remove_missing_flags_and_impossible_data
from scoring.measurement_processing import process_in_chunks


def prepare_data(view_statistics=False):
    """
    Pre-process the original HOSP and ICU data to the relevant patients and features
    :param view_statistics: Boolean to decide whether to print initial statistics of the processed data
    :return:
    """
    processed_stays = process_into_full_icu_stays()

    if view_statistics:
        initial_statistics(processed_stays)


def perform_feature_selection():
    """
    Combines the processed ICU stays with their relevant chart events, specifically those used in APACHE. Produces two
    separate files for the worst readings within 24 hours and the first readings of their stay. The worst are used in
    APACHE II scoring
    """
    # Reads the chart events data in chunks and extracts both the worst readings within 24 hours and the first readings
    worst_readings = process_in_chunks()

    # Removing incorrect flags from the data and plotting box plots
    worst_readings = remove_missing_flags_and_impossible_data(worst_readings, enforce_limits=False)
    check_distributions(worst_readings, data_ref="Original Raw")

    # Removing impossible values from the data, saving it and plotting box plots
    processed_stays = remove_missing_flags_and_impossible_data(worst_readings, enforce_limits=True)
    processed_stays.to_csv("../data/readings/processed_worst_24_hour_readings.csv", index=False)
    check_distributions(processed_stays, data_ref="Limited Raw")


def prepare_data_for_imputation(check_missing=False, missing_limits=None):
    """
    Assumin the data has been pre-processed and feature selection is complete this will enforce different levels of
    missingness on the data. As a result 4 datasets will be produced containing different levels of values missing per
    patient as specified by missing_limits.
    :param check_missing: Boolean to decide whether to check missing data in the APACHE II feature selection.
    :param missing_limits: List containing the different limits to be applied, default is 0, 2, 5, 10 and 12. NOTE: 0 is
    required when doing ground truth imputation.
    """
    if missing_limits is None:
        missing_limits = [0, 2, 5, 10, 12]

    measurement_data = pd.read_csv("../data/readings/processed_worst_24_hour_readings.csv")

    if check_missing:
        check_missing_scores(measurement_data)

    # Limits complete data to different levels of missingness
    for missing_limit in missing_limits:
        file_name = MISSING_LIMIT_INPUT.format(missing_limit)
        final_cleaning(measurement_data, missing_limit=missing_limit).to_csv(file_name, index=False)

    ground_truth_data = pd.read_csv("../data/missing/raw/measurements_0.csv")
    check_distributions(ground_truth_data, data_ref="ground_truth")


def impute_measurements(types=None, exclude=None):
    """
    Given the imputer types to check this will impute the ground truth data with these imputers, using the default
    hyperparameters. Optimisation is available in ground_truth_optimisation.
    :param types: A list containing the imputer types to check. i.e. ["mean", "knn", "mice"]
    :param exclude: List to specify if any files should be excluded in this test.
    """
    if types is None:
        types = ["mean", "knn", "mice"]
    if exclude is None:
        exclude = []

    for file in os.listdir(MAIN_MISSING_ARTIFICIAL_DIR):
        if file.endswith(".csv") and file not in exclude:
            print(MAIN_MISSING_ARTIFICIAL_DIR + file)
            data = pd.read_csv(os.path.join(MAIN_MISSING_ARTIFICIAL_DIR, file))
            file_name = os.path.splitext(file)[0]

            for imputation_type in types:
                impute_and_save(data, imputation_type, file_name)


# Preprocesses the stays and links them to their patient records (no measurements or further features)
# prepare_data(view_statistics=True)

# Links patients to their measurement data - this is very long and memory intensive, default chunk size is 2000000, if
# to large it can be modified in the constants file.
# perform_feature_selection()

# This creates the different versions of the datasets which are actually being imputed. It will limit the records for
# each patient to be missing either 0, 2, 5, 10 or 12 values per record.
prepare_data_for_imputation()

# This uses the ground truth data i.e. no missing values and removes data under specific mechanisms described in the
# README. Produces 12 datasets, each with a different combination of missing level and missing mechanism.
create_artificially_missing_datasets(complete_data_dir="../data/missing/raw/measurements_0.csv",
                                     artificial_dir="../data/missing/artificial")