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
    :param view_statistics: Boolean to decide whether to print initial statistics of he processed data
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

    # Removing incorrect flags from the data and plotting boxplots
    worst_readings = remove_missing_flags_and_impossible_data(worst_readings, enforce_limits=False)
    check_distributions(worst_readings, data_ref="Original Raw")

    # Removing impossible values from the data, saving it and plotting boxplots
    processed_stays = remove_missing_flags_and_impossible_data(worst_readings, enforce_limits=True)
    processed_stays.to_csv('../data/readings/processed_worst_24_hour_readings.csv', index=False)
    check_distributions(processed_stays, data_ref="Limited Raw")


def prepare_data_for_imputation(check_missing=False, missing_limits=None):
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


# prepare_data(view_statistics=True)
# perform_feature_selection()
prepare_data_for_imputation()
create_artificially_missing_datasets(complete_data_dir="../data/missing/raw/measurements_0.csv", artificial_dir="../data/missing/artificial")
# Now can test the different imputations through the valuation folder.