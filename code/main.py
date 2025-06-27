import os

import pandas as pd

from code.constants import IMPUTATION_DIRECTORIES, SAMPLED_GROUND_TRUTH_FILE, MAIN_RESAMPLED_DIR, \
    MAIN_IMPUTED_DATA_DIR
from constants import MISSING_LIMIT_INPUT, MAIN_MISSING_ARTIFICIAL_DIR
from code.create_missing.enforce_missingness import pivot_readings, final_cleaning, check_missing_scores
from imputation.ml import impute_and_save
from preprocessing.exploration import statistics
from preprocessing.pre_processing import process_into_full_icu_stays
from scoring.apache import save_score_data
from scoring.feature_selection import process_in_chunks


def prepare_data(view_statistics=False):
    """
    Pre-process the original HOSP and ICU data to the relevant patients and features
    :param view_statistics: Boolean to decide whether to print initial statistics of he processed data
    :return:
    """
    processed_stays = process_into_full_icu_stays()

    if view_statistics:
        statistics(processed_stays)


def perform_feature_selection():
    """
    Combines the processed ICU stays with their relevant chart events, specifically those used in APACHE. Produces two
    separate files for the worst readings within 24 hours and the first readings of their stay. The worst are used in
    APACHE II scoring
    """
    # Reads the chart events data in chunks and extracts both the worst readings within 24 hours and the first readings
    process_in_chunks()


def prepare_data_for_imputation(check_missing=False, missing_limits=None):
    if missing_limits is None:
        missing_limits = [0, 2, 3, 5]

    pivoted_measurements = pivot_readings()

    if check_missing:
        check_missing_scores(pivoted_measurements)

    for missing_limit in missing_limits:
        file_name = MISSING_LIMIT_INPUT.format(missing_limit)
        final_cleaning(pivoted_measurements, missing_limit=missing_limit).to_csv(file_name, index=False)


def impute_measurements(types=None, exclude=None):
    if types is None:
        types = ["mean", "knn", "mice"]
    if exclude is None:
        exclude = []

    for file in os.listdir(MAIN_MISSING_ARTIFICIAL_DIR):
        if file.endswith(".csv") and file not in exclude:
            data = pd.read_csv(os.path.join(MAIN_MISSING_ARTIFICIAL_DIR, file))
            file_name = os.path.splitext(file)[0]

            for imputation_type in types:
                impute_and_save(data, imputation_type, file_name)


def score_with_apache():
    # Scoring data with no missing values
    data_to_be_scored = pd.read_csv(os.path.join(MAIN_RESAMPLED_DIR, SAMPLED_GROUND_TRUTH_FILE))
    save_score_data(data_to_be_scored, "measurements_0_downsample")

    for imputation_type in IMPUTATION_DIRECTORIES:
        imputed_dir = "{}/{}".format(MAIN_IMPUTED_DATA_DIR, imputation_type)
        for file in os.listdir(imputed_dir):
            if file.endswith(".csv"):
                data = pd.read_csv(os.path.join(imputed_dir, file))
                file_name = os.path.splitext(file)[0]

                save_score_data(data, file_name)


# imputation grid search with prediction model
# def evaluate_without_ground_truth():


# prepare_data(view_statistics=True)
# perform_feature_selection()
# prepare_data_for_imputation()
impute_measurements()
# score_with_apache()