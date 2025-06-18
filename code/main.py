from constants import MISSING_LIMIT_INPUT, ZERO_MISSING_INPUT, MISSING_DATA_DIR, IMPUTED_DATA_DIR
from imputation.enforce_missingness import pivot_scores, final_cleaning, check_missing_scores
from preprocessing.pre_processing import process_into_full_icu_stays
from imputation.simple_imputation import impute_and_save
from scoring.feature_selection import process_in_chunks
from preprocessing.exploration import statistics
from scoring.apache import save_score_data
import pandas as pd
import os


def prepare_data(view_statistics=False):
    processed_stays = process_into_full_icu_stays()

    if view_statistics:
        statistics(processed_stays)


def perform_feature_selection():
    process_in_chunks()


def prepare_data_for_imputation(check_missing=False, missing_limits=None):
    if missing_limits is None:
        missing_limits = [0, 2, 3, 5]

    pivoted_measurements = pivot_scores()

    if check_missing:
        check_missing_scores(pivoted_measurements)

    for missing_limit in missing_limits:
        file_name = MISSING_LIMIT_INPUT.format(missing_limit)
        final_cleaning(pivoted_measurements, missing_limit=missing_limit).to_csv(file_name, index=False)


def impute_measurements(types=None, exclude=None):
    if types is None:
        types = ["mean", "knn"]
    if exclude is None:
        exclude = []

    for file in os.listdir(MISSING_DATA_DIR):
        if file.endswith(".csv") and file not in exclude:
            data = pd.read_csv(os.path.join(MISSING_DATA_DIR, file))
            file_name = os.path.splitext(file)[0]

            for imputation_type in types:
                impute_and_save(data, imputation_type, file_name)


def score_with_apache():
    # Scoring data with no missing values
    data_to_be_scored = pd.read_csv(ZERO_MISSING_INPUT)
    save_score_data(data_to_be_scored, "measurements_0_missing")

    for file in os.listdir(IMPUTED_DATA_DIR):
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(IMPUTED_DATA_DIR, file))
            file_name = os.path.splitext(file)[0]

            save_score_data(data, file_name)


# prepare_data()
# perform_feature_selection()
# prepare_data_for_imputation()
# impute_measurements()
# score_with_apache()