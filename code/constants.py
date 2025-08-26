"""
Constants used throughout the code, specifying directories, files, etc. It is not the most tidy but commented.
"""
from skopt.space import Categorical  # Required for parameter dictionaries

"""Directories used in main.py to get relevant data."""
CHART_EVENTS_INPUT = "../data/icu/chartevents.csv"
ZERO_MISSING_INPUT = "../data/missing/resampled/measurements_0_downsample.csv"
MISSING_LIMIT_INPUT = "../data/missing/raw/measurements_{}.csv"
READINGS_OUTPUT = "../data/readings"
ICU_STAYS_OUTPUT = "../data/icu_stays.csv"
IMPUTATION_OUTPUT = "../data/imputed/"
MAIN_MISSING_ARTIFICIAL_DIR = "../data/missing/artificial"
MAIN_MISSING_RAW_DIR = "../data/missing/raw"
MAIN_RESAMPLED_DIR = "../data/missing/resampled/"
MAIN_IMPUTED_DATA_DIR = "../data/imputed/"

""" Direct file directories to be used inside "code" directory."""
WORST_READINGS_FILE = "../data/readings/processed_worst_24_hour_readings.csv"
ICU_STAYS_FILE = "../data/icu/icustays.csv"
APACHE_FEATURES_FILE = "../data/feature_measurements.csv"
GROUND_TRUTH_FILE = "measurements_0.csv"
SAMPLED_GROUND_TRUTH_FILE = "measurements_0_downsample.csv"

"""Directories used inside "code" directory."""
HOSP_DIR = "../data/hosp/"
ICU_DIR = "../data/icu/"
MISSING_DATA_DIR = "../../data/missing"
ARTIFICIAL_MISSING_DATA_DIR = "../../data/missing/artificial/"
RAW_MISSING_DATA_DIR = "../../data/missing/raw/"
IMPUTED_DATA_DIR = "../../data/imputed"
GRID_SEARCH_OUTPUT = "../../data/grid_searches/"
PREDICTION_GS_RESULTS = "../../data/grid_searches/raw/gs_results_records"
PREDICTION_GS_RECORD = "../../data/grid_searches/raw/gs_records.csv"
RESAMPLED_DIR = "../../data/missing/resampled/"

""""Represents the max size in which data can be read."""
CHUNK_SIZE = 2000000

"""Lists used throughout the codebase."""
# The APACHE II features
MEASUREMENTS = ["mean arterial pressure", "heart rate", "respiratory rate", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2",
                "arterial pH", "sodium", "potassium", "creatinine", "hematocrit", "white blood cell", "HCO3 (serum)",
                "anchor_age"]
# Demographic information that is not used in imputation
NON_SCORED_COLUMNS = ["subject_id", "first_careunit", "los", "admittime", "admission_type", "admission_location",
                      "gender", "anchor_age", "outcome"]
# Complete features for prediction
COMPLETE_FEATURES = ["FiO2", "admission_location", "admittime", "anchor_age",
                     "arterial pH", "creatinine", "first_careunit", "gender", "heart rate", "hematocrit",
                     "mean arterial pressure", "potassium", "respiratory rate", "sodium", "white blood cell"]
# Features that are categorical and not currently handled
CATEGORICAL_FEATURES = ["admission_location", "admittime", "first_careunit", "gender"]
# The ID columns for the dataset, representing the patient and their stay
ID_COLS = ["subject_id", "stay_id"]
# Represent the columns required to extract the APACHE II measurements (represented by itemid)
FEATURE_COLUMNS = ["charttime", "itemid", "value"]
# When outputting the results of the downstream optimisation these columns are used
GRID_SEARCH_RESULT_COLUMNS = ["param_gamma", "param_learning_rate", "param_max_depth", "param_n_estimators",
                              "mean_test_accuracy", "std_test_accuracy", "mean_test_precision", "std_test_precision",
                              "mean_test_recall", "std_test_recall", "mean_test_f1", "std_test_f1", "std_test_roc_auc",
                              "mean_test_roc_auc"]
# Metric used to assess predictions
DOWNSTREAM_METRICS = ["precision", "recall", "accuracy", "f1", "roc_auc"]
# Represents the different levels of missingness, i.e. 2 features missing or 20% missing.
RAW_MISSING_LEVELS = [2, 5, 10]
ARTIFICIAL_MISSING_PERCENTAGES = [0.2, 0.5, 0.7]
# Artificial missing mechanism types
MISSING_TYPES = ["mcar", "mnar_central", "mnar_lower", "mnar_upper"]
# Imputer labels
NON_OPTIMISED_IMPUTERS = ["median", "mean", "wgain", "miwae"]
NON_DL_IMPUTERS = ["mean", "knn", "mice"]
DL_IMPUTERS = ["wgain", "miwae"]

"""Dictionary to identify which distributions should be visualised separately for different missing levels"""
KEY_DISTRIBUTION_FEATURES = {
    2: ["mean arterial pressure", "FiO2", "potassium"],
    5: ["mean arterial pressure", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2", "arterial ph"],
    10: ["mean arterial pressure", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2", "arterial pH"]
}

"""Training parameters."""""
XGBOOST_PARAMS = {
    "gamma": Categorical([0.01, 0.1]),
    "learning_rate": Categorical([0.001, 0.01, 0.1]),
    "max_depth": Categorical([3, 6, 9]),
    "n_estimators": Categorical([100, 200, 300])
}
PARAM_TYPES = {
    "param_gamma": float,
    "param_learning_rate": float,
    "param_max_depth": int,
    "param_n_estimators": int
}
KNN_PARAMS = {
    "n_neighbors": [3, 5, 10, 15, 20, 30]
}
MICE_PARAMS = {
    "max_iters": [20, 50, 100, 200],
}
MICE_FOREST_PARAMS = {
    "max_iters": [20, 50, 100]
}

"""Glasgow Coma Score variable mappings to scores - not used in final version."""
GCS_MOTOR = {
    "None": 1,
    "Involuntary": 2,
    "Abnormal Flexion": 3,
    "Abnormal extension": 3,
    "Flex-withdraws": 4,
    "Localizes Pain": 5,
    "Obeys Commands": 6
}
GCS_VERBAL = {
    "None": 1,
    "No Response-ETT": 1,
    "Incomprehensible sounds": 2,
    "Inappropriate Words": 3,
    "Confused": 4,
    "Oriented": 5
}
GCS_EYE = {
    "None": 1,
    "To Pain": 2,
    "To Speech": 3,
    "Spontaneously": 4
}