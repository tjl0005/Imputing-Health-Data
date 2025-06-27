"""
Constants used throughout the code, specifying directories, files, etc. 
"""
# Directories for main.py
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

# Direct file directories
WORST_READINGS_FILE = "../data/readings/worst_24_hour_readings.csv"
ICU_STAYS_FILE = "../data/icu/icustays.csv"
APACHE_FEATURES_FILE = "../data/apache_measurements.csv"
GROUND_TRUTH_FILE = "measurements_0.csv"
SAMPLED_GROUND_TRUTH_FILE = "measurements_0_downsample.csv"

# Directories used outside of main
HOSP_DIR = "../data/hosp/"
ICU_DIR = "../data/icu/"
MISSING_DATA_DIR = "../../data/missing"
ARTIFICIAL_MISSING_DATA_DIR = "../../data/missing/artificial/"
RAW_MISSING_DATA_DIR = "../../data/missing/raw/"
IMPUTED_DATA_DIR = "../../data/imputed"
GRID_SEARCH_OUTPUT = "../../data/grid_searches/"
PREDICTION_GS_RESULTS = "../../data/grid_searches/gs_results_records"
PREDICTION_GS_RECORD = "../../data/grid_searches/gs_records.csv"
RESAMPLED_DIR = "../../data/missing/resampled/"

CHUNK_SIZE = 2000000

# Lists used throughout
CATEGORIES = ["mean arterial pressure", "heart rate", "respiratory rate", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2",
              "arterial pH", "sodium", "postassium", "creatinine", "hematocrit", "white blood cell", "anchor_age"]
MEASUREMENTS = ["mean arterial pressure", "heart rate", "respiratory rate", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2",
              "arterial pH", "sodium", "postassium", "creatinine", "hematocrit", "white blood cell"]
NON_SCORED_COLUMNS = ["subject_id", "first_careunit", "los", "admittime", "admission_type", "admission_location",
                      "gender",
                      "anchor_age", "outcome"]
COMPLETE_FEATURES = ["FiO2", "admission_location", "admittime", "anchor_age",
                     "arterial pH", "creatinine", "first_careunit", "gender", "heart rate", "hematocrit",
                     "mean arterial pressure", "postassium", "respiratory rate", "sodium", "white blood cell"]
CATEGORICAL_FEATURES = ["admission_location", "admittime", "first_careunit", "gender"]
ID_COLS = ["subject_id", "stay_id"]
FEATURE_COLUMNS = ["charttime", "itemid", "value"]
GRID_SEARCH_RESULT_COLUMNS = ["param_gamma", "param_learning_rate", "param_max_depth", "param_n_estimators",
                              "mean_test_accuracy", "std_test_accuracy", "mean_test_precision", "std_test_precision",
                              "mean_test_recall", "std_test_recall", "mean_test_f1", "std_test_f1"]
MISSING_TYPES = ["mcar", "mnar_central", "mnar_lower", "mnar_upper"]
IMPUTATION_DIRECTORIES = ["mean", "knn", "mice"]

# Training parameters
XGBOOST_PARAMS = {
    "gamma": [0.01, 0.1],
    "learning_rate": [0.001, 0.01, 0.1, 1],
    "max_depth": [3, 6, 9],
    "n_estimators": [100, 200, 500]
}
PARAM_TYPES = {
    'param_gamma': float,
    'param_learning_rate': float,
    'param_max_depth': int,
    'param_n_estimators': int
}
KNN_PARAMS = {
    "n_neighbors": [3, 5, 10, 15, 20]
}
MICE_PARAMS = {
    "max_iters": [50, 100, 200]
}
MICE_FOREST_PARAMS = {
    "max_iters": [20, 50, 100]
}