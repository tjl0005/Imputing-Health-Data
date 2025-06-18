"""
Constants used throughout the code, specifying directories, files, etc. 
"""
# Directories for main.py
CHART_EVENTS_INPUT = "../data/icu/chartevents.csv"
ZERO_MISSING_INPUT = "../data/missing/measurements_0.csv"
MISSING_LIMIT_INPUT = "../data/missing/measurements_{}.csv"
READINGS_OUTPUT = "../data/readings"
ICU_STAYS_OUTPUT = "../data/icu_stays.csv"
IMPUTATION_OUTPUT = "../data/imputed/"

# Direct file directories
WORST_READINGS_FILE = "../data/readings/worst_24_hour_readings.csv"
ICU_STAYS_FILE = "../data/icu/icustays.csv"
APACHE_FEATURES_FILE = "../data/apache_measurements.csv"
GROUND_TRUTH_FILE = "measurements_0.csv"

# Directories used outside of main
HOSP_DIR = "../data/hosp/"
ICU_DIR = "../data/icu/"
MISSING_DATA_DIR = "../../data/missing"
IMPUTED_DATA_DIR = "../../data/imputed"

CHUNK_SIZE = 2000000

# Lists used throughout
CATEGORIES = ["mean arterial pressure", "heart rate", "respiratory rate", "PCO2 (Arterial)", "PO2 (Arterial)", "FiO2",
              "arterial pH", "sodium", "postassium", "creatinine", "hematocrit", "white blood cell", "anchor_age"]
NON_SCORED_COLUMNS = ["subject_id", "first_careunit", "los", "admittime", "admission_type", "admission_location", "gender",
                  "anchor_age", "outcome"]
ID_COLS = ["subject_id", "stay_id"]
FEATURE_COLUMNS = ["charttime", "itemid", "value"]
