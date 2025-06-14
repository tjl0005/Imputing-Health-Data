""""
Preparing the EHR data to be used with the Imputation Models
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Directories for the data to be read from
HOSP_DIR = "./data/hosp/"
ICU_DIR = "./data/icu/"

# Lists identifying the features to be kept
all_features = ["subject_id", "hadm_id", "stay_id", "itemid", "value"]
procedure_features = ["location", "locationcategory", "patientweight"]

# Handled through one-hot encoding
categorical_features = ["admission_type", "admission_location"]

# Starting point data
icu_stays = pd.read_csv("./data/icu/icustays.csv")


def limit_to_appropriate_patients(data):
    """
    Given health data limit it to appropriate patients for findings. Limiting to patients over 18 where it is their
    first known stay, which is between 12 hours and 10 days.
    :param data: Original EHR data
    :return: Reduced dataset for initial statistics and further cleaning
    """
    # Ensuring of age
    data = data[data["anchor_age"] > 18]

    # Only keeping first ICU stays for each subject
    data = data.sort_values(by=['subject_id', 'intime']).drop_duplicates(subset="subject_id", keep="first")

    # Limiting to stays between 12hrs and 10 days
    data = data[(data['los'] >= 0.5) & (data['los'] <= 10)]

    # Data should always be the same so safe to drop
    data = data.drop(["outtime", "admittime", "dischtime", "deathtime", "admit_provider_id", "edregtime", "edouttime",
                      "hospital_expire_flag", "anchor_year", "anchor_year_group", "dod", "language", "marital_status",
                      "race"], axis=1)

    return data


def merge_datasets(data, directory, exclude=None):
    """
    The given dataset will be joined with all csv files in the given directory through the "subject_id" column.
    A list of exclusions can be passed to avoid joining with all data in the given directory.
    :param data: Original dataset that will be joined with all csv files.
    :param directory: The directory of the datasets to be joined.
    :param exclude: A list of file names to be excluded from the joined dataset.
    :return: The original data joined with all csv files in the provided directory.
    """
    if not exclude:
        exclude = []

    for file in os.listdir(directory):
        if file.endswith(".csv") and file not in exclude:
            print("Merging passed dataset with {}".format(file))

            # When reading ICU data only want specific columns
            if "icu" in directory:
                cols = all_features.copy()
                # Determine which features to read from data
                if file.startswith("ingredientevents") or file.startswith("inputevents"):
                    cols[4] = "amount"
                elif file.startswith("procedure"):
                    cols += procedure_features

                new_data = pd.read_csv(os.path.join(directory, file), usecols=cols, nrows=1000)
                data = pd.merge(data, new_data, on="stay_id", suffixes=("", "icu"))

            # Hospital data for admin details, using links from documentation
            else:
                if file.startswith("admissions"):
                    shared = "hadm_id"
                else:
                    shared = "subject_id"

                new_data = pd.read_csv(os.path.join(directory, file))
                data = pd.merge(data, new_data, on=shared, suffixes=("", "hosp"))

    return data


def one_hot_encode(data):
    """
    Using defined categorical columns one-hot encode them in the provided data.
    :param data: Complete data with categorical columns
    :return: Data with categorical columns one-hot encoded
    """
    # One-hot encoding specified categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[categorical_features])

    # Only encoded specific columns so need to re-emerge
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features), index=data.index)

    df_encoded = data.drop(columns=categorical_features)

    return pd.concat([encoded_df, df_encoded], axis=1)


# Merging ICU stays with hospital data for admin details to limit scope
icu_stays = merge_datasets(icu_stays, HOSP_DIR)

# Reducing number of patients before merging with ICU data
icu_stays = limit_to_appropriate_patients(icu_stays)

# icu_stays = merge_datasets(icu_stays, ICU_DIR, exclude=["icustays.csv", "d_items.csv"])

# Encoding categorical columns
# icu_stays = one_hot_encode(icu_stays)

icu_stays.to_csv("./data/icu_stays.csv", index=False, header=True)
