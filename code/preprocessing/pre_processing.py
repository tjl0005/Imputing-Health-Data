""""
Preparing the EHR data to be used with the Imputation Models
"""
import os
import pandas as pd
from code.constants import ICU_STAYS_FILE, HOSP_DIR


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
    data = data.sort_values(by=['subject_id', 'admittime']).drop_duplicates(subset="subject_id", keep="first")

    # Limiting to stays between 12hrs and 10 days
    data = data[(data['los'] >= 0.5) & (data['los'] <= 10)]

    # Adding prediction value column, limiting to survival only
    data['outcome'] = data['discharge_location'].where(data['discharge_location'] == 'DIED', "SURVIVED")

    # Data should always be the same so safe to drop irrelevant columns here
    data = data.drop(["intime", "outtime", "dischtime", "discharge_location", "deathtime", "admit_provider_id", "edregtime",
                      "edouttime", "hospital_expire_flag", "anchor_year", "anchor_year_group", "dod", "language",
                      "marital_status", "race", "insurance", "discharge_location", "subject_idhosp", "last_careunit",
                      "hadm_id"], axis=1)

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
            print("Merging dataset with {}".format(file))

            # When reading ICU data only want specific columns
            if "icu" in directory:
                cols = ["subject_id", "hadm_id", "stay_id", "itemid", "value"]

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


def process_into_full_icu_stays():
    """
    A wrapper function to merge the ICU stays and relevant hosptial admin data, limited to patients
    within desired subset. The data is also saved directly to the "data" directory under "icu_stays.csv"
    :return: The processed dataframe containing details of relevant patients ICU stays
    """
    # Merging ICU stays with hospital data for admin details to limit scope
    icu_stays = pd.read_csv(ICU_STAYS_FILE)
    icu_stays = merge_datasets(icu_stays, HOSP_DIR)

    # Reducing number of patients before merging with ICU data
    icu_stays = limit_to_appropriate_patients(icu_stays)
    icu_stays.to_csv("../data/icu_stays.csv", index=False, header=True)

    return icu_stays