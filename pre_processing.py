""""
Preparing the EHR data to be used with the Imputation Models
"""
import pandas as pd

# Reading relevant data
admissions = pd.read_csv("./hosp/admissions.csv")
patients = pd.read_csv("./hosp/patients.csv")
icu_stays = pd.read_csv("./icu/icustays.csv")

# Combining relevant datasets before cleaning
merged_stays = pd.merge(icu_stays, admissions, on="subject_id", how="inner")
merged_stays = pd.merge(merged_stays, patients, on="subject_id", how="inner")


def limit_to_appropriate_patients(data):
    """
    Given health data limit it to appropriate patients for findings. Limiting to patients over 18 where it is their
    first known stay, which is between 12 hours and 10 days.

    :param data: Original EHR data
    :return: Reduced dataset for initial statistics and further cleaning
    """
    # Ensuring of age
    data = data[data["anchor_age"] > 18]

    # Keeping first ICU stays
    data = data.drop_duplicates(subset="subject_id")

    # Limiting to stays between 12hrs and 10 days
    data = data[(data['los'] >= 0.5) & (data['los'] <= 10)]

    return data


# Cleaning data with functions and saving
clean_data = limit_to_appropriate_patients(merged_stays)

clean_data.to_csv("./processed/icu_stays.csv", index=False, header=True)
