"""
Used to process the icustays and chartevents to get the features required for prediction and imputation.
"""
from code.constants import CHUNK_SIZE, APACHE_FEATURES_FILE, ICU_STAYS_OUTPUT, ID_COLS, FEATURE_COLUMNS, \
    READINGS_OUTPUT, CHART_EVENTS_INPUT, CATEGORIES
import pandas as pd
import os

def first_readings(data):
    """
    Returns all the patients with their first readings of the defined features
    :param data:
    :return:
    """
    # Sort subject readings by time and only keep the first of each measurement
    data = data.sort_values(by=["subject_id", "charttime"]).drop_duplicates(subset=["subject_id", "measurement"], keep="first")
    return data


def find_worst_readings(data, interval=24):
    """
    Given a dataset with the specified measurements this will return the worst readings for each of the measurements
    for each subject in the provided data chunk. The worst readings will be selected from the specified time interval.
    :param data: Dataset containing the specified measurements
    :param interval: Number representing the hours passed since the patients admission
    :return: Data containing the worst readings for each measurement in the provided chunk
    """
    # Converting to datetime types
    data["admittime"] = pd.to_datetime(data["admittime"])
    data["charttime"] = pd.to_datetime(data["charttime"])

    # Finding time passed between reading time and admission time and converting to hours
    data["time_diff"] = data["charttime"] - data["admittime"]
    data["time_diff"] = data["time_diff"].dt.total_seconds() / 3600

    # Limiting to rows within the 24 hours of the admittance time
    data_within_interval = data[data["time_diff"] <= interval]

    worst_readings = []

    # Getting the readings for each measurement in this chunk
    for measurement in CATEGORIES:
        # Limiting to data within specified time interval
        measurement_data = data_within_interval[data_within_interval["measurement"] == measurement]
        measurement_data.loc[:, "value"] = pd.to_numeric(measurement_data["value"], errors="coerce")

        # Group by subject_id and find the max value within each group
        max_values_per_subject = measurement_data.groupby("subject_id")["value"].max().reset_index()

        # Merge the max values with the original data to get the corresponding rows
        worst_rows = measurement_data.merge(max_values_per_subject, on=["subject_id", "value"], how="inner")

        # Append to list
        worst_readings.append(worst_rows)

    # Combine all worst readings to be appended
    worst_measurement_readings = pd.concat(worst_readings)

    return worst_measurement_readings


def process_in_chunks(interval=24):
    """
    Given an input directory the file will be read and processed in chunks. The size is determined by the chunk_size
    parameter at the beginning of this file. Each chunk is appended to the specified output directory.
    :param interval: Number representing the timeframe to keep readings in
    """
    chunk_num = 0
    first_chunk = True

    first_readings_dir = "{}/first_readings.csv".format(READINGS_OUTPUT)
    worst_readings_dir = "{}/worst_{}_hour_readings.csv".format(READINGS_OUTPUT, interval)

    features = pd.read_csv(APACHE_FEATURES_FILE)
    icu_stays = pd.read_csv(ICU_STAYS_OUTPUT)

    # Appending file so need to ensure that a new file is created
    if os.path.exists(first_readings_dir):
        os.remove(first_readings_dir)
        print("Removed old first readings file")
    if os.path.exists(worst_readings_dir):
        os.remove(worst_readings_dir)
        print("Removed old worst readings file")

    # Need to process chart events in chunks due to memory usage
    for chunk in pd.read_csv(CHART_EVENTS_INPUT, usecols=ID_COLS+FEATURE_COLUMNS, chunksize=CHUNK_SIZE):
        chunk_num += 1
        print("Chunk {} with {} lines read in total".format(chunk_num, (chunk_num * CHUNK_SIZE)))

        # Limiting chart events to desired measurements
        relevant_chart_events = chunk[chunk["itemid"].isin(features["itemid"])]
        relevant_chart_events = relevant_chart_events.merge(features[["itemid", "measurement"]], on="itemid", how="left")
        relevant_chart_events = relevant_chart_events.drop(columns=["itemid"])

        # Combine with ICU stays and drop excess data
        combined_chunk = pd.merge(icu_stays, relevant_chart_events, on=["subject_id", "stay_id"]).drop(["stay_id"], axis=1)

        # Get first readings and the worst readings for the current chunk
        processed_first_readings = first_readings(combined_chunk)
        processed_worst_readings = find_worst_readings(combined_chunk, interval).drop_duplicates(subset=["subject_id", "measurement"], keep="first")

        # Appending chunk data to output files
        processed_first_readings.to_csv(first_readings_dir, mode="a", header=first_chunk, index=False)
        processed_worst_readings.to_csv(worst_readings_dir, mode="a", header=first_chunk, index=False)

        first_chunk = False
