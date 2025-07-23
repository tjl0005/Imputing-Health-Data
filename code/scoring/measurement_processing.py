"""
Used to process the icustays and chartevents to get the features required for prediction and imputation.
"""
import os
import pandas as pd
from code.constants import CHUNK_SIZE, APACHE_FEATURES_FILE, ICU_STAYS_OUTPUT, ID_COLS, FEATURE_COLUMNS, \
    READINGS_OUTPUT, CHART_EVENTS_INPUT, MEASUREMENTS, GCS_MOTOR, GCS_VERBAL, GCS_EYE


def map_gcs_scores(aspect, values):
    if aspect == "Gcsmotor":
        return values.map(lambda x: GCS_MOTOR.get(x))
    elif aspect == "Gcsverbal":
        return values.map(lambda x: GCS_VERBAL.get(x))
    else:
        return values.map(lambda x: GCS_EYE.get(x))


def find_interval_readings(data, worst_readings=True, interval=1, time=24):
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
    data["time_diff"] = (data["time_diff"].dt.total_seconds() / 3600).round()

    # Limiting to rows within the 24 hours of the admittance time
    data_within_time = data[data["time_diff"] <= time]

    readings = []

    # Getting the readings for each measurement in this chunk
    for measurement in MEASUREMENTS:
        # Limiting to data within specified time
        measurement_data = data_within_time[data_within_time["measurement"] == measurement]

        # Needs to be converted into numerical
        if "Gcs" in measurement:
            gcs_value = measurement_data["value"]

            gcs_aspect_score = map_gcs_scores(measurement, gcs_value)
            measurement_data.loc[:, "value"] = gcs_aspect_score
            readings.append(measurement_data)

        # Other numerical measurements
        else:
            measurement_data = measurement_data.copy()
            measurement_data.loc[:, "value"] = pd.to_numeric(measurement_data["value"], errors="coerce")

            if worst_readings:
                # Group by subject_id and find the max value within each group
                max_values_per_subject = measurement_data.groupby("subject_id")["value"].max().reset_index()

                # Merge the max values with the original data to get the corresponding rows
                worst_rows = measurement_data.merge(max_values_per_subject, on=["subject_id", "value"], how="inner")

                readings.append(worst_rows)
            else:
                measurement_data.loc[:, "hour"] = measurement_data["charttime"].dt.floor('h')

                hourly_avg = measurement_data.groupby(["hour", "subject_id", "measurement"])["value"].mean().reset_index()

                measurement_data = measurement_data.merge(hourly_avg, on=["hour", "subject_id", "measurement"],
                                                          suffixes=("", "_avg"))
                readings.append(measurement_data)

    # Combine all worst readings to be appended
    worst_measurement_readings = pd.concat(readings)

    return worst_measurement_readings


def pivot_and_merge(readings_data, time_series=False):
    if time_series:
        final_readings_data = readings_data.pivot_table(index=["subject_id", "anchor_age", "time_diff"],
                                    columns="measurement",
                                    values="value",
                                    aggfunc="first")

        # Step 2: Reset index and handle missing values (replace NaN with 'n/a')
        final_readings_data = final_readings_data.reset_index()

    else:
        pivoted_readings_data = readings_data.pivot_table(index="subject_id", columns="measurement", values="value", aggfunc="max").reset_index()

        final_readings_data = pd.merge(readings_data[['subject_id', 'admittime', 'anchor_age', 'outcome', 'charttime']], pivoted_readings_data, on="subject_id", how="left")

        final_readings_data = final_readings_data.drop_duplicates(subset=["subject_id"], keep="first")

    return final_readings_data


def process_in_chunks(interval=24):
    """
    Given an input directory the file will be read and processed in chunks. The size is determined by the chunk_size
    parameter at the beginning of this file. Each chunk is appended to the specified output directory.
    :param interval: Number representing the timeframe to keep readings in
    """
    chunk_num = 0
    first_chunk = True

    worst_readings_dir = "{}/worst_{}_hour_readings.csv".format(READINGS_OUTPUT, interval)
    hourly_readings_dir = "../data/readings/24_hourly_readings.csv"

    features = pd.read_csv(APACHE_FEATURES_FILE)
    icu_stays = pd.read_csv(ICU_STAYS_OUTPUT)

    # Appending file so need to ensure that a new file is created
    if os.path.exists(worst_readings_dir):
        os.remove(worst_readings_dir)
        print("Removed old worst readings file")
    if os.path.exists(hourly_readings_dir):
        os.remove("../data/readings/24_hourly_readings.csv")
        print("removed old hourly readings file")

    # Need to process chart events in chunks due to memory usage
    # NOTE: Remember to remove nrows limit
    for chunk in pd.read_csv(CHART_EVENTS_INPUT, usecols=ID_COLS + FEATURE_COLUMNS, chunksize=CHUNK_SIZE):
        chunk_num += 1
        print("Chunk {} with {} lines read in total".format(chunk_num, (chunk_num * CHUNK_SIZE)))

        # Limiting chart events to desired measurements
        relevant_chart_events = chunk[chunk["itemid"].isin(features["itemid"])]
        relevant_chart_events = relevant_chart_events.merge(features[["itemid", "measurement"]], on="itemid",
                                                            how="left")
        relevant_chart_events = relevant_chart_events.drop(columns=["itemid"])

        # Combine with ICU stays and drop excess data
        combined_chunk = pd.merge(icu_stays, relevant_chart_events, on=["subject_id", "stay_id"]).drop(["stay_id"],
                                                                                                       axis=1)

        # Get first readings and the worst readings for the current chunk
        processed_worst_readings = find_interval_readings(combined_chunk, interval=interval).drop_duplicates(subset=["subject_id", "measurement"], keep="first")
        processed_hourly_readings = find_interval_readings(combined_chunk, interval=interval, worst_readings=False).drop_duplicates(subset=["subject_id", "measurement", "hour"])

        # Appending chunk data to output files
        processed_worst_readings.to_csv(worst_readings_dir, mode="a", header=first_chunk, index=False)
        processed_hourly_readings.to_csv(hourly_readings_dir, mode="a", header=first_chunk, index=False)

        first_chunk = False

    # Combine all worst readings after the loop - reduced so not needed to be read in chunks
    worst_measurement_readings = pd.read_csv(worst_readings_dir)
    hourly_measurement_readings = pd.read_csv(hourly_readings_dir)

    final_worst_measurements = pivot_and_merge(worst_measurement_readings)
    final_hourly_measurement_readings = pivot_and_merge(hourly_measurement_readings, time_series=True)

    final_worst_measurements.to_csv(worst_readings_dir, index=False)
    final_hourly_measurement_readings.to_csv(hourly_readings_dir, index=False)

    return worst_readings_dir