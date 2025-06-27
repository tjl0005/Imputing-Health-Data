"""
Re-sample the mortality outcomes using under and over sampling to improve class balance.
Required as negative mortality is so low in the ground truth data

NOTE: When expanding to using non-ground truth data need to account for missingness and why
"""
import os
from collections import Counter
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from code.constants import RESAMPLED_DIR, GROUND_TRUTH_FILE, RAW_MISSING_DATA_DIR

ground_truth_data = pd.read_csv(os.path.join(RAW_MISSING_DATA_DIR, GROUND_TRUTH_FILE))

def downsample(X, y):
    """
    Given the features and target variable reduce the dataset so the target classes are balanced.
    :param X: Features
    :param y: Target
    :return: Downsampled dataframe with equal class sizes
    """
    # Using random downsampling to balance mortality outcomes
    rus = RandomUnderSampler(sampling_strategy="majority")
    X_resampled_down, y_resampled_down = rus.fit_resample(X, y)
    print("Downsampling: {}".format(Counter(y_resampled_down)))

    return X_resampled_down, y_resampled_down


def oversample(X, y):
    """
    Given the features and target variable increase the dataset so the target classes are balanced.
    :param X: Features
    :param y: Target
    :return: Oversampled dataframe with equal class sizes
    """
    # Using random over sampling
    ros = RandomOverSampler(sampling_strategy="minority")
    X_resampled_up, y_resampled_up = ros.fit_resample(X, y)
    print("Oversampling: {}".format(Counter(y_resampled_up)))

    return X_resampled_up, y_resampled_up


def sample_and_save(data, reference, target="outcome", sample_type="downsample"):
    """
    Given a dataset and a target variable, sample and save the sampled dataset and target variable.
    :param data: Data containing the features and target variable
    :param reference: Reference to save the file with for identification i.e. missing_2
    :param target: The name of the variable to targeted
    :param sample_type: "downsample" or "oversample
    """
    # Splitting into features and prediction value
    X = data.drop(target, axis=1)
    y = data[target]

    if sample_type == "downsample":
        x_sampled, y_sampled = downsample(X, y)
    elif sample_type == "oversample":
        x_sampled, y_sampled = oversample(X, y)
    else:
        print("error")
        return

    sampled_data = pd.DataFrame(x_sampled, columns=X.columns)
    sampled_data["outcome"] = y_sampled

    sampled_data = sampled_data.sample(frac=1, random_state=507).reset_index(drop=True)

    save_dir = "{}/{}_{}.csv".format(RESAMPLED_DIR, reference, sample_type)
    sampled_data.to_csv(save_dir, index=False)


# Example to downsample ground truth data
sample_and_save(ground_truth_data, "measurements_0")