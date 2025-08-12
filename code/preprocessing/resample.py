"""
Re-sample the mortality outcomes using under and over sampling to improve class balance. Used to test the difference
between downsampled and original data.
"""
import os
from collections import Counter
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from code.constants import RESAMPLED_DIR, GROUND_TRUTH_FILE, RAW_MISSING_DATA_DIR

ground_truth_data = pd.read_csv(os.path.join(RAW_MISSING_DATA_DIR, GROUND_TRUTH_FILE))


def downsample(features, targets):
    """
    Given the features and target variable reduce the dataset so the target classes are balanced.
    :param features: Dataframe containing the features which need to be downsampled.
    :param targets: Dataframe containing the target variable, which needs downsampling for.
    :return: Downsampled dataframe with equal class sizes
    """
    # Using random downsampling to balance mortality outcomes
    rus = RandomUnderSampler(sampling_strategy="majority")
    features_downsampled, targets_downsampled = rus.fit_resample(features, targets)

    return features_downsampled, targets_downsampled


def oversample(features, targets):
    """
    Given the features and target variable increase the dataset so the target classes are balanced.
    :param features: Dataframe containing the features which need to be downsampled.
    :param targets: Dataframe containing the target variable, which needs downsampling for.
    :return: Oversampled dataframe with equal class sizes
    """
    # Using random over sampling
    ros = RandomOverSampler(sampling_strategy="minority")
    features_downsampled, targets_downsampled = ros.fit_resample(features, targets)

    return features_downsampled, targets_downsampled


def resample_data(data, reference, print_out=False, target="outcome", sample_type="downsample"):
    """
    Given a dataset and a target variable, sample and save the sampled dataset and target variable.
    :param data: Data containing the features and target variable
    :param reference: Reference to save the file with for identification i.e. missing_2
    :param print_out: Boolean specifying whether to print the changes of the downsample.
    :param target: The name of the variable to targeted
    :param sample_type: "downsample" or "oversample
    :return: Both the resampled dataset and the save directory.
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
        return None

    if print_out:
        print("Resampling: {}".format(Counter(y_sampled)))

    sampled_data = pd.DataFrame(x_sampled, columns=X.columns)
    sampled_data["outcome"] = y_sampled

    sampled_data = sampled_data.sample(frac=1, random_state=507).reset_index(drop=True)

    resample_save_dir = "{}/{}_{}.csv".format(RESAMPLED_DIR, reference, sample_type)

    return sampled_data, resample_save_dir

# Example to downsample ground truth data
# resampled_data, save_dir = resample_data(ground_truth_data, "measurements_0")
# resampled_data.to_csv(save_dir)