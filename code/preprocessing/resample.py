"""
Re-sample the mortality outcomes using under and over sampling to improve class balance.
NOTE: Functional but not currently implemented anywhere currently.
"""
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

apache_scores = pd.read_csv("../../data/scores/final_apache_scores.csv")


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
    print("Undersampling: {}".format(Counter(y_resampled_down)))

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

def sample_and_save(data, target, sample_type):
    """
    Given a dataset and a target variable, sample and save the sampled dataset and target variable.
    :param data: Data containing the features and target variable
    :param target: The name of the variable to targeted
    :param sample_type: "downsample" or "oversample
    :return: Data with the specified sampling method applied
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

    sampled_data = pd.DataFrame(x_sampled, columns=X.columns)
    sampled_data["outcome"] = y_sampled

    return sampled_data

down_sampled_scores = sample_and_save(apache_scores, "outcome", "downsample")
over_sampled_scores = sample_and_save(apache_scores, "outcome", "oversample")

down_sampled_scores.to_csv("../../data/scores/apache_downsampled.csv")
over_sampled_scores.to_csv("../../data/scores/apache_oversampled.csv")