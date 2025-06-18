"""
Implementation of XGBoost using the apache scores. Intended to be way of testing different imputation methods and
sampling techniques.
NOTE: Not yet fully implemented and will be done after finalising imputation of ground truth tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Need to see how literature optimises and how to properly implement a grid search

apache_scores = pd.read_csv("../../data/scores/complete_scores.csv")
apache_scores_limit_2 = pd.read_csv("../../data/scores/scores_limit_2.csv")
apache_scores_limit_3 = pd.read_csv("../../data/scores/scores_limit_3.csv")
apache_scores_limit_5 = pd.read_csv("../../data/scores/scores_limit_5.csv")

non_features = ["subject_id", "los", "outcome", "total_score"]
features = apache_scores.columns.difference(non_features).tolist()
categorical_columns = ["admission_location", "admission_type", "admittime", "first_careunit", "gender"]

le = LabelEncoder()


def data_setup(score_data):
    """
    Split the data into training and test data for both the features and predicted values. Using stratified sampling to
    get even class distributions.
    :param score_data: The data to be split
    :return: X_train, X_test, y_train, y_test
    """
    # Splitting into features and target variables
    X = score_data[features]
    y = score_data["outcome"]

    # Splitting into training and test data, stratifying due to limited data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=507, stratify=y)

    y_train = le.fit_transform(y_train)

    for col in categorical_columns:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, X_test, y_train, y_test


def xgb_predictions(X_train, X_test, y_train, n_estimators):
    """"
    Uses defined training and test data to train an XGB model with the specified n_estimators and makes predictions
    :return the trained XGB model and its predictions on the test data
    """
    model = xgb.XGBClassifier(n_estimators=n_estimators, enable_categorical=True)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions


def evaluate_predictions(predictions, y_test):
    """
    Produce metrics for the provided predictions.
    :param predictions: The output predictions of a model
    :param y_test: The test data containing the actual values
    :return: The accuracy, precision, recall, F1 score and confusion matrix of the predictions
    """
    # Transform the predictions back to original labels
    predictions = le.inverse_transform(predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="DIED")
    recall = recall_score(y_test, predictions, pos_label="DIED")

    f1 = f1_score(y_test, predictions, pos_label="DIED")

    # Confusion Matrix with percentages per category
    cm = confusion_matrix(y_test, predictions)
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return accuracy, precision, recall, f1, cm_percentage


def xgb_feature_importance(model, model_name):
    """
    Given an XGB model plot the feature importance
    :param model_name: Label used to title and save the plot
    :param model: The trained XGBoost model
    """
    # Feature importance
    feature_importance = model.feature_importances_

    # Plotting feature importance
    plt.barh(features, feature_importance)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance for {}".format(model_name))

    plt.savefig("{}.png".format(model_name))


def build_and_test_model(score_data, model_name=None, n_estimators=10, output_stats=True, plot=False):
    """
    Function to train and test an XGBoost model with the specified n_estimators.
    :param score_data: Dataset to train model on
    :param model_name: String used to label the output statistics and chart
    :param n_estimators:
    :param output_stats: Boolean to decide whether to print model metrics and confusion matrix
    :param plot: Boolean to decide whether to plot the feature importance of the model
    :return:
    """
    X_train, X_test, y_train, y_test = data_setup(score_data)

    model, predictions = xgb_predictions(X_train, X_test, y_train, n_estimators)

    if plot:
        xgb_feature_importance(model, model_name)

    accuracy, precision, recall, f1, cm = evaluate_predictions(predictions, y_test)

    if output_stats:
        print("Stats for {}".format(model_name))
        print("Accuracy: {:.2f}%".format((accuracy * 100)))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1-Score: {:.2f}".format(f1))
        print("Confusion Matrix:\n{}\n".format(cm))

    return accuracy, precision, recall, f1


def grid_search(data, n_estimators_values, plot_results=True):
    """
    Given the relevant dataset containing both training and test data this will perform a grid search with the provided
    hyperparameter values. It will return the value which produced the highest F1 score and produce a plot showing how
    the score changed the hyperparameter values.
    :param data: Dataset containing training and test data
    :param n_estimators_values:
    :param plot_results: Boolean, plot a line chart showing the scores per value
    :return: The best value for producing a high F1 score
    """
    search_results = {}

    # For each value test it in a model and save the metrics
    for n_estimators in n_estimators_values:
        accuracy, precision, recall, f1 = build_and_test_model(data, n_estimators=n_estimators, output_stats=False)
        search_results.update({n_estimators: f1})

    if plot_results:
        plt.plot(search_results.keys(), search_results.values())
        plt.xlabel('n_estimators')
        plt.ylabel('F-1 Score')
        plt.title('Grid Search for n_estimators')
        plt.show()

    # Selecting the best value
    best_result = max(search_results, key=search_results.get)
    best_value = search_results[best_result]

    print("Best value: {} with F1: {:.2f}".format(best_result, best_value))

    return best_value


build_and_test_model(apache_scores, model_name="default_scores")
# build_and_test_model(apache_scores_limit_2, model_name="2")
# build_and_test_model(apache_scores_limit_3, model_name="3")
# build_and_test_model(apache_scores_limit_5, model_name="3")
