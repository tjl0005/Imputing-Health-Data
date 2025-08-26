"""
Implementation of XGBoost using the apache scores. Intended to be way of testing different imputation methods and
sampling techniques.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV

from code.constants import (GRID_SEARCH_OUTPUT, PREDICTION_GS_RECORD, GRID_SEARCH_RESULT_COLUMNS, XGBOOST_PARAMS,
                            MEASUREMENTS)

# Used to convert outcome into binary
le = LabelEncoder()


def data_setup(score_data):
    """
    Split the data into training and test data for both the features and predicted values. Using stratified sampling to
    get even class distributions.
    :param score_data: The data to be split
    :return: X_train, X_test, y_train, y_test
    """
    # Splitting into features and target variables
    X = score_data[MEASUREMENTS].copy()
    y = score_data["outcome_encoded"].copy()

    # Splitting into training and test data, stratifying due to limited data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=507, stratify=y)

    return X_train, X_test, y_train, y_test


def cross_validate_xgb(score_data, model_name=None, output_stats=False, show_roc_auc=False,
                       show_feature_importance=False, save_feature_weightings=False, gamma=0.01, learning_rate=0.001,
                       max_depth=3, n_estimators=100, n_splits=5):
    """
    Perform cross-validation for XGBClassifier outside of grid search.
    :param score_data: Full dataset to perform cross-validation on
    :param output_stats: Boolean - Decide whether to print metrics of the model including F-1 and confusion matrix
    :param show_feature_importance: Boolean specifying whether to plot the feature importance plot for the final model.
    :param show_roc_auc: Boolean specifying whether to plot the ROC-AUC curve for the final model.
    :param save_feature_weightings: Boolean specifying whether to save the feature importance's for the final model as
    a csv file.
    :param model_name: Reference for the model when plotting or outputting results
    :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
    :param learning_rate: Step size of optimisation
    :param max_depth: Depth of the tree
    :param n_estimators: Number of trees
    :param n_splits: Number of stratified folds
    :return: Metrics for model performance, average of cross validation
    """
    # Splitting the data and using encoded prediction variable
    X = score_data[MEASUREMENTS].copy()
    y = score_data["outcome_encoded"].copy()

    # Using stratified folds to ensure an even distribution of mortalities
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=507)

    # Track the metrics for each of the models
    accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []

    predictions = None

    # Cross validation across the found splits
    for fold_no, (train_index, test_index) in enumerate(skf.split(X, y), 0):
        fold_no += 1

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Defining and training model with given values
        model = xgb.XGBClassifier(gamma=gamma, learning_rate=learning_rate, max_depth=int(max_depth),
                                  n_estimators=int(n_estimators), enable_categorical=True)
        model.fit(X_train, y_train)

        # Getting predictions on test data
        predictions = model.predict(X_test)

        # Tracking metrics when looking for negative mortality
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions, pos_label=0))
        recalls.append(recall_score(y_test, predictions, pos_label=0))
        f1s.append(f1_score(y_test, predictions, pos_label=0))

        # Finding and tracking AUC
        prob_predictions = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, prob_predictions)
        roc_aucs.append(roc_auc)

    if predictions is None:
        raise AssertionError("No predictions were made")

    # Using feature importance of the last model
    if show_feature_importance:
        xgb_feature_importance(model, model_name, save_feature_weightings=save_feature_weightings)
    if show_roc_auc:
        plot_roc(model, X_test, y_test, model_name)

    # Averaging metrics to get better idea of performance
    accuracy, precision, recall, f1, auc = ((np.mean(accuracies)), np.mean(precisions), np.mean(recalls), np.mean(f1s),
                                            np.mean(roc_aucs)
                                            )

    # Confusion matrix to see what the model is doing prediction wise
    cm = confusion_matrix(y_test, predictions)
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Printing metrics of cross validation
    if output_stats:
        print("Stats for {}".format(model_name))
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1-Score: {:.2f}".format(f1))

        # Labelling confusion matrix for readability
        cm_df = pd.DataFrame(
            cm_percentage,
            index=["True Died", "True Survived"],
            columns=["Predicted Died", "Predicted Survived"]
        )
        print(cm_df.map(lambda x: f"{x:.2%}"))

    return accuracy, precision, recall, f1, auc, cm_percentage


def plot_roc(model, X_test, y_test, model_name):
    """
    Given the XGBoost model, the test training data and test labels and the model name this will plot the ROC curve.
    :param model: The trained prediction model to be used to find the probabilities for ROC
    :param X_test: The feature test dataset
    :param y_test: The label test dataset
    :param model_name: The reference for the model to identify the visualisation and file.
    :return:
    """
    prob_predictions = model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, prob_predictions)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label="ROC curve (AUC = %0.2f)" % roc_auc_score(y_test, prob_predictions))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic - {}".format(model_name))
    plt.legend(loc="lower right")
    plt.savefig("../../visualisations/gs/{}_auc_roc.png".format(model_name))
    plt.close()


def xgb_feature_importance(model, model_name, plot_feature_weightings=True, save_feature_weightings=False):
    """
    Given an XGB model plot the feature importance
    :param model_name: Label used to title and save the plot
    :param model: The trained XGBoost model
    :param plot_feature_weightings: If true, plot feature importance weightings
    :param save_feature_weightings: If true, save feature importance weightings as csv.
    """
    feature_importance = model.feature_importances_

    if plot_feature_weightings:
        # Plotting feature importance
        plt.figure()
        bars = plt.barh(range(len(MEASUREMENTS)), feature_importance)
        plt.yticks(range(len(MEASUREMENTS)), MEASUREMENTS)
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title("Feature Importance for {}".format(model_name))

        for i, (bar, score) in enumerate(zip(bars, feature_importance)):
            plt.text(score + 0.01, i, "{:.3f}".format(score), va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig("../../visualisations/gs/{}_feature_importance.png".format(model_name))
        plt.close()

    if save_feature_weightings:
        feature_importance_df = pd.DataFrame({"Feature": MEASUREMENTS, "Importance": feature_importance})
        feature_importance_df.to_csv("../../data/predictions/{}_feature_importance.csv".format(model_name))


def xgb_grid_search_optimisation(score_data, search_reference="no missing data", save_results=True):
    """
    Perform a grid search hyperparameter optimisation for XGBoost using the specified parameters in constants.py.
    Models are evaluated using accuracy, recall and the F-1 score with cross validation.
    :param score_data: The training data as a dataframe, this will be prepared through the defined function prior.
    :param search_reference: Used to label saved results
    :param save_results: Boolean flag specifying whether the results will be saved to a csv file or not.
    """
    # Setting up model for grid search
    xgb_model = xgb.XGBClassifier(enable_categorical=True)
    stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=507)
    X_train, X_test, y_train, y_test = data_setup(score_data)

    # Defining and performing grid search with evaluation of F1
    bayes_search = BayesSearchCV(estimator=xgb_model, search_spaces=XGBOOST_PARAMS,
                                 scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                 refit="roc_auc", n_iter=20, cv=stratified_cv, verbose=0)
    bayes_search.fit(X_train, y_train)

    # Display the best parameters and results
    print(bayes_search.best_params_)
    print(bayes_search.best_score_)

    # Saving results of grid search in order of F1 score
    df = pd.DataFrame(bayes_search.cv_results_)
    df = df[GRID_SEARCH_RESULT_COLUMNS]
    df = df.sort_values("mean_test_roc_auc", ascending=False)

    # Recording the best results of all grid searches with given reference
    best_result = df.iloc[0].to_frame().T

    if save_results:
        save_dir = GRID_SEARCH_OUTPUT + search_reference + ".csv"
        df.to_csv(save_dir, index=False)
        best_result["test_name"] = search_reference

        if not os.path.exists(PREDICTION_GS_RECORD):
            best_result.to_csv(PREDICTION_GS_RECORD, index=False)
        else:
            best_result.to_csv(PREDICTION_GS_RECORD, mode="a", header=False, index=False)

        return None
    else:
        return best_result


def plot_grid_search_results(gs_results, reference):
    """
    Used to plot the findings of the grid search. Each hyperparameter value is plotted for each of the metrics to see
    influence.
    :param gs_results: The previous grid search results as a dataframe.
    :param reference: String to label the plot with
    """
    # Columns containing the parameter values used
    param_columns = [col for col in gs_results.columns if col.startswith("param_")]

    # Grouping results for plotting
    grouped_results = gs_results.groupby(param_columns).agg(
        accuracy_mean=("mean_test_accuracy", "mean"),
        accuracy_std=("std_test_accuracy", "mean"),
        precision_mean=("mean_test_precision", "mean"),
        precision_std=("std_test_precision", "mean"),
        recall_mean=("mean_test_recall", "mean"),
        recall_std=("std_test_recall", "mean"),
        f1_mean=("mean_test_f1", "mean"),
        f1_std=("std_test_f1", "mean"),

    ).reset_index()

    # Setting up plot with details
    fig, ax = plt.subplots(1, len(param_columns), figsize=(20, 5))
    fig.suptitle("Metrics from Grid Search")
    fig.text(0, 0.5, "Score", va="center", rotation="vertical")

    # Plotting results for each of the hyperparameters
    for i, p in enumerate(param_columns):
        # Sorting so in order of the current parameter
        sorted_group = grouped_results.sort_values(by=p)

        # Plotting error bars for the metrics
        ax[i].errorbar(sorted_group[p], sorted_group["accuracy_mean"], sorted_group["accuracy_std"], linestyle="--",
                       label="Accuracy")
        ax[i].errorbar(sorted_group[p], sorted_group["precision_mean"], sorted_group["precision_std"], linestyle="--",
                       label="Precision")
        ax[i].errorbar(sorted_group[p], sorted_group["recall_mean"], sorted_group["recall_std"], linestyle="-",
                       label="Recall")
        ax[i].errorbar(sorted_group[p], sorted_group["f1_mean"], sorted_group["f1_std"], linestyle="-.",
                       label="F1 Score")

        ax[i].set_xlabel(p.split("param_")[1].upper())

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("../../visualisations/gs/{}.png".format(reference))
    plt.close()


def summarise_grid_search(score_data, search_reference="no missing data"):
    """
    Given the original dataset used in the grid search and the reference of the grid search (file name) plot the results
    and train and evaluate the model using the best parameters and output full details including confusion matrix and
    feature importance.
    :param score_data: Original unsplit dataset used for training and evaluating the model in the grid search
    :param search_reference: Name of the grid search (file name of results)
    """
    # Reading grid search results from file
    results_dir = GRID_SEARCH_OUTPUT + search_reference + ".csv"
    results = pd.read_csv(results_dir)

    # Plotting the results for all the searches
    plot_grid_search_results(results, search_reference)

    best_result = results.iloc[0]

    # Training best found model, outputting the stats with the confusion matrix and plotting feature importance
    cross_validate_xgb(score_data, search_reference, gamma=best_result["param_gamma"],
                       n_estimators=best_result["param_n_estimators"],
                       max_depth=best_result["param_max_depth"],
                       learning_rate=best_result["param_learning_rate"], output_stats=True,
                       show_feature_importance=True, show_roc_auc=True
                       )
