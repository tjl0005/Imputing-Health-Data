"""
Implementation of XGBoost using the apache scores. Intended to be way of testing different imputation methods and
sampling techniques.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from code.constants import KNN_PARAMS, MICE_PARAMS, GRID_SEARCH_OUTPUT, GAIN_PARAMS
from code.evaluation.static_prediction import cross_validate_xgb, xgb_grid_search_optimisation
from code.imputation.ml import knn_impute, mice_impute, single_impute
from code.preprocessing.resample import resample_data

le = LabelEncoder()

MISSING_TYPES = ["mcar", "mnar_central", "mnar_lower", "mnar_upper"]


def test_imputed_data(test_data, resample=True):
    # Resampling otherwise performance poor, reference set to n/a as it is not needed
    if resample:
        test_data, _ = resample_data(test_data, "n/a")

    # # Testing imputation in downstream with XGBoost
    best_scores = xgb_grid_search_optimisation(test_data, search_reference="testing", save_results=False)

    return best_scores


def imputer_grid_search_optimisation(data_to_impute, missing_level=2, imputation_type="knn", downsample=True):
    """
    Perform a grid search for the provided imputation type and data to optimise the imputation approach
    :param data_to_impute: Artificially missing data from the ground truth
    :param missing_level:
    :param imputation_type: mean, knn or mice with the hyperparameters being pre-defined
    """
    results = []

    # Getting correct parameters to test
    if imputation_type == "knn":
        param_grid = KNN_PARAMS
    elif imputation_type == "mice":
        param_grid = MICE_PARAMS
    elif imputation_type == "gain":
        param_grid = GAIN_PARAMS
    else:
        # Either simple imputation (no optimisation needed) or gain (optimised in notebook)
        param_grid = {"blank": [0]}

    # Going through the hyperparameters
    for key in param_grid:
        for value in param_grid[key]:
            # Required to avoid overwriting reference
            data = data_to_impute.copy()
            print("Testing {} with {} at {}".format(imputation_type, key, value))

            # Run specified imputer with hyperparameters
            if imputation_type == "knn":
                imputed_data = knn_impute(data, k=value)
            elif imputation_type == "mice":
                imputed_data = mice_impute(data, max_iter=value)
            elif imputation_type == "gain":
                imputed_data = pd.read_csv("../../data/imputed/raw/gain/{}_{}.csv".format(missing_level, value))
            else:
                imputed_data = single_impute(data, imputation_type)

            # Defining labels for prediction
            imputed_data["outcome_encoded"] = le.fit_transform(imputed_data["outcome"])

            # Evaluating on predictive metrics
            optimised_scores = test_imputed_data(imputed_data, downsample).iloc[0].to_dict()

            # Handle the result based on imputation type (separate logic for gain)
            if imputation_type == "gain":
                result_reference = "gain_{}".format(value)
                # Append results specifically for gain with additional parameters
                results.append({
                    "Imputation": result_reference,
                    "Hyperparameter": value,
                    **optimised_scores
                })
            else:
                # For other imputation methods, we append normally
                results.append({
                    "Imputation": imputation_type,
                    "Hyperparameter": value,
                    **optimised_scores
                })

    if downsample:
        save_dir = GRID_SEARCH_OUTPUT + "/raw/missing_" + str(missing_level) + "_down_sampled.csv"
    else:
        save_dir = GRID_SEARCH_OUTPUT + "/raw/missing_" + str(missing_level) + ".csv"
    results_df = pd.DataFrame(results)

    # Saving results to a new file if no previous searches or appending to file if matching previous search
    if not os.path.exists(save_dir):
        results_df.to_csv(save_dir, index=False)
    else:
        results_df.to_csv(save_dir, mode="a", header=False, index=False)


def perform_grid_searches():
    raw_data_dir = "../../data/missing/raw/measurements_{}.csv"
    levels_of_missingness = [2, 5, 10]

    for m_level in levels_of_missingness:
        # for imputation_type in ["gain", "mean", "knn", "mice"]:
        for imputation_type in ["mean", "knn", "mice"]:
            missing_data = pd.read_csv(raw_data_dir.format(m_level))

            if imputation_type == "gain":
                print("Testing gain with {}_missing with {} temporary imputation".format(m_level, imputation_type))
            else:
                print("\nPerforming grid search for {}_missing with {}".format(m_level, imputation_type))

            imputer_grid_search_optimisation(missing_data, missing_level=m_level, imputation_type=imputation_type, downsample=True)
            imputer_grid_search_optimisation(missing_data, missing_level=m_level, imputation_type=imputation_type, downsample=False)


def plot_grid_searches(downsampled=False):
    # Decide whether plotting the results with or without downsampling
    if downsampled:
        grid_search_dir = "../../data/grid_searches/raw/missing_{}_down_sampled.csv"
    else:
        grid_search_dir = "../../data/grid_searches/raw/missing_{}.csv"

    # Change to constant
    types = ["gain", "mean", "knn", "mice"]
    missing_levels = [2, 5, 10]
    metrics = ["precision", "recall", "accuracy", "f1", "roc_auc"]

    # Track the results for all combinations
    complete_results = []

    # Going through each missing level to get the scores
    for missing_level in missing_levels:
        # Reading specified data from the grid search
        optimised_imputation_results = pd.read_csv(grid_search_dir.format(missing_level))
        # Tracking results at this missing level only
        best_results_at_missing_level = []

        # Getting results for each imputation, keeping combinations with the highest ROC-AUC
        for imputation_type in types:
            for metric in metrics:
                if imputation_type == "gain":
                    relevant_results = optimised_imputation_results[optimised_imputation_results["Imputation"].str.contains(imputation_type)]
                else:
                    relevant_results = optimised_imputation_results[optimised_imputation_results["Imputation"] == imputation_type]

                best_imputation_result = relevant_results[relevant_results["mean_test_{}".format(metric)] == relevant_results["mean_test_{}".format(metric)].min()].drop_duplicates(subset=["Imputation"])

            if imputation_type == "gain":
                best_imputation_result["Imputation"] = "gain"

            best_results_at_missing_level.append(best_imputation_result)

        # Converting this missing levels results and adding label before appending to overall results
        best_results_df = pd.concat(best_results_at_missing_level, ignore_index=True)
        best_results_df['missing_level'] = missing_level
        complete_results.append(best_results_df)

    # Final results dataframe
    complete_results_df = pd.concat(complete_results, ignore_index=True)

    plot_metrics(complete_results_df, metric="precision")
    plot_metrics(complete_results_df, metric="recall")
    plot_metrics(complete_results_df, metric="accuracy")
    plot_metrics(complete_results_df, metric="f1")
    plot_metrics(complete_results_df, metric="roc_auc")


def plot_metrics(results, metric="roc_auc"):
    # Selecting the relevant columns
    df_plot = results[["Imputation", "mean_test_{}".format(metric), "std_test_{}".format(metric), "missing_level"]]

    missing_levels = [2, 5, 10]
    types = df_plot["Imputation"].unique()

    plt.figure(figsize=(12, 6))

    min_score, max_score = 1, 0
    width = 0.25
    x = np.arange(len(types))

    # Find better colour map
    colours = plt.cm.Accent(np.linspace(0, 1, len(missing_levels)))

    # Tracking labels to avoid duplication
    labels_added = {missing_level: False for missing_level in missing_levels}

    # Go through each imputation type and plot the results for each missingness level
    for i, imputation_type in enumerate(types):
        # Select results for the current imputation type
        imputation_data = df_plot[df_plot["Imputation"] == imputation_type]

        # Plotting the metrics for each missing level
        for j, missing_level in enumerate(missing_levels):
            print(imputation_type, missing_level)
            mean_score = imputation_data[imputation_data["missing_level"] == missing_level]["mean_test_{}".format(metric)].values[0]
            std_score = imputation_data[imputation_data["missing_level"] == missing_level]["std_test_{}".format(metric)].values[0]

            # Only keeping labels on their first occurrence so legned works properly
            label = ""

            if not labels_added[missing_level]:
                label = f"Missing Level {missing_level}"
                labels_added[missing_level] = True

            # Plotting bar and error (standard deviation) for the current results
            plt.bar(x[i] + j * width, mean_score, color=colours[j], width=width, label=label)
            plt.errorbar(x[i] + j * width, mean_score, yerr=std_score, ecolor='red', capsize=5)

            # Tracking bounds
            min_score = min(min_score, mean_score - std_score)
            max_score = max(max_score, mean_score + std_score)

    plt.title("Mean and Std of {} for Each Imputation Method and Missingness Level".format(metric))
    plt.xlabel("Imputation Method")
    plt.ylabel(metric)

    # Dynamically adjusting the ticks to match the score range and rounding to 2 d.p
    step = (max_score - min_score) / 10
    y_ticks = np.arange(min_score, max_score + 0.01 , step=step)
    y_tick_labels = ["{:.2f}".format(tick) for tick in y_ticks]
    plt.ylim(min_score, max_score)
    plt.yticks(y_ticks, y_tick_labels)
    plt.xticks(x + width, types)

    plt.grid(alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("../../visualisations/raw/{}_grid_search_results.png".format(metric))


perform_grid_searches()
plot_grid_searches(downsampled=False)
plot_grid_searches(downsampled=True)