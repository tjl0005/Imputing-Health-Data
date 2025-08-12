"""
Implementation of XGBoost using the apache scores. Intended to be way of testing different imputation methods and
sampling techniques.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from code.constants import KNN_PARAMS, MICE_PARAMS, GRID_SEARCH_OUTPUT, MEASUREMENTS, NON_OPTIMISED_IMPUTERS, \
    DL_IMPUTERS, RAW_MISSING_LEVELS, NON_DL_IMPUTERS, DOWNSTREAM_METRICS
from code.evaluation.static_prediction import xgb_grid_search_optimisation, cross_validate_xgb
from code.imputation.ml import knn_impute, mice_impute, single_impute
from code.preprocessing.resample import resample_data

le = LabelEncoder()


def get_imputer_types(include_gain=False, include_miwae=False):
    """
    Used to retrieve a list of the imputer types to use, always returns mean, knn and MICE. WGAIN and MIWAE are optional
    :param include_gain: Boolean specifying whether WGAIN results are available.
    :param include_miwae: Boolean specifying whether MIWAE results are available.
    :return: List containing the relevant imputer types. i.e. ["mean", "knn", "mice"]
    """
    # Always testing mean, knn and MICE
    imputer_types = NON_DL_IMPUTERS.copy()

    # Adding DL separately if they are specified
    if include_gain:
        imputer_types.append("gain")
    if include_miwae:
        imputer_types.append("miwae")

    return imputer_types


def test_imputed_data(test_data, downsample=False):
    """
    Given test data, which has been fully imputed this function will test the data with an XGBoost grid search for
    mortality prediction.
    :param test_data: Dataframe with no missing data.
    :param downsample: Boolean to specify whether the data needs to be downsampled prior to testing or not.
    :return: The downstream metrics of the optimal XGBoost model. i.e. ROC-AUC and F1.
    """
    # Defining labels for prediction
    test_data["outcome_encoded"] = le.fit_transform(test_data["outcome"])

    # Resampling otherwise performance poor, reference set to n/a as it is not needed
    if downsample:
        test_data, _ = resample_data(test_data, "n/a")

    # # Testing imputation in downstream with XGBoost
    best_scores = xgb_grid_search_optimisation(test_data, search_reference="testing", save_results=False)

    return best_scores


def imputer_grid_search_optimisation(data_to_impute, missing_level=2, imputation_type="knn", downsample=False):
    """
    Perform a grid search for the provided imputation type and data to optimise the imputation approach
    :param data_to_impute: Artificially missing data from the ground truth.
    :param missing_level: An integer (raw_missing) or float (artificial_missing) representing the level of missing data.
    :param imputation_type: mean, knn or mice with the hyperparameters being pre-defined.
    :param downsample: If True, downsample the data before performing the imputation.
    """
    results = []

    # Getting correct parameters to test
    if imputation_type == "knn":
        param_grid = KNN_PARAMS
    elif imputation_type == "mice":
        param_grid = MICE_PARAMS
    elif imputation_type in NON_OPTIMISED_IMPUTERS:
        # Either simple imputation (no optimisation needed) or DL (optimised in notebook)
        param_grid = {"blank": [0]}
    else:
        raise ValueError(
            "Invalid imputation type chosen. Must be either 'mean', 'knn', 'mice', 'wgain', or 'miwae'")

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
            else:
                imputed_data = single_impute(data, imputation_type)

            # Evaluating on predictive metrics
            optimised_scores = test_imputed_data(imputed_data, downsample).iloc[0].to_dict()

            # Handle the result based on imputation type (separate logic for gain/miwae)
            if imputation_type in DL_IMPUTERS:
                # Append results specifically for gain with additional parameters
                results.append({
                    "Imputation": imputation_type,
                    "Hyperparameter": None,
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
    """
    Perform the grid search for the raw missing data. Optimises a XGBoost model for predicting mortality outcomes for
    each of the imputer datasets.
    """
    raw_data_dir = "../../data/missing/raw/measurements_{}.csv"

    for m_level in RAW_MISSING_LEVELS:
        # GAIN and MIWAE handled in notebooks so just looking at mean, mice and knn
        for imputation_type in NON_DL_IMPUTERS:
            missing_data = pd.read_csv(raw_data_dir.format(m_level))

            print("\nPerforming grid search for {}_missing with {}".format(m_level, imputation_type))

            # Testing with both original and downsampled versions
            imputer_grid_search_optimisation(missing_data, missing_level=m_level, imputation_type=imputation_type,
                                             downsample=True)
            imputer_grid_search_optimisation(missing_data, missing_level=m_level, imputation_type=imputation_type,
                                             downsample=False)


def plot_grid_searches(downsampled=False, include_gain=False, include_miwae=False):
    """
    Plot the raw missing data grid search results. This plots
    :param downsampled: Boolean to specify whether the data needs to be downsampled prior to testing or not.
    :param include_gain: Boolean specifying whether WGAIN results are available.
    :param include_miwae: Boolean specifying whether MIWAE results are available.
    """
    # Decide whether plotting the results with or without downsampling
    if downsampled:
        grid_search_dir = "../../data/grid_searches/raw/missing_{}_down_sampled.csv"
    else:
        grid_search_dir = "../../data/grid_searches/raw/missing_{}.csv"

    imputer_types = get_imputer_types(include_gain, include_miwae)

    # Track the results for all combinations
    complete_results = []

    # Going through each missing level to get the scores
    for missing_level in RAW_MISSING_LEVELS:
        # Reading specified data from the grid search
        optimised_imputation_results = pd.read_csv(grid_search_dir.format(missing_level))
        # Tracking results at this missing level only
        best_results_at_missing_level = []

        # Getting results for each imputation, keeping combinations with the highest ROC-AUC
        for imputation_type in imputer_types:
            if imputation_type in DL_IMPUTERS:
                # Getting the DL result for this missing level, they are already processed by Notebook
                dl_result_dir = "../../data/grid_searches/raw/best_raw_{}_scores.csv".format(imputation_type)
                dl_results_df = pd.read_csv(dl_result_dir)
                best_row = dl_results_df[dl_results_df["reference"] == "raw_{}".format(missing_level)]
            elif imputation_type in NON_DL_IMPUTERS:
                relevant_results = optimised_imputation_results[
                    optimised_imputation_results["Imputation"] == imputation_type]
                # Getting row with the best roc-auc
                best_row = relevant_results.loc[[relevant_results["mean_test_roc_auc"].idxmax()]]
            else:
                raise ValueError("Imputation type {} not recognised".format(imputation_type))

            best_row = best_row.copy()
            best_row["Imputation"] = imputation_type
            best_row["missing_level"] = missing_level
            best_results_at_missing_level.append(best_row)

        # Converting this missing levels results and adding label before appending to overall results
        best_results_df = pd.concat(best_results_at_missing_level, ignore_index=True)
        best_results_df["missing_level"] = missing_level
        complete_results.append(best_results_df)

    # Final results dataframe
    complete_results_df = pd.concat(complete_results, ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Increased figure size for better clarity
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust space between subplots

    # Loop through metrics and plot them on the same shared subplot
    # using downstream, i.e. ROC-AUC and F1
    for i, metric in enumerate(DOWNSTREAM_METRICS):
        # Assuming nRows 2 and nCols 3, so able to identify the relevant row/col.
        row = i // 3
        col = i % 3
        plot_metric(complete_results_df, metric=metric, ax=axes[row, col])

    # Getting the handles and labels of first plot and setting them as figure legend as they are shared
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout()
    plt.savefig("../../visualisations/raw/all_grid_search_results.png")
    plt.show()


def plot_metric(results, metric="roc_auc", ax=None):
    """
    This will create a plot for the given results and metric. It is to be called multiple times for each metric
    so that they are plotted together.

    :param results: Dataframe containing the grid search results.
    :param metric: The downstream metric to plot, default is "roc_auc"
    :param ax: the axis
    """
    # Selecting the relevant columns for this metric
    df_plot = results[["Imputation", "mean_test_{}".format(metric), "std_test_{}".format(metric), "missing_level"]]
    # Checking dynamically as they can vary
    types = df_plot["Imputation"].unique()

    # Initialising the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    colours = plt.cm.Accent(np.linspace(0, 1, len(RAW_MISSING_LEVELS)))

    # Tracking the labels to avoid duplication in the features
    labels_added = {missing_level: False for missing_level in RAW_MISSING_LEVELS}

    # Go through each imputation type and plot the results for each missingness level
    width = 0.25
    x = np.arange(len(types))
    min_score, max_score = 1, 0

    for i, imputation_type in enumerate(types):
        # Select results for the current imputation type
        imputation_data = df_plot[df_plot["Imputation"] == imputation_type]

        # Plotting the metrics for each missing level
        for j, missing_level in enumerate(RAW_MISSING_LEVELS):
            mean_score = imputation_data[imputation_data["missing_level"] == missing_level] \
                ["mean_test_{}".format(metric)].values[0]
            std_score = imputation_data[imputation_data["missing_level"] == missing_level] \
                ["std_test_{}".format(metric)].values[0]

            # Only keep labels on their first occurrence to prevent duplicate legends
            label = ""
            if not labels_added[missing_level]:
                label = "Missing Level {}".format(missing_level)
                labels_added[missing_level] = True

            # Plotting bar and error (standard deviation) for the current results
            ax.bar(x[i] + j * width, mean_score, color=colours[j], width=width, label=label)
            ax.errorbar(x[i] + j * width, mean_score, yerr=std_score, ecolor="red", capsize=5)

            # Tracking bounds for y-axis scaling
            min_score = min(min_score, mean_score - std_score)
            max_score = max(max_score, mean_score + std_score)

    ax.set_title("Mean and Std of {} for Each Imputation Method".format(metric))
    ax.set_xlabel("Imputation Method")
    ax.set_ylabel(metric)

    # Adjusting ticks for readability
    step = (max_score - min_score) / 10
    y_ticks = np.arange(min_score, max_score + 0.01, step=step)
    y_tick_labels = ["{:.2f}".format(tick) for tick in y_ticks]

    ax.set_ylim(min_score, max_score)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xticks(x + width)
    ax.set_xticklabels(types)

    ax.grid(alpha=0.2)


def plot_imputed_distributions(imputed_data, reference):
    """
    This will plot the differences in distributions of the imputed and original datasets. It works
    with both raw and artificially missing data and the histograms are normalised.
    """
    original_features = pd.read_csv("../../data/missing/raw/measurements_0.csv")

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()

    for i, col in enumerate(MEASUREMENTS):
        if col == "anchor_age":
            continue

        ax = axes[i]

        # Density normalises it - look into better colours
        ax.hist(imputed_data[col], alpha=0.5, label="Imputed", color="blue", edgecolor="black", density=True)
        ax.hist(original_features[col], alpha=0.5, label="Original", color="orange", edgecolor="black", density=True)

        ax.set_title(col)
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.legend()

    # Odd number of features so need to remove extra plots
    axes[13].axis("off")
    axes[14].axis("off")
    axes[15].axis("off")

    plt.suptitle("Variable Distributions for {}".format(reference), fontsize=20)
    plt.tight_layout()
    plt.savefig("../../Visualisations/imputed distributions/{}_variable_distributions.png".format(reference))

    plt.close(fig)


def plot_best_imputations(include_gain=False, include_miwae=False):
    """
    Extracts the best identified model configurations and cross-validates them again but producing visualisations for
    the ROC-AUC, feature importance and the imputed distributions. Enables insight into the decision-making of the
    imputers per variable imputations, the feature contributions to the model and how it handles true predictions.
    """
    imputer_types = get_imputer_types(include_gain, include_miwae)

    for m_level in RAW_MISSING_LEVELS:
        # Getting grid search results
        file_dir = GRID_SEARCH_OUTPUT + "/raw/missing_{}.csv".format(m_level)
        search_results = pd.read_csv(file_dir)

        # Identifying which configurations where the best
        best_rows = search_results.loc[search_results.groupby("Imputation")["mean_test_roc_auc"].idxmax()]

        best_configurations = best_rows[
            ["Imputation", "Hyperparameter", "param_gamma", "param_learning_rate", "param_max_depth",
             "param_n_estimators", "mean_test_roc_auc", "std_test_roc_auc"]]

        # Running the imputers again with the best configurations and running XGBoost again but with feature importance.
        for imputer in imputer_types:
            value = best_configurations.loc[best_configurations["Imputation"] == imputer, "Hyperparameter"].values[0]
            test_reference = "{}_{}".format(m_level, imputer)

            # Getting missing data, re-reading each time to avoid conflicts.
            raw_data_dir = "../../data/missing/raw/measurements_{}.csv"
            missing_data = pd.read_csv(raw_data_dir.format(m_level))

            # Run specified imputer with hyperparameters
            if imputer == "knn":
                imputed_data = knn_impute(missing_data, k=value)
            elif imputer == "mice":
                imputed_data = mice_impute(missing_data, max_iter=value)
            else:
                imputed_data = single_impute(missing_data, imputer)

            plot_imputed_distributions(imputed_data, test_reference)

            imputed_data["outcome_encoded"] = le.fit_transform(imputed_data["outcome"])

            gamma = best_configurations.loc[best_configurations["Imputation"] == imputer, "param_gamma"].values[0]
            learning_rate = \
                best_configurations.loc[best_configurations["Imputation"] == imputer, "param_learning_rate"].values[0]
            max_depth = best_configurations.loc[best_configurations["Imputation"] == imputer, "param_max_depth"].values[
                0]
            n_estimators = \
                best_configurations.loc[best_configurations["Imputation"] == imputer, "param_n_estimators"].values[0]

            # # Testing imputation in downstream with XGBoost but now showing plots as well
            best_scores = cross_validate_xgb(imputed_data, model_name=test_reference, show_feature_importance=True,
                                             show_roc_auc=True, gamma=gamma, learning_rate=learning_rate,
                                             max_depth=max_depth, n_estimators=n_estimators)

            print(imputer, m_level)
            print(best_scores)


perform_grid_searches()
plot_grid_searches(downsampled=False, include_gain=False, include_miwae=False)
plot_grid_searches(downsampled=True, include_gain=False, include_miwae=False)
plot_best_imputations(include_gain=False, include_miwae=False)
