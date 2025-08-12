"""
Using the data from feature selection with no missing values, which have been modified to be artificially missing data
that has been imputed, this file will evaluate the imputations using the ground truth through MAE.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from code.constants import (MEASUREMENTS, GRID_SEARCH_OUTPUT, MISSING_TYPES, ARTIFICIAL_MISSING_DATA_DIR, KNN_PARAMS,
                            MICE_PARAMS, MICE_FOREST_PARAMS, ARTIFICIAL_MISSING_PERCENTAGES, NON_DL_IMPUTERS)
from code.imputation.ml import knn_impute, mice_impute, single_impute, mice_forest_impute

ground_truth_data = pd.read_csv("../../data/missing/raw/measurements_0.csv")

# Excluding age and changing percentages to strings so they can be used properly.
MEASUREMENTS = [measurement for measurement in MEASUREMENTS if measurement not in ["anchor_age"]]
STRING_PERCENTAGES = ["{}%".format(int(p * 100)) for p in ARTIFICIAL_MISSING_PERCENTAGES]


def evaluate_ground_truth(imputed_data, mask):
    """
    Evaluate imputation using both raw MAE and IQR-Normalised MAE for each of the features.
    :param imputed_data: A dataframe containing the imputed data to be evaluated.
    :param mask: The missing mask representing which values are imputations.
    :return: Two dataframes containing the per-feature mean absolute error and the per feature normalised mean absolute
     error.
    """
    raw_mae_results = {}
    normalised_mae_results = {}
    normalised_mae_values = []

    # Calculate MAE and nMAE for each measurement
    for measurement in MEASUREMENTS:
        # Selecting on imputed values for comparison
        missing_ground_truth = ground_truth_data[measurement][mask[measurement]]
        imputed_values = imputed_data[measurement][mask[measurement]]

        # Raw MAE
        mae = mean_absolute_error(missing_ground_truth, imputed_values)
        raw_mae_results[measurement] = mae

        # Using IQR for the normalisation
        q1 = ground_truth_data[measurement].quantile(0.25)
        q3 = ground_truth_data[measurement].quantile(0.75)
        iqr = q3 - q1

        # Normalising
        normalised_mae = mae / iqr

        # Storing the normalised results
        normalised_mae_results[measurement] = normalised_mae
        normalised_mae_values.append(normalised_mae)

    # Average normalised MAE across features
    normalised_mae_results["average_normalised_mae"] = np.mean(normalised_mae_values)

    # Returning both the raw MAE and the normalised version
    return raw_mae_results, normalised_mae_results


def grid_search_optimisation(missing_data, imputation_type="knn", file_reference="mcar"):
    """
    Perform a grid search for the provided imputation type and data to optimise the imputation approach
    :param missing_data: Artificially missing data from the ground truth
    :param imputation_type: mean, knn or mice with the hyperparameters being pre-defined
    :param file_reference: Specification of the type of missing data to identify results
    """
    # Tracking both the Mean Absolute Error and the Normalised Mean Absolute Error
    raw_results, norm_results = [], []
    # Represents where the missing data is, i.e. mask of 1"s and 0"s
    mask = missing_data.isna()

    # Getting correct parameters to test from constants.py
    if imputation_type == "knn":
        param_grid = KNN_PARAMS
    elif imputation_type == "mice":
        param_grid = MICE_PARAMS
    elif imputation_type == "mice_forest":
        param_grid = MICE_FOREST_PARAMS
    else:
        # Doing simple imputation (i.e. mean) so skipping grid search and running once
        param_grid = {"blank": [0]}

    # Going through the parameter grid and testing. One dimensional as imputers only have 1 value to test.
    for key in param_grid:
        for value in param_grid[key]:
            # Required to avoid overwriting reference
            data = missing_data.copy()
            print("Testing {} with {} at {}".format(imputation_type, key, value))

            if imputation_type == "knn":
                imputed_data = knn_impute(data, k=value)
            elif imputation_type == "mice":
                # Handling different hyperparameters. Not actually utilised in current code as to expensive.
                if key == "max_iters":
                    imputed_data = mice_impute(data, max_iter=value)
                else:
                    imputed_data = mice_impute(data, estimator_choice=value)
            elif imputation_type == "mice_forest":
                imputed_data = mice_forest_impute(data, max_iter=value)
            else:
                imputed_data = single_impute(data, imputation_type)

            raw_mae_results, norm_mae_results = evaluate_ground_truth(imputed_data, mask)

            # Storing the raw results
            raw_entry = {"imputation_type": imputation_type, "hyperparameter": value}
            raw_entry.update(raw_mae_results)
            raw_results.append(raw_entry)

            # Storing the normalised results
            norm_entry = {"imputation_type": imputation_type, "hyperparameter": value}
            norm_entry.update(norm_mae_results)
            norm_results.append(norm_entry)

    # Preparing to save results
    raw_save_dir = GRID_SEARCH_OUTPUT + "/ground_truth/raw_" + file_reference + ".csv"
    norm_save_dir = GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv"

    raw_results_df = pd.DataFrame(raw_results)
    norm_results_df = pd.DataFrame(norm_results)

    # Saving results to a new file if no previous searches or appending to file if matching previous search
    if not os.path.exists(raw_save_dir):
        raw_results_df.to_csv(raw_save_dir, index=False)
    else:
        raw_results_df.to_csv(raw_save_dir, mode="a", header=False, index=False)

    # Repeating for normalised results
    if not os.path.exists(norm_save_dir):
        norm_results_df.to_csv(norm_save_dir, index=False)
    else:
        norm_results_df.to_csv(norm_save_dir, mode="a", header=False, index=False)


def grid_search_artificially_missing():
    """
    Wrapper to go through all artificially missing data and perform grid searches for them on k-NN and MICE
    """
    for missing_percentage in ARTIFICIAL_MISSING_PERCENTAGES:
        for m_type in MISSING_TYPES:
            print("Performing grid search for {} with {}% missing".format(m_type, missing_percentage * 100))
            file_reference = "{}_{}".format(m_type, missing_percentage)

            missing_dir = "{}/measurements_{}_{}.csv".format(ARTIFICIAL_MISSING_DATA_DIR, missing_percentage, m_type)
            missing_data = pd.read_csv(missing_dir)

            # Optimising each of the imputation types for each type of missing data
            grid_search_optimisation(missing_data, imputation_type="mean", file_reference=file_reference)
            grid_search_optimisation(missing_data, imputation_type="knn", file_reference=file_reference)
            grid_search_optimisation(missing_data, imputation_type="mice", file_reference=file_reference)


def extract_nmae_values(data, individual=True, imputation_types=None):
    """
    After a completed grid search extract either the best individual or average nMAE scores for each of the imputation
    types
    :param data: Grid search results
    :param individual: Boolean to specify whether to return individual nMAE results or the average
    :param imputation_types: The types of imputation tested in the grid search
    :return: The best individual or average nMAE results
    """
    if imputation_types is None:
        imputation_types = NON_DL_IMPUTERS

    # Store results for each of the imputation types
    nmae_data = {"mean": [], "knn": [], "mice": []}

    for imputation_type in imputation_types:
        # Selecting the specific results for the current imputation type
        imputation_results = data[data["imputation_type"] == imputation_type]
        nmae = []

        # Get individual measurement scores or the average result
        if individual:
            for measurement in MEASUREMENTS:
                measurement_nmae = imputation_results[measurement].values
                nmae.append(np.min(measurement_nmae))
        else:
            nmae = np.min(imputation_results["average_normalised_mae"].values)

        nmae_data[imputation_type] = nmae

    return nmae_data


def deep_learning_scores(dl_type, m_type, m_percent, average=True):
    """
    If the gain scores exist from the notebook then this will extract the relevant nmae scores. Either average for the
     given reference (missing type and percentage) or the nmae for all the MEASUREMENTS.
    :param dl_type: String representing which DL imputer is being referred to. i.e. wgain or miwae
    :param m_type: String representing the missing type, i.e. MCAR.
    :param m_percent: Float representing the percentage of missing data. i.e. o.2
    :param average: Boolean to specify whether to return the average score or the individual nMAE scores per feature.
    :return: Either the average nMAE of the imputer or the individual nMAE scores per feature.
    """
    if dl_type == "wgain":
        dl_scores = pd.read_csv("../../data/grid_searches/ground_truth/artificial_wgain_scores.csv")
    elif dl_type == "miwae":
        dl_scores = pd.read_csv("../../data/grid_searches/ground_truth/artificial_miwae_scores.csv")
    else:
        raise ValueError("Invalid deep learning type for scores")

    # Limiting findings to just those matching reference
    m_type_scores = dl_scores[(dl_scores["missing_type"] == m_type) & (dl_scores["missing_level"] == m_percent)]

    # Either return the average nMAE or all the MAE's for each feature.
    if average:
        return m_type_scores["average_norm_mae"].values[0]
    else:
        return m_type_scores[MEASUREMENTS].values[0]


def format_label(missing_type):
    """
    Given the missing type this function will return a formatted string i.e. given "mnar_upper" it returns MNAR (Upper).
    :param missing_type: String containing the missing type reference.
    :return: Formatted string to be used as a label in plots.
    """
    # Applying formatting to the plot labels
    split_m_type = missing_type.split("_")

    if len(split_m_type) == 2:
        label = "{} ({})".format(split_m_type[0].upper(), split_m_type[1].capitalize())
    else:
        label = split_m_type[0].upper()

    return label


def plot_individual_nmae(include_gain=False, include_miwae=True):
    """
    Plot the individual nmae results for each of the MEASUREMENTS. A plot is created for each type of missing data and
    stored in the visualisations/ground_truth folder.
    :param include_gain: Boolean to confirm whether the GAIN findings are available. If so they will be included.
    :param include_miwae: Boolean to confirm whether the MIWAE findings are available. If so they will be included.
    """
    # Individual plots for each of the missing types.
    for m_type in MISSING_TYPES:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axes = axes.flatten()

        for i, missing_percentage in enumerate(ARTIFICIAL_MISSING_PERCENTAGES):
            ax = axes[i]

            file_reference = "{}_{}".format(m_type, missing_percentage)
            # Reading the relevant grid search the missing type and extracting the individual nmae scores
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv")
            nmae_results = extract_nmae_values(data)

            x = np.arange(len(MEASUREMENTS)) * 1.2
            width = 0.2

            ax.bar((x - 2 * width), nmae_results["mean"], width=width, label="mean")
            ax.bar((x - 1 * width), nmae_results["knn"], width=width, label="k-NN")
            ax.bar((x + 0 * width), nmae_results["mice"], width=width, label="MICE")

            if include_gain:
                m_type_gain_nmae = deep_learning_scores("gain", m_type, missing_percentage, average=False)
                nmae_results["gain"] = m_type_gain_nmae.flatten()
                ax.bar((x + 1 * width), nmae_results["gain"], width=width, label="GAIN")
            if include_miwae:
                m_type_miwae_nmae = deep_learning_scores("miwae", m_type, missing_percentage, average=False)
                nmae_results["miwae"] = m_type_miwae_nmae.flatten()
                ax.bar((x + 2 * width), nmae_results["miwae"], width=width, label="MIWAE")

            # Labels
            ax.set_xlabel("Feature")
            ax.set_ylabel("Normalised MAE")
            ax.set_xticks(x)
            ax.set_xticklabels(MEASUREMENTS, rotation=45, ha="right")

            ax.set_title("{}% Missing".format(missing_percentage * 100))

        axes[3].axis("off")
        handles, labels = ax.get_legend_handles_labels()

        label = format_label(m_type)

        fig.legend(handles, labels, ncol=5, loc="lower center")
        fig.suptitle("Normalised MAE per Feature for {}".format(label), fontsize=16)

        plt.tight_layout()
        plt.savefig("../../visualisations/ground_truth/{}_feature_mae.png".format(m_type))


def plot_average_nmae(include_gain=False, include_miwae=True):
    """
    Plot the average nmae results for each type of missing data, with the plot stored in the
    visualisations/ground_truth folder.
    :param include_gain: Boolean to confirm whether the GAIN findings are available. If so they will be included.
    :param include_miwae: Boolean to confirm whether the MIWAE findings are available. If so they will be included.
    """
    for missing_percentage in ARTIFICIAL_MISSING_PERCENTAGES:
        nmae_data = {"mean": [], "knn": [], "mice": []}

        # Updating dictionary to include optional imputation results
        if include_gain:
            nmae_data["gain"] = []
        if include_miwae:
            nmae_data["miwae"] = []

        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv")
            nmae_results = extract_nmae_values(data, individual=False)

            nmae_data["mean"].append(nmae_results["knn"])
            nmae_data["knn"].append(nmae_results["knn"])
            nmae_data["mice"].append(nmae_results["mice"])

            # Gain results structured differently so extracted differently
            if include_gain:
                m_type_gain_nmae = deep_learning_scores("gain", m_type, missing_percentage, average=True)
                nmae_data["gain"].append(m_type_gain_nmae)
            if include_miwae:
                m_type_miwae_nmae = deep_learning_scores("miwae", m_type, missing_percentage, average=True)
                nmae_data["miwae"].append(m_type_miwae_nmae)

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(MISSING_TYPES)) * 1.2
        width = 0.2

        # Create bars for each imputation type with the relevant nmae data
        ax.bar((x - 2 * width), nmae_data["mean"], width=width, label="mean")
        ax.bar((x - 1 * width), nmae_data["knn"], width=width, label="k-NN")
        ax.bar((x + 0 * width), nmae_data["mice"], width=width, label="MICE")

        if include_gain:
            ax.bar((x + 1 * width), nmae_data["gain"], width=width, label="GAIN")
        if include_miwae:
            ax.bar((x + 2 * width), nmae_data["miwae"], width=width, label="MIWAE")

        # Add labels and title
        ax.set_xlabel("Missing Type")
        ax.set_ylabel("Normalised MAE")
        ax.set_title(
            "Comparison of Normalised MAE between Imputation Types For Different Types of Missingness at {}%".format(
                missing_percentage * 100))
        ax.set_xticks(x)
        ax.set_xticklabels(MISSING_TYPES, rotation=45, ha="right")
        ax.legend(title="Imputation Type", loc="lower right")

        # Display the plot
        plt.tight_layout()
        plt.savefig("../../visualisations/ground_truth/average_nmae_{}.png".format(missing_percentage))


def plot_nmae_with_missing_rates(include_gain=True, include_miwae=True, separate_plots=True):
    """
    Plot the NMAE for each of the missing types in a 2 by 2 subplot, showing the NMAE per missing level for each of the
    imputers.
    :param include_gain: Boolean to confirm whether the GAIN findings are available. If so they will be included.
    :param include_miwae: Boolean to confirm whether the MIWAE findings are available. If so they will be included.
    :param separate_plots:
    """
    # Dictionary to contain the findings for all the missing types for each imputation type
    nmae_data = {method: {m_type: [] for m_type in MISSING_TYPES} for method in
                 ["mean", "knn", "mice", "gain", "miwae"]}

    # Getting the required nmae data
    for missing_percentage in ARTIFICIAL_MISSING_PERCENTAGES:
        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)

            # Read the grid search results for the specific missing data and imputation type
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv")

            # Extract average nmae for each imputation type
            nmae_results = extract_nmae_values(data, individual=False)  # Only average nmae

            # Append the average nmae for each imputation type and missing data type to the nmae_data dictionary
            nmae_data["mean"][m_type].append(nmae_results["mean"])
            nmae_data["knn"][m_type].append(nmae_results["knn"])
            nmae_data["mice"][m_type].append(nmae_results["mice"])

            # Get the average nmae for the DL imputations from their separate results file
            if include_gain:
                m_type_gain_nmae = deep_learning_scores("gain", m_type, missing_percentage, average=True)
                nmae_data["gain"][m_type].append(m_type_gain_nmae)
            if include_miwae:
                m_type_miwae = deep_learning_scores("miwae", m_type, missing_percentage, average=True)
                nmae_data["miwae"][m_type].append(m_type_miwae)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    # Plotting the nmae data
    for i, m_type in enumerate(MISSING_TYPES):
        ax = axes[i]

        # Plotting data for current missing type with the nmae of each imputation type
        ax.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["mean"][m_type], marker="x", label="Mean", linestyle="-")
        ax.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["knn"][m_type], marker="x", label="k-NN", linestyle="-")
        ax.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["mice"][m_type], marker="x", label="MICE", linestyle="-")

        if include_gain:
            ax.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["gain"][m_type], marker="x", label="GAIN", linestyle="-")
        if include_miwae:
            ax.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["miwae"][m_type], marker="x", label="MIWAE",
                    linestyle="-")

        # Set the labels and title for each subplot
        ax.set_xlabel("Missing Percentage")
        ax.set_ylabel("Average nMAE")
        ax.set_title(m_type)

        ax.set_xticks(ARTIFICIAL_MISSING_PERCENTAGES)
        ax.set_xticklabels(STRING_PERCENTAGES)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle("Average Normalised MAE Across Imputation Methods and Missing Data Type", fontsize=16)

    plt.subplots_adjust(hspace=0.3, bottom=0.1, top=0.92)
    plt.savefig("../../visualisations/ground_truth/nmae_vs_missing_percentage_subplots.png")

    # Creating 4 separate plots with the same information
    if separate_plots:
        for m_type in MISSING_TYPES:
            plt.figure(figsize=(6, 4))

            plt.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["mean"][m_type], marker="x", label="Mean", linestyle="-")
            plt.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["knn"][m_type], marker="x", label="k-NN", linestyle="-")
            plt.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["mice"][m_type], marker="x", label="MICE", linestyle="-")

            if include_gain:
                plt.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["gain"][m_type], marker="x", label="GAIN",
                         linestyle="-")
            if include_miwae:
                plt.plot(ARTIFICIAL_MISSING_PERCENTAGES, nmae_data["miwae"][m_type], marker="x", label="MIWAE",
                         linestyle="-")

            label = format_label(m_type)

            plt.xlabel("Missing Percentage")
            plt.ylabel("Average nMAE")
            plt.title("Average Normalised MAE when Data is {}".format(label))
            plt.xticks(ARTIFICIAL_MISSING_PERCENTAGES, STRING_PERCENTAGES)
            plt.legend()
            plt.tight_layout()

            output_path = "../../visualisations/ground_truth/nmae_vs_missing_percentage_{}.png".format(m_type)
            plt.savefig(output_path)
            plt.close()


# Complete a grid search whilst recording the nMAE
# grid_search_artificially_missing()

# Visualisation for the individual feature nMAES.
plot_individual_nmae(include_gain=True, include_miwae=True)
# Visualisation for the average nMAE as the combination of missing level and type changes.
plot_average_nmae(include_gain=True, include_miwae=True)
# Visualisation for how the nMAE changes with different levels of missingness
plot_nmae_with_missing_rates(include_gain=True, include_miwae=True)
