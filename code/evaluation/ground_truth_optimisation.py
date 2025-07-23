"""
Using the data from feature selection with no missing values, which have been modified to be artificially missing data
that has been imputed, this file will evaluate the imputations using the ground truth through MAE. This is not
currently implemented in main.py.
# """
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from code.constants import (MEASUREMENTS, GRID_SEARCH_OUTPUT, MISSING_TYPES, ARTIFICIAL_MISSING_DATA_DIR, KNN_PARAMS,
                            MICE_PARAMS, MICE_FOREST_PARAMS)
from code.imputation.ml import knn_impute, mice_impute, single_impute, mice_forest_impute

ground_truth_data = pd.read_csv("../../data/missing/raw/measurements_0.csv")

# Need to resolve this, probably get rid of constants as its more confusing
MEASUREMENTS = [measurement for measurement in MEASUREMENTS if measurement not in ["anchor_age"]]
MISSING_PERCENTAGES = [0.2, 0.5, 0.7]
STRING_PERCENTAGES = ["{}%".format(int(p * 100)) for p in MISSING_PERCENTAGES]


def evaluate_ground_truth(imputed_data, mask):
    """
    Evaluate imputation using both raw MAE and IQR-Normalised MAE for each of the features.
    :param imputed_data: A dataframe containing the imputed data to be evaluated
    :param mask: THe missing mask representing which values are imputations.
    :return: Two dataframes containing the per feature mean absolute error and the per feature normalised mean absolute
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
        Q1 = ground_truth_data[measurement].quantile(0.25)
        Q3 = ground_truth_data[measurement].quantile(0.75)
        iqr = Q3 - Q1

        # Normalising
        normalised_mae = mae / iqr

        # Stroing the normalised results
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
    norm_results = []
    raw_results = []
    # Represents where the missing data is
    mask = missing_data.isna()

    # Getting correct parameters to test
    if imputation_type == "knn":
        param_grid = KNN_PARAMS
    elif imputation_type == "mice":
        param_grid = MICE_PARAMS
    elif imputation_type == "mice_forest":
        param_grid = MICE_FOREST_PARAMS
    else:
        # Doing simple imputation so skipping grid search and running once for mean and median
        param_grid = {"blank": [0]}

    for key in param_grid:
        for value in param_grid[key]:
            # Required to avoid overwriting reference
            data = missing_data.copy()
            print("Testing {} with {} at {}".format(imputation_type, key, value))

            if imputation_type == "knn":
                imputed_data = knn_impute(data, k=value)
            elif imputation_type == "mice":
                # Handling different hyperparameters
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

            # Storing the normalissed results
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
    Wrapper to go through all artifically missing data and perform grid searches for them on k-NN and MICE
    """
    for missing_percentage in MISSING_PERCENTAGES:
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
    After a completed grid search extract either the best individual or average nmae scores for each of the imputation
    types
    :param data: Grid search results
    :param individual: Boolean to specify whether to return individual Nnmae results or the average
    :param imputation_types: The types of imputation tested in the grid search
    :return: The best individual or average nmae results
    """
    if imputation_types is None:
        imputation_types = ["mean", "knn", "mice"]

    # Store results for each of the imputation types
    nmae_data = {"mean": [], "knn": [], "mice": []}

    for imputation_type in imputation_types:
        # Selecting the specific results for the current imputation type
        imputation_results = data[data["imputation_type"] == imputation_type]
        nmae = []

        # Return individual measurement scores or the average result
        if individual:
            for measurement in MEASUREMENTS:
                measurement_nmae = imputation_results[measurement].values
                nmae.append(np.min(measurement_nmae))
        else:
            nmae = np.min(imputation_results["average_normalised_mae"].values)

        nmae_data[imputation_type] = nmae

    return nmae_data


def gain_scores(m_type, m_percent, average=True):
    """"
    If the gain scores exist from the notebook then this will extract the relevant nmae scores. Either average
    for the given reference (missing type and percentage) or the nmae for all the MEASUREMENTS.
    """
    gain_findings = pd.read_csv("../../data/grid_searches/ground_truth/complete_artificial_nmae.csv")

    # Limiting findings to just those matching reference
    m_type_scores = gain_findings[gain_findings["reference"].str.contains("{}_{}".format(m_percent, m_type))]

    if average:
        avg_nmae = m_type_scores["average_normalised_mae"].values[0]
        return avg_nmae
    else:
        return m_type_scores[MEASUREMENTS].values[0]


def plot_individual_nmae(include_gain=False):
    """
    Plot the individual nmae results for each of the MEASUREMENTS. A plot is created for each type of missing data and
    stored in the visualisations/ground_truth folder.
    :param include_gain: Boolean to confirm whether the GAIN findings are available. If so they will be included.
    """
    for missing_percentage in MISSING_PERCENTAGES:
        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)
            # Reading the relevant grid search the missing type and extracting the individual nmae scores
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv")
            nmae_results = extract_nmae_values(data)

            if include_gain:
                m_type_gain_nmae = gain_scores(m_type, missing_percentage, average=False)
                nmae_results["gain"] = m_type_gain_nmae.flatten()

            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(MEASUREMENTS))
            width = 0.2

            ax.bar((x - 1.5 * width), nmae_results["mean"], width=width, label="mean")
            ax.bar((x - 0.5 * width), nmae_results["knn"], width=width, label="k-NN")
            ax.bar((x + 0.5 * width), nmae_results["mice"], width=width, label="MICE")

            if include_gain:
                ax.bar((x + 1.5 * width), nmae_results["gain"], width=width, label="GAIN")

            # Labels
            ax.set_xlabel("MEASUREMENTS")
            ax.set_ylabel("Normalised MAE")
            ax.set_title("Normalised MAE of Imputation Types when Data is {} at {}%".format(m_type, missing_percentage*100))
            ax.set_xticks(x)
            ax.set_xticklabels(MEASUREMENTS, rotation=45, ha="right")
            ax.legend(title="Imputation Type")

            plt.tight_layout()
            plt.savefig("../../visualisations/ground_truth/{}_{}_mae.png".format(m_type, missing_percentage))


def plot_average_nmae(include_gain=False):
    """
    Plot the average nmae results for each type of missing data, with the plot stored in the
    visualisations/ground_truth folder.
    :param include_gain: Boolean to confirm whether the GAIN findings are available. If so they will be included.
    """
    for missing_percentage in MISSING_PERCENTAGES:

        # Changing dictionary to match expectations
        if include_gain:
            nmae_data = {"mean": [], "knn": [], "mice": [], "gain": []}
        else:
            nmae_data = {"mean": [], "knn": [], "mice": []}

        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/normalised_" + file_reference + ".csv")
            nmae_results = extract_nmae_values(data, individual=False)

            nmae_data["mean"].append(nmae_results["knn"])
            nmae_data["knn"].append(nmae_results["knn"])
            nmae_data["mice"].append(nmae_results["mice"])

            # Gain results structured differently so extracted differently
            if include_gain:
                m_type_gain_nmae = gain_scores(m_type, missing_percentage, average=True)
                nmae_data["gain"].append(m_type_gain_nmae)

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(MISSING_TYPES))
        width = 0.2

        # Create bars for each imputation type with the relevant nmae data
        ax.bar((x - 1.5 * width), nmae_data["mean"], width=width, label="mean")
        ax.bar((x - 0.5 * width), nmae_data["knn"], width=width, label="k-NN")
        ax.bar((x + 0.5 * width), nmae_data["mice"], width=width, label="MICE")

        if include_gain:
            ax.bar((x + 1.5 * width), nmae_data["gain"], width=width, label="GAIN")

        # Add labels and title
        ax.set_xlabel("Missing Type")
        ax.set_ylabel("Normalised MAE")
        ax.set_title("Comparison of nmae between Imputation Types For Different Types of Missingness at {}%".format(missing_percentage*100))
        ax.set_xticks(x)
        ax.set_xticklabels(MISSING_TYPES, rotation=45, ha="right")
        ax.legend(title="Imputation Type", loc="lower right")

        # Display the plot
        plt.tight_layout()
        plt.savefig("../../visualisations/ground_truth/average_nmae_{}.png".format(missing_percentage))


def plot_nmae_with_missing_rates(include_gain=True):
    # Dictionary to contain the findings for all the missing types for each imputation type
    nmae_data = {method: {m_type: [] for m_type in MISSING_TYPES} for method in ["mean", "knn", "mice", "gain"]}

    # Getting the required nmae data
    for missing_percentage in MISSING_PERCENTAGES:
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

            if include_gain:
                # Get the average nmae for the "gain" imputation type
                m_type_gain_nmae = gain_scores(m_type, missing_percentage, average=True)
                nmae_data["gain"][m_type].append(m_type_gain_nmae)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    # Plotting the nmae data
    for i, m_type in enumerate(MISSING_TYPES):
        ax = axes[i]

        # Plotting data for current missing type with the nmae of each imputation type
        ax.plot(MISSING_PERCENTAGES, nmae_data["mean"][m_type], marker="x", label="Mean", linestyle="-")
        ax.plot(MISSING_PERCENTAGES, nmae_data["knn"][m_type], marker="x", label="k-NN", linestyle="-")
        ax.plot(MISSING_PERCENTAGES, nmae_data["mice"][m_type], marker="x", label="MICE", linestyle="-")

        if include_gain:
            ax.plot(MISSING_PERCENTAGES, nmae_data["gain"][m_type], marker="x", label="GAIN", linestyle="-")

        # Set the labels and title for each subplot
        ax.set_xlabel("Missing Percentage")
        ax.set_ylabel("Average nMAE")
        ax.set_title(m_type)

        ax.set_xticks(MISSING_PERCENTAGES)
        ax.set_xticklabels(STRING_PERCENTAGES)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Average Normalised MAE Across Imputation Methods and Missing Data Type", fontsize=16)

    plt.subplots_adjust(hspace=0.3, bottom=0.1, top=0.92)
    plt.savefig("../../visualisations/ground_truth/nmae_vs_missing_percentage_subplots.png")


grid_search_artificially_missing()
# Still tuning GAIN
plot_individual_nmae(include_gain=False)
plot_average_nmae(include_gain=False)
plot_nmae_with_missing_rates(include_gain=False)
