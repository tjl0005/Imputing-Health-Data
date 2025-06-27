"""
Using the data from feature selection with no missing values, which have been modified to be artificially missing data
that has been imputed, this file will evaluate the imputations using the ground truth through RMSE. This is not 
currently implemented in main.py.
# """
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from code.constants import CATEGORIES, RESAMPLED_DIR, SAMPLED_GROUND_TRUTH_FILE, KNN_PARAMS, MICE_PARAMS, \
    GRID_SEARCH_OUTPUT, MISSING_TYPES, ARTIFICIAL_MISSING_DATA_DIR, MICE_FOREST_PARAMS
from code.imputation.ml import knn_impute, mice_impute, mean_impute, mice_forest_impute

ground_truth_data = pd.read_csv("../../data/missing/resampled/measurements_0_downsample.csv")

# Need to resolve this, probably get rid of constants as its more confusing
CATEGORIES = [category for category in CATEGORIES if category != "anchor_age"]

# note: Consider keeping more than necessary measurements -> find what is correlated with them


def test_correlations():
    linear_correlations = ground_truth_data[CATEGORIES].corr(method="pearson")
    non_linear_correlations = ground_truth_data[CATEGORIES].corr(method="spearman")

    plt.figure(figsize=(8, 6))  # Set the figure size for the heatmap
    sns.heatmap(linear_correlations, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

    # Display the heatmap
    plt.title("Linear Correlation Heatmap")
    plt.show()

    plt.figure(figsize=(8, 6))  # Set the figure size for the heatmap
    sns.heatmap(non_linear_correlations, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)

    # Display the heatmap
    plt.title("Non-Linear Correlation Heatmap")
    plt.show()


def evaluate_ground_truth(imputed_data, mask):
    """
    Given data imputed from the artificially missing data evaluate the imputed values using normalised RMSE
    :param imputed_data: Artificially missing data that has been imputed
    :param mask: Mask representing where missing data is in the passed dataset
    :return: The NRMSE for each of the feature measurements and the average RMSE
    """
    nrmse_values = []
    nrmse_results = {}

    # Finding the NRMSE for each of the measurements
    for category in CATEGORIES:
        # Limiting data to those that were missing for comparison
        missing_ground_truth = ground_truth_data[category][mask[category]]
        imputed_values = imputed_data[category][mask[category]]

        # RMSE for the current category
        category_rmse = np.sqrt(mean_squared_error(missing_ground_truth, imputed_values))

        # Normalising RMSE with min-max
        measurement_range = ground_truth_data[category].max() - ground_truth_data[category].min()
        category_n_rmse = category_rmse / measurement_range

        nrmse_results[category] = category_n_rmse
        nrmse_values.append(category_n_rmse)

    # Adding the average NRMSE across the columns
    nrmse_results["average_nrmse"] = np.mean(nrmse_values)

    return nrmse_results


def grid_search_optimisation(missing_data, imputation_type="knn", file_reference="mcar"):
    """
    Perform a grid search for the provided imputation type and data to optimise the imputation approach
    :param missing_data: Artificially missing data from the ground truth
    :param imputation_type: mean, knn or mice with the hyperparameters being pre-defined
    :param file_reference: Specification of the type of missing data to identify results
    """
    results = []
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
        # Doing mean imputation so skipping grid search and running once
        param_grid = {"blank": [0]}

    for key in param_grid:
        for value in param_grid[key]:
            # Required to avoid overwriting reference
            data = missing_data.copy()
            print("Testing {} with {} at {}".format(imputation_type, key, value))

            search_reference = {"imputation_type": imputation_type, "hyperparameter": value}

            if imputation_type == "knn":
                imputed_data = knn_impute(data, k=value)
            elif imputation_type == "mice":
                imputed_data = mice_impute(data, max_iter=value)
            elif imputation_type == "mice_forest":
                imputed_data = mice_forest_impute(data, max_iter=value)
            else:
                imputed_data = mean_impute(data)

            nrmse_results = evaluate_ground_truth(imputed_data, mask)

            search_reference.update(nrmse_results)
            results.append(search_reference)

    # Preparing to save results
    save_dir = GRID_SEARCH_OUTPUT + "/ground_truth/" + file_reference + ".csv"
    results_df = pd.DataFrame(results)

    # Saving results to a new file if no previous searches or appending to file if matching previous search
    if not os.path.exists(save_dir):
        results_df.to_csv(save_dir, index=False)
    else:
        results_df.to_csv(save_dir, mode="a", header=False, index=False)


def grid_search_artificially_missing():
    """
    Wrapper to go through all artifically missing data and perform grid searches for them on k-NN and MICE
    """
    for missing_percentage in [0.2, 0.5, 0.7]:
        for m_type in MISSING_TYPES:
            print("Performing grid search for {} with {}% missing".format(m_type, missing_percentage * 100))
            file_reference = "{}_{}".format(m_type, missing_percentage)

            missing_dir = "{}/measurements_{}_{}.csv".format(ARTIFICIAL_MISSING_DATA_DIR, missing_percentage, m_type)
            missing_data = pd.read_csv(missing_dir)

            # Optimising each of the imputation types for each type of missing data
            grid_search_optimisation(missing_data, imputation_type="mean", file_reference=file_reference)
            grid_search_optimisation(missing_data, imputation_type="knn", file_reference=file_reference)
            grid_search_optimisation(missing_data, imputation_type="mice", file_reference=file_reference)
            grid_search_optimisation(missing_data, imputation_type="mice_forest", file_reference=file_reference)


def extract_nrmse_values(data, individual=True, imputation_types=["mean", "knn", "mice"]):
    """
    After a completed grid search extract either the best individual or average NRMSE scores for each of the imputation
    types
    :param data: Grid search results
    :param individual: Boolean to specify whether to return individual NNRMSE results or the average
    :param imputation_types: The types of imputation tested in the grid search
    :return: The best individual or average NRMSE results
    """
    # Store results for each of the imputatoin types
    nrmse_data = {"mean": [], "knn": [], "mice": []}

    for imputation_type in imputation_types:
        # Selecting the specific results for the current imputation type
        imputation_results = data[data["imputation_type"] == imputation_type]
        nrmse = []

        # Return indvidual category scores or the average result
        if individual:
            for category in CATEGORIES:
                category_nrmse = imputation_results[category].values
                nrmse.append(np.min(category_nrmse))
        else:
            nrmse = np.min(imputation_results["average_nrmse"].values)

        nrmse_data[imputation_type] = nrmse

    return nrmse_data


def plot_individual_nrmse():
    """
    Plot the individual NRMSE results for each of the categories. A plot is created for each type of missing data and
    stored in the visualisations/ground_truth folder.
    :return:
    """
    for missing_percentage in [0.2, 0.5, 0.7]:
        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)
            # Reading the relevant grid search the missing type and extracting the individual NRMSE scores
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/" + file_reference + ".csv")
            nrmse_results = extract_nrmse_values(data)

            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(CATEGORIES))
            width = 0.2

            # Bars for each of the imputation types
            ax.bar((x - width), nrmse_results["mean"], width=width, label="mean")
            ax.bar(x, nrmse_results["knn"], width=width, label="k-NN")
            ax.bar((x + width), nrmse_results["mice"], width=width, label="MICE")

            # Labels
            ax.set_xlabel("Categories")
            ax.set_ylabel("Normalized RMSE")
            ax.set_title("NRMSE of Imputation Types when Data is {} at {}%".format(m_type, missing_percentage*100))
            ax.set_xticks(x)
            ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
            ax.legend(title="Imputation Type")

            plt.tight_layout()
            plt.savefig("../../visualisations/ground_truth/{}_{}_rmse.png".format(m_type, missing_percentage))


def plot_average_nrmse():
    """
    Plot the average NRMSE results for each type of missing data, with the plot stored in the
    visualisations/ground_truth folder.
    """
    for missing_percentage in [0.2, 0.5, 0.7]:
        nrmse_data = {"mean": [], "knn": [], "mice": []}

        for m_type in MISSING_TYPES:
            file_reference = "{}_{}".format(m_type, missing_percentage)
            data = pd.read_csv(GRID_SEARCH_OUTPUT + "/ground_truth/" + file_reference + ".csv")
            nrmse_results = extract_nrmse_values(data, individual=False)

            nrmse_data["mean"].append(nrmse_results["knn"])
            nrmse_data["knn"].append(nrmse_results["knn"])
            nrmse_data["mice"].append(nrmse_results["mice"])

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(MISSING_TYPES))
        width = 0.2

        # Create bars for both k-NN and MICE
        ax.bar((x - width), nrmse_data["mean"], width=width, label="mean")
        ax.bar(x, nrmse_data["knn"], width=width, label="k-NN")
        ax.bar((x + width), nrmse_data["mice"], width=width, label="MICE")

        # Add labels and title
        ax.set_xlabel("Missing Type")
        ax.set_ylabel("Normalized RMSE")
        ax.set_title("Comparison of NRMSE between k-NN and MICE For Different Types of Missingness at {}%".format(missing_percentage*100))
        ax.set_xticks(x)
        ax.set_xticklabels(MISSING_TYPES, rotation=45, ha="right")
        ax.legend(title="Imputation Type", loc="lower right")

        # Display the plot
        plt.tight_layout()
        plt.savefig("../../visualisations/ground_truth/average_nrmse_{}.png".format(missing_percentage))


grid_search_artificially_missing()
plot_individual_nrmse()
plot_average_nrmse()
# test_correlations()
