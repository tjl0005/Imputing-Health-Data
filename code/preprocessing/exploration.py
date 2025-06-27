"""
Exploring the limited and processed datasets. Results are printed to the console.
"""
import pandas as pd


pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

def check_duplicates(data):
    """
    Given a dataset check for any duplicate rows. If they exist a count will be printed and the identified rows returned
    :param data: Data to check
    :return: Duplicated rows
    """
    duplicates = data.duplicated()
    duplicate_count = duplicates.sum()

    print("{} duplicates\n".format(duplicate_count))

    return duplicates


def check_missing_data(data):
    """
    Given a dataset check how many values are missing for each of the columns. Function will output counts for columns
    missing data.
    :param data:
    """
    missing_df = data.isnull().sum()
    missing_df = missing_df[missing_df > 0]

    if missing_df.empty:
        print("No missing data")
    else:
        print("Missing data for\n{}\n".format(missing_df))


def death_statistics(data):
    """
    Display the number of patients who died and survived in ICU as well as percentage.
    :param data:
    """
    death_count = (data["outcome"] == "DIED").sum()
    alive_count = (data["outcome"] != "DIED").sum()

    rate = (death_count / alive_count) * 100

    print("{:.2f}% death rate with {} dying and {} surviving\n".format(rate, death_count, alive_count))


def categorical_statistics(data):
    """
    Given a dataset with categorical data return the number of instances for each variable
    :param data: data containing categorical variables
    :return: dictionary containing number of value instances
    """
    cat_cols = data.select_dtypes(include=["object", "category"]).columns

    return {col: data[col].value_counts() for col in cat_cols}


def statistics(data):
    print(data.describe())
    print(categorical_statistics(data))

    check_duplicates(data)
    death_statistics(data)
    check_missing_data(data)
