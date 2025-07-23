"""
Get all variable types from d_items which linksto chartevents,  category is labs, routine vital signs and param_type is numeric
"""

import pandas as pd


def identify_item_ids():
    all_variables = pd.read_csv("../../data/icu/d_items.csv")

    # Limiting to chart event variables that are vital signs, respiratory or laboratory and numeric (following apache)
    all_chart_events = all_variables[all_variables["linksto"] == "chartevents"]
    labs_and_vitals = all_chart_events[all_chart_events["category"].isin(["Routine Vital Signs", "Respiratory", "Labs"])]
    numeric_chart_events = labs_and_vitals[labs_and_vitals["param_type"] == "Numeric"]

    selected_chart_events = numeric_chart_events[["itemid", "label", "linksto"]]

    print("Identified {} chart events".format(len(selected_chart_events)))

    return selected_chart_events