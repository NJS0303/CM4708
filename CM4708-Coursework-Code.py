# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:55:18 2023

@author: nsilvest
"""

# --- imports & settings ---

import pandas as pd
import seaborn as sns

from sktree import ExtendedIsolationForest

# --- load from data source ---

raw_data = pd.read_csv("Sample Export.csv",
                       thousands=",",
                       dtype={"Personal Reference:Timesheet":"object"})

# set up list of claim templates not relevant to the application i.e.,
# T&S Entry Form [..] is not used by employees in the claiming process
templates_to_exclude = ["T&S Entry Form v1.1"]

# apply initial set of filters to the raw data
df = raw_data[
    (raw_data["Authorisation Status:Timesheet"] == "Authorised") &
    (~raw_data["Fast Input Design:Timesheet"].isin(templates_to_exclude)) &
    (raw_data["Miles Claimed:Timesheet"].notnull()) &
    (raw_data["Submitted Date:Timesheet"].notnull())]

# --- create preprocessing functions ---

# start of function argument/parameter lists
columns_to_retain = [
    "Timesheet ID",
    "Personal Reference:Timesheet",
    "Start Date:Timesheet",
    "Journey Date:Timesheet",
    "Miles Claimed:Timesheet",
    "V_AllMileage_A",
    "V_CommuteMileage_B",
    "Submitted Date:Timesheet"]

columns_to_datetime = [
    "Start Date:Timesheet",
    "Journey Date:Timesheet",
    "Submitted Date:Timesheet"]

columns_to_groupby = [
    "Timesheet ID",
    "Personal Reference:Timesheet"]
# end of function argument/parameter lists

# start of function argument/parameter dicts
columns_to_rename = {
    "Timesheet ID": "claim_id",
    "Personal Reference:Timesheet": "employee_id"}
# end of function argument/parameter dicts

def apply_initial_prepro(filtered_df,
                         col_retain_list,
                         col_datetime_list):
    """
    Placeholder
    
    """
    # restrict columns to those specified
    df = filtered_df[columns_to_retain]
    
    # convert specified columns to datetime format
    df[columns_to_datetime] = (df[columns_to_datetime].
                               apply(pd.to_datetime,
                                     format="%d/%m/%Y",
                                     errors="coerce"))
    
    # limit the scope of data to claims submitted in the last 13 months
    df = df[(df["Submitted Date:Timesheet"] <= pd.to_datetime("now")) &
            (df["Submitted Date:Timesheet"] >= pd.to_datetime("now") -
             pd.DateOffset(months=13))]
    
    return df

def apply_group_and_agg(preprocessed_df,
                        cols_groupby_list):
    """
    Placeholder
    
    """
    # group by timesheet & employee id then calculate sum of mileage elements
    df = (preprocessed_df.groupby(columns_to_groupby, as_index=False)
          .agg(
              element_count=("Timesheet ID", "count"),
              paid_miles=("Miles Claimed:Timesheet", "sum"),
              total_miles=("V_AllMileage_A", "sum"),
              commute_miles=("V_CommuteMileage_B", "sum")
              )
          .rename(columns=columns_to_rename))
    
    return df

def apply_filt_to_group(df_grouped):
    """
    Placeholder
    
    """
    df = df_grouped[
        (df_grouped["paid_miles"] >= 0) &
        (df_grouped["total_miles"] < 10000) &
        (df_grouped["commute_miles"] < 10000)]
    
    return df    

# --- apply preprocessing functions ---

df = (df.pipe(apply_initial_prepro, columns_to_retain, columns_to_datetime)
      .pipe(apply_group_and_agg, columns_to_groupby)
      .pipe(apply_filt_to_group))

# -- instantiate and implement extended isolation forest algo ---

anomaly_inputs = ["element_count",
                  "paid_miles",
                  "total_miles",
                  "commute_miles"]

model_eif = ExtendedIsolationForest(contamination=0.01,
                                    random_state=0)

model_eif.fit(df[anomaly_inputs])

df["anomaly_score"] = model_eif.decision_function(df[anomaly_inputs])
df["anomaly"] = model_eif.predict(df[anomaly_inputs])

# --- generate output report & anomaly distribution visualisation ---

# generate pairplot
pp = sns.pairplot(df,
                  vars=anomaly_inputs,
                  hue="anomaly",
                  palette=["red", "green"])

sample_count = df["anomaly"].value_counts().values[0]
anomaly_count = df["anomaly"].value_counts().values[1]

pairplot_title = (
    f'Distribution of anomaly labels ({anomaly_count:,} anomalous from '
    f'{sample_count:,} total samples)'
    )

pp.fig.suptitle(t=pairplot_title, y=1.05, fontsize=16)

# output report for evaluation
df[df["anomaly"] == -1].to_csv("Sample Export Labelled.csv", index=False)
