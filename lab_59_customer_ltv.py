# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE ----
# CUSTOMER LIFETIME VALUE WITH MACHINE LEARNING ----
# **** ----

# CONDA ENV USED: lab_59_customer_ltv_py

# LIBRARIES ----
import pandas as pd
import numpy as np

import plotnine as pn

pn.options.dpi = 300


# 1.0 DATA PREPARATION ----

cdnow_raw_df = pd.read_csv(
    "data/CDNOW_master.txt", 
    sep   = "\s+",
    names = ["customer_id", "date", "quantity", "price"]
)

cdnow_raw_df.info()

cdnow_df = cdnow_raw_df \
    .assign(
        date = lambda x: x['date'].astype(str)
    ) \
    .assign(
        date = lambda x: pd.to_datetime(x['date'])
    ) \
    .dropna()

# 2.0 COHORT ANALYSIS ----
# - Only the customers that have joined at the specific business day

# Get Range of Initial Purchases ----
cdnow_first_purchase_tbl = cdnow_df \
    .sort_values(['customer_id', 'date']) \
    .groupby('customer_id') \
    .first()

cdnow_first_purchase_tbl

cdnow_first_purchase_tbl['date'].min()

cdnow_first_purchase_tbl['date'].max()

# Visualize: All purchases within cohort

cdnow_df \
    .reset_index() \
    .set_index('date') \
    [['price']] \
    .resample(
        rule = "MS"
    ) \
    .sum() \
    .plot()

# Visualize: Individual Customer Purchases

ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]

cdnow_cust_id_subset_df = cdnow_df \
    [cdnow_df['customer_id'].isin(ids_selected)] \
    .groupby(['customer_id', 'date']) \
    .sum() \
    .reset_index()

pn.ggplot(
    pn.aes('date', 'price', group = 'customer_id'),
    data = cdnow_cust_id_subset_df
) \
    + pn.geom_line() \
    + pn.geom_point() \
    + pn.facet_wrap('customer_id') \
    + pn.scale_x_date(
        date_breaks = "1 year",
        date_labels = "%Y"
    )


# 3.0 MACHINE LEARNING ----
#  Frame the problem:
#  - What will the customers spend in the next 90-Days? (Regression)
#  - What is the probability of a customer to make a purchase in next 90-days? (Classification)


# 3.1 SPLITTING (2-Stages) ----

# Stage 1: Random Splitting by Customer ID ----

customer_ids = pd.Series(
    cdnow_df['customer_id'].unique()
)

ids_train = customer_ids \
    .sample(frac=0.8, random_state=123) \
    .sort_values()
ids_train

split_1_train_df = cdnow_df \
    [cdnow_df['customer_id'].isin(ids_train)]

split_1_test_df = cdnow_df \
    [~ cdnow_df['customer_id'].isin(ids_train)]

# Stage 2: Time Splitting ----

n_days   = 90
max_date = split_1_train_df['date'].max() 
cutoff   = max_date - pd.to_timedelta(n_days, unit = "d")

split_2_train_in_df = split_1_train_df \
    [split_1_train_df['date'] <= cutoff]

split_2_train_out_df = split_1_train_df \
    [split_1_train_df['date'] > cutoff]

split_2_test_in_df = split_1_test_df \
    [split_1_test_df['date'] <= cutoff]

split_2_test_out_df = split_1_test_df \
    [split_1_test_df['date'] > cutoff]

# 3.2 FEATURE ENGINEERING (RFM) ----
#   - Most challenging part
#   - 2-Stage Process
#   - Need to frame the problem
#   - Need to think about what features to include

# Make Targets from out data ----

targets_train_df = split_2_train_out_df \
    .drop('quantity', axis=1) \
    .groupby('customer_id') \
    .sum() \
    .rename({'price': 'spend_90_total'}, axis = 1) \
    .assign(spend_90_flag = 1)

targets_test_df = split_2_test_out_df \
    .drop('quantity', axis=1) \
    .groupby('customer_id') \
    .sum() \
    .rename({'price': 'spend_90_total'}, axis = 1) \
    .assign(spend_90_flag = 1)

# Make Recency (Date) Features from in data ----

max_date = split_2_train_in_df['date'].max()

recency_features_train_df = split_2_train_in_df \
    [['customer_id', 'date']] \
    .groupby('customer_id') \
    .apply(
        lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")
    )

recency_features_test_df = split_2_test_in_df \
    [['customer_id', 'date']] \
    .groupby('customer_id') \
    .apply(
        lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, "day")
    )

# Make Frequency (Count) Features from in data ----

frequency_features_train_df = split_2_train_in_df \
    [['customer_id', 'date']] \
    .groupby('customer_id') \
    .count() \
    .set_axis(['frequency'], axis=1)

frequency_features_test_df = split_2_test_in_df \
    [['customer_id', 'date']] \
    .groupby('customer_id') \
    .count() \
    .set_axis(['frequency'], axis=1)

# Make Price (Monetary) Features from in data ----

price_features_train_df = split_2_train_in_df \
    .groupby('customer_id') \
    .aggregate(
        {
            'price': ["sum", "mean"]
        }
    ) \
    .set_axis(['price_sum', 'price_mean'], axis = 1)

price_features_test_df = split_2_test_in_df \
    .groupby('customer_id') \
    .aggregate(
        {
            'price': ["sum", "mean"]
        }
    ) \
    .set_axis(['price_sum', 'price_mean'], axis = 1)


