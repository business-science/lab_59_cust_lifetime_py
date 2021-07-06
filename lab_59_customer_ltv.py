# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE ----
# CUSTOMER LIFETIME VALUE WITH MACHINE LEARNING ----
# **** ----

# CONDA ENV USED: lab_59_customer_ltv_py

import pandas as pd
import numpy as np

import plotnine as pn
from plotnine.ggplot import ggplot


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



