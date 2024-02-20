#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
base_credit = pd.read_csv("data/raw/home-credit-credit-risk-model-stability/csv_files/train/train_base.csv")
base_credit

### My goal is to make a simple dataset with 10 domain features and 1 target variable leveragin domain knowledge

# %%
# Selecting the credit appliations for 2020 - Less computational power, most recent and more positive labels percentage
base_credit_2020 = base_credit[base_credit['MONTH'] > 201912]
base_credit_2020

#%%
previous_application_df = pd.read_csv("data/raw/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_0.csv")
previous_application_df_2 = pd.read_csv("data/raw/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_1.csv")

# %%
previous_application_fetures = {
    "case_id" : "case_id",
    "employment_duration" : "employedfrom_700D",
    "outstanding_debt" : "outstandingdebt_522A",
    "previous_income" : "byoccupationinc_3656910L",
    "client profession" : "profession_152M",
    "loan_rejection_reason" : "rejectreason_755M"}
cols = previous_application_fetures.values()
case_ids = base_credit_2020[['case_id']]

#%%
#Creating a sample dataset with case ids for 2020 and the selected features
prev_app = previous_application_df[cols]
prev_app_2 = previous_application_df_2[cols]


# %%
bureau_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_1.parquet")
bureau_data

# %%
debit_card_data  = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_debitcard_1.parquet")
debit_card_data
# %%
deposit_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_deposit_1.parquet")
deposit_data

#%%
other_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_other_1.parquet")
other_data
# %%
static_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_0_0.parquet")
static_data
# %%
tax_registry = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_a_1.parquet")
tax_registry

# %%
person_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_person_1.parquet")
person_data
# %%


#From literature review, I have selected the following features to build a simple dataset
# 1. Age
# 2.Current employment 
# 3.Marital status - found in previous_application_df - column name: familystate_726L
# 4.Gender 
# 5.duration of loan 
# 6.number of people to support 
# 7.Loan amount
# 8.existing bank status 
# 9.Education 
# 10 exisitig debt amount

# My goal is to identiy which file holds each of the features and then merge them to create a simple dataset
