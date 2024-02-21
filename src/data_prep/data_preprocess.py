#This script is used to idenitfy features from the home credit dataset and create a simple dataset for model building

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### My goal is to make a simple dataset with 10 domain features and 1 target variable leveragin domain knowledge

#From literature review, I have selected the following features to build a simple dataset
# 1.Age - found in static_data_2 - column name: dateofbirth_337D, dateofbirth_342D
# 2.Main income amount -  found in static data - column name: maininc_215A & found in person_data - column name: mainoccupationinc_384A & found in previous_application_df - column name: mainoccupationinc_437A
# 3.Marital status - found in previous_application_df - column name: familystate_726L
# 4.Gender - found in person_data - column: gender_992L & found in person_data - column: sex_738L
# 5.Duration of loan - 
# 6.number of children - found in previous_application_df - column name:childnum_21L  & person_data - column name: childnum_185L
# 7.Loan amount on previous application - found in previous_application_df - column name:credamount_370L
# 8.existing previous credit status if any - found in previous_application_df - column name: credacc_status_367L 
# 9.Existing debt amount - found in previous_application_df - column name: outstandingdebt_522A

#other adhoc column names 
    # birth_259D,Date of birth of the person.
    # birthdate_574D,Client's date of birth (credit bureau data).
    # birthdate_87D,Birth date of the person.

#required dataframes = train_base, static_data_2, static_data, previous_application_df, person_data

#%%
credit_base_df = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet")
previous_application_df = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_1.parquet")
static_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_0_0.parquet")
static_data_2 = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_cb_0.parquet")
person_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_person_1.parquet")

#%%
# getting the columns from the dataframes
def get_columns(df:pd.DataFrame, col_list:list):
    return df[col_list]

prev_app = get_columns(previous_application_df, ["case_id", "credamount_590A", "outstandingdebt_522A","mainoccupationinc_437A", "familystate_726L", "childnum_21L","credacc_status_367L"])
personal = get_columns(person_data, ["case_id", "mainoccupationinc_384A", "childnum_185L", "gender_992L", "sex_738L"])
static_1 = get_columns(static_data, ["case_id", "maininc_215A"])
static_2 = get_columns(static_data_2, ["case_id", "dateofbirth_337D", "dateofbirth_342D"])

#%%
# Now i need to start prepeocessing each dataframe with the columns I have selected - this is to ensure that the data is clean and ready to be merged into the credit_base_df to create the simple dataset


#date of bith from static_2
def fuse_date_of_birth(df:pd.DataFrame, col_list:list):
    df['dateofbirth'] = df[col_list[0]].fillna("") + df[col_list[1]].fillna("")
    df['dateofbirth'] = df['dateofbirth'].replace("", None)
    return df

static_dob = static_2.copy()
static_dob = fuse_date_of_birth(static_dob, ["dateofbirth_337D", "dateofbirth_342D"])

# %%
