#This script is used to idenitfy features from the home credit dataset and create a simple dataset for model building ans stores intermediate output in intermedoate foldeer

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### My goal is to make a simple dataset with 10 domain features and 1 target variable leveragin domain knowledge

#From literature review, I have selected the following features to build a simple dataset
# 1.Age - found in static_data_2 - column name: dateofbirth_337D, dateofbirth_342D
# 2.Main income amount -  found in static data - column name: maininc_215A & found in person_data - column name: mainoccupationinc_384A & found in previous_application_df - column name: mainoccupationinc_437A
# 3.Marital status - found in previous_application_df - column name: familystate_726L & found in person_data - column name: familystate_447L
# 4.Gender - found in person_data - column: gender_992L & found in person_data - column: sex_738L
# 5.Duration of loan - 
# 6.number of children - found in previous_application_df - column name:childnum_21L  & person_data - column name: childnum_185L
# 7. 
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
#Put it all into a class but refine the functions to be more modular

def get_inidivual_feature(df:pd.DataFrame, col_list:list, recency:str) -> pd.DataFrame:
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep=recency)
    return df_uniq

def merge_features_into_df(dfs:list, on_col:str, join_type:str) -> pd.DataFrame:
    df_combined = pd.merge(dfs[0], dfs[1], on=on_col, how=join_type)
    return df_combined

def create_consolicated_df(df:pd.DataFrame, new_col:str) -> pd.DataFrame:
    col_list = list(df.columns)
    df[new_col] = df[col_list[1]].fillna('').astype(str) + df[col_list[2]].fillna('').astype(str)
    df[new_col] = df[new_col].replace({"NaN" : ""})
    return df[[col_list[0], new_col]]

def handle_feature_datatype(df:pd.DataFrame, col_data:dict) -> pd.DataFrame:
    for col_name, datatype in col_data.items():
        if datatype == "string":
            df.loc[:, col_name] = df[col_name].astype(str)
        elif datatype == "integer":
            df.loc[:, col_name] = df[col_name].astype(int)
        elif datatype == "date":
            df.loc[:, col_name] = pd.to_datetime(df[col_name], 
                                format="%Y-%m-%d", errors='coerce').dt.date
        else:
            df.loc[:, col_name] = df[col_name].astype(float)
    return df
    
#%%
# Tested for the dob feature
dob_df1 = get_feature(static_data_2, ["case_id", "dateofbirth_337D"], "last")
dob_df2 = get_feature(static_data_2, ["case_id", "dateofbirth_342D"], "last")
dob_comb = merge_dfs([age_df1, age_df2], "case_id", "outer")
dob_df = create_consolicated_df(age_comb, "dateofbirth")
dob_df = handle_feature_datatype(age_df, {"case_id":"integer", "dateofbirth":"date"})

# %%
# Sorting gender from personal dataframe - accouting for the depth of the data (i.e num_group1: this is historical data)
# For the personal dataframe I am keeping the first instance because the gender is static
def get_gender(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    df = df[col_list]
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="first")
    return df_uniq

gender_df = get_gender(person_data, ["case_id", "sex_738L"])

#%%
# getting it from 2 dataframes; personal and previous_application_df

#Looking within the previous application: I will use the childnum_21L column to select the first value and if no first value,
# i will look at other instances and fill with the value found if not then null

def get_children_num(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    
    #select the case_id and childnum colum
    df = df[col_list]
    #drop rows with one null value
    df = df.dropna(subset=[col_list[1]])
    #drop duplicates and keep the most recent value
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="last")
    return df_uniq

child_df1 = get_children_num(previous_application_df, ["case_id", "childnum_21L"])
child_df2 = get_children_num(person_data, ["case_id", "childnum_185L"])

# %%
# Deal with getting the marital status from the previous_application_df

def get_familystate(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="last")
    return df_uniq
    
famstate_df1 = get_familystate(previous_application_df, ["case_id", "familystate_726L"])
famstate_df2 = get_familystate(person_data, ["case_id", "familystate_447L"])

# %%

def get_recent_income(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="last")
    return df_uniq

income_df1 = get_recent_income(static_data, ["case_id", "maininc_215A"])
income_df2 = get_recent_income(person_data, ["case_id", "mainoccupationinc_384A"])
income_df3 = get_recent_income(previous_application_df, ["case_id", "mainoccupationinc_437A"])

# %%
def get_existing_debt(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="last")
    return df_uniq

debt_df = get_existing_debt(previous_application_df, ["case_id", "outstandingdebt_522A"])

# %%
def get_credit_status(df:pd.DataFrame, col_list:list) -> pd.DataFrame:
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep="last")
    return df_uniq

credit_status_df = get_credit_status(previous_application_df, ["case_id", "credacc_status_367L"])

# %%
#Now i need to start merging or concatenating the dataframes to create the simple dataset

#Function to merge dataframes from the previous steps
def merge_dfs(dfs:list, on_col:str, join_type:str) -> pd.DataFrame:
    df_combined = pd.merge(dfs[0], dfs[1], on=on_col, how=join_type)
    return df_combined

famstate_df = merge_dfs([famstate_df1, famstate_df2], "case_id", "outer")

# %%
#Function to create a new column for  the merged dataframes because they hold the same information in 2 different columns
def create_consolicated_df(df:pd.DataFrame, new_col:str) -> pd.DataFrame:
    col_list = list(df.columns)
    df[new_col] = df[col_list[1]].fillna('').astype(str) + df[col_list[2]].fillna('').astype(str)
    df[new_col] = df[new_col].replace({"NaN" : ""})
    return df[[col_list[0], new_col]]

# %%
def handle_column_datatype(df:pd.DataFrame, col_data:dict) -> pd.DataFrame:
    
    for col_name, datatype in col_data.items():
        if datatype == "string":
            df.loc[:, col_name] = df[col_name].astype(str)
        elif datatype == "integer":
            df.loc[:, col_name] = df[col_name].astype(int)
        else:
            df.loc[:, col_name] = df[col_name].astype(float)
    return df
