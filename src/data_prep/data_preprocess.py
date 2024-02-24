import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def get_individual_feature(df:pd.DataFrame, col_list:list, recency:str) -> pd.DataFrame:
        df = df[col_list]
        df = df.dropna(subset=[col_list[1]])
        df_uniq = df.drop_duplicates(subset=col_list[0], keep=recency)
        return df_uniq

    def merge_features_into_df(dfs:list, on_col:str, join_type:str) -> pd.DataFrame:
        df_combined = pd.merge(dfs[0], dfs[1], on=on_col, how=join_type)
        return df_combined

    def create_consolicated_df(df:pd.DataFrame, new_colname:str) -> pd.DataFrame:
        col_list = list(df.columns)
        if len(col_list) > 2:
            if all(pd.api.types.is_numeric_dtype(df[col]) for col in col_list[1:]):
                df[new_colname] = df.iloc[:, 1:].max(axis=1)
            else:
                df[new_colname] = df.iloc[:, 1:].fillna('').astype(str).agg(''.join, axis=1)
                df[new_colname] = df[new_colname].replace({"NaN" : ""})
        else:
            df[new_colname] = df[col_list[1]]
        return df[[col_list[0], new_colname]]

    def handle_features_datatype(df:pd.DataFrame, col_datatype:dict) -> pd.DataFrame:
        for col_name, datatype in col_data.items():
            if datatype == "string":
                df.loc[:, col_name] = df[col_name].astype(str)
            elif datatype == "date":
                df.loc[:, col_name] = pd.to_datetime(df[col_name],
                                format="%Y-%m-%d", errors='coerce').dt.date
            elif datatype == "integer":
                df.loc[:, col_name] = df[col_name].astype(int)
            else:
                df.loc[:, col_name] = df[col_name].astype(float)
        return df
    

# # Example usage
# credit_base_df = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet")
# previous_application_df = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_1.parquet")
# static_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_0_0.parquet")
# static_data_2 = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_cb_0.parquet")
# person_data = pd.read_parquet("data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_person_1.parquet")