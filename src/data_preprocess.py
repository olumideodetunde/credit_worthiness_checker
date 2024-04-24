''' 
This script holds functions to preprocess the data for creating a simple dataset
'''
import pandas as pd

def read_data(file_path:str) -> pd.DataFrame:
    '''
    This function is to read parent data from a file and return a dataframe
    '''
    df = pd.read_parquet(file_path)
    return df

def get_individual_feature(df:pd.DataFrame, col_list:list, recency:str) -> pd.DataFrame:
    '''
    This function is to get the individual feature from a dataframe with many features
    '''
    df = df[col_list]
    df = df.dropna(subset=[col_list[1]])
    df_uniq = df.drop_duplicates(subset=col_list[0], keep=recency)
    return df_uniq

def merge_features_into_df(dfs:list, on_col:str, join_type:str) -> pd.DataFrame:
    '''
    This function is to merge the individual features into a single dataframe
    '''
    df_combined = pd.merge(dfs[0], dfs[1], on=on_col, how=join_type)
    return df_combined

def create_consolicated_df(df:pd.DataFrame, new_colname:str) -> pd.DataFrame:
    '''
    This function fuses the individual features into a single column if the feature is not
    in more than one column of the dataframe
    '''
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
    '''
    This function is to convert the datatype of each column in the dataframe, it 
    takes a dictionary of column names and desired datatypes 
    (string, date, integer, float).
    '''
    for col_name, datatype in col_datatype.items():
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


#End of script