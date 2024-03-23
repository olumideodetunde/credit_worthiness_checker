#%%
'''This module contains the FeatureEngineering class for engineering the data'''
from typing import Union
import pandas as pd
import numpy as np
from pickle import dump, load
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import logger
class FeatureEngineering:
    '''This class contains methods for feature engineering the data'''
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.raw_df = None
        self.cat_df = None
        self.df = None
        self.transformed_df = None

    def _convert_to_date(self, date_cols:list) -> pd.DataFrame:
        '''This method converts the date columns to datetime format'''
        self.df[date_cols] = self.df[date_cols].apply(
            pd.to_datetime, format="%Y-%m-%d", errors="coerce")
        return self.df

    def _fill_na(self, col:str, value:Union[str, int]) -> pd.DataFrame:
        '''This method fills the missing values in the column with the specified value'''
        self.df[col] = self.df[col].fillna(value)
        return self.df
    
    def _dump_object(self, obj:object, output_path:str) -> None:
        '''This method dumps the object to the specified output path'''
        with open(output_path, 'wb') as f:
            dump(obj, f)
    
    def _load_object(self, input_path:str) -> object:
        '''This method loads the object from the specified input path'''
        with open(input_path, 'rb') as f:
            obj = load(f)
        return obj

    def load_data(self) -> pd.DataFrame:
        '''This method loads the data from the parquet filepath'''
        self.raw_df = pd.read_parquet(self.file_path)
        self.df = self.raw_df.copy().reset_index(drop=True)
        return self.df

    def drop_columns(self, cols:list) -> pd.DataFrame:
        '''This method drops the specified columns from the dataframe'''
        self.df = self.df.drop(columns=cols)
        return self.df

    def drop_rows(self, col_subset:list) -> pd.DataFrame:
        '''This method drops the rows with missing values in the 
        specified columns from the dataframe'''
        self.df = self.df.dropna(subset=col_subset).reset_index(drop=True)
        return self.df

    def bin_numerical_feature(self, cols:str, bins:list, labels:list) -> pd.DataFrame:
        '''This method bins the numerical columns into specified
        bins and labels them accordingly'''
        self.df[str(cols) + '_binned'] = pd.cut(self.df[cols],
                                            bins=bins, labels=labels, include_lowest=True)
        return self.df

    def engineer_age(self, date_cols:list, feature_name:str) -> pd.DataFrame:
        '''This method calculates the age feature in the dataset'''
        self.df = self._convert_to_date(date_cols)
        self.df[feature_name] = self.df[date_cols[0]].dt.year - self.df[date_cols[1]].dt.year
        return self.df

    def engineer_time_of_year(self, date_col:str) -> pd.DataFrame:
        '''This method assigns the time of the year based on a date column'''
        self.df['time_of_year'] = self.df[date_col].dt.month.apply(lambda x:
            'winter' if x in [12, 1, 2] else 'spring' 
            if x in [3, 4, 5] else 'summer'
            if x in [6, 7, 8] else 'autumn')
        return self.df

    def engineer_family_size(self, col:str, value:str, new_col_name:str, bins:list, labels:str) -> pd.DataFrame:
        '''This method calculates the family size based on the number of children'''
        self._fill_na(col=col, value=value)
        self.bin_numerical_feature(col, bins=bins, labels=labels)
        self.df.rename(columns={col+'_binned': new_col_name}, inplace=True)
        return self.df

    def engineer_marital_status(self, col:str, value:str, grouping_dict:dict) -> pd.DataFrame:
        '''This method aggregates the marital status into broader categories '''
        self._fill_na(col=col, value=value)
        self.df[col] = self.df[col].str.title()
        self.df[col+'_new'] = self.df[col]
        for group, values in grouping_dict.items():
            self.df.loc[self.df[col].isin(values), col+'_new'] = group
        return self.df

    def one_hot_encode(self, cat_col_list:list) -> pd.DataFrame:
        '''This method transforms the categorical columns to be ml ready'''
        cat = self.df[cat_col_list]
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
        cat_encoded = encoder.fit_transform(cat)
        cat_columns = []
        for i, col in enumerate(cat.columns):
            for cat in encoder.categories_[i]:
                cat_columns.append(f"{col}_{cat}")
        self.cat_df = pd.DataFrame(cat_encoded, columns=cat_columns)
        return self.cat_df

    def log_transform(self, col:str) -> pd.DataFrame:
        '''This method log transforms the specified column in the dataframe'''
        self.df[str(col)+'_log_transformed'] = np.log(self.df[col]+1)
        return self.df

    def standardise_feature(self, cols:list, ml_datatype:str, path:str) -> pd.DataFrame:
        '''This method standardises the numerical features in the dataframe'''
        scaler = StandardScaler()
        if ml_datatype == 'train':
            for col in cols:
                col_data = self.df[[col]]
                scaler.fit(col_data)
                self.df[str(col)+'_standardised'] = scaler.transform(col_data)
                self._dump_object(scaler, f"{path}/{col}_scaler.pkl")
        else:
            for col in cols:
                scaler = self._load_object(f"{path}/{col}_scaler.pkl")
                col_data = self.df[[col]]
                self.df[str(col)+'_standardised'] = scaler.transform(col_data)
        return self.df

    def create_ml_ready_df(self) -> pd.DataFrame:
        '''This method creates the ml ready dataframe with raw and transformed features'''
        self.transformed_df = self.df.join(self.cat_df) #.set_index('case_id')
        col_types = self.transformed_df.dtypes
        obj_col = col_types[col_types == 'object'].index.tolist()
        num_col = col_types[col_types != 'object'].index.tolist()
        sorted_cols = obj_col + num_col
        self.transformed_df = self.transformed_df[sorted_cols]
        return self.transformed_df

    def save_data(self, output_filepath:str) -> None:
        '''This method saves the dataframe to the specified output filepath'''
        self.df.to_parquet(output_filepath)

def main(input_filepath:str, output_filepath:str, ml_datatype:str) -> None:
    '''This function executes the feature engineering process flow and logic on the data
     and saves the transformed data to the output filepath'''

    marital_dict = {
    'Married': ['Married','Marriedmarried','Singlemarried','Widowedmarried',
                'Divorcedmarried','Living_With_Partnermarried'], 
    'Single':  ['Single', 'Singlesingle','Divorcedsingle','Widowedsingle',
                'Marriedsingle','Living_With_Partnersingle'],
    'Divorced': ['Divorced', 'Singledivorced', 'Marrieddivorced', 'Divorceddivorced',
                 'Widoweddivorced', 'Living_With_Partnerdivorced'],
    'Widowed': ['Widowed', 'Marriedwidowed', 'Singlewidowed', 'Divorcedwidowed',
                'Widowedwidowed','Living_With_Partnerwidowed'],
    'Not Supplying': ['Not Disclosed'],
    'civil_partnership': ['Living_With_Partner','Singleliving_With_Partner',
                          'Marriedliving_With_Partner','Living_With_Partnerliving_With_Partner',
                          'Widowedliving_With_Partner','Divorcedliving_With_Partner']}
    example = FeatureEngineering(file_path=input_filepath)
    example.load_data()
    example.engineer_age(date_cols=['date_decision', 'dateofbirth'], feature_name='age')
    example.engineer_time_of_year(date_col='date_decision')
    example.engineer_marital_status(col='marital_status', value='Not disclosed',
                                    grouping_dict=marital_dict)
    example.bin_numerical_feature('age', bins=[0,25,50,125], labels=['young', 'middle_aged', 'old'])
    example.bin_numerical_feature('income', bins=[0, 25000, 50000, 200000],
                                  labels=['low', 'medium', 'high'])
    example.engineer_family_size(col='no_of_children', value=0, new_col_name='family_size',
                                bins=[0, 1, 2, 10], labels=['single', 'small', 'large'])
    example.drop_columns(cols=['credit_status', "debt"])
    example.drop_rows(col_subset=['age', 'family_size'])
    example.one_hot_encode(cat_col_list=['time_of_year','gender', 'family_size',
                                        'marital_status_new', 'age_binned', 'income_binned'])
    example.log_transform(col='income')
    example.standardise_feature(cols=['age', 'income'], ml_datatype=ml_datatype,
                                path="artifacts/model_dev")
    example.create_ml_ready_df()
    example.save_data(output_filepath)

if __name__ == "__main__":
    PROJECT_STAGE = "Feature Engineering"
    try:
        logger.info(">>>>> Executing the feature engineering process")
        dataset = ["train", "dev"]
        for data_type in dataset:
            main(input_filepath=f"artifacts/data_prep/output/{data_type}.parquet",
                 output_filepath=f"artifacts/data_prep/output/ml_{data_type}.parquet",
                 ml_datatype=data_type)
            logger.info(">>>> {data_type} data has been transformed and saved \
                successfully %s", data_type)
        logger.info(">>>>> Ended process for feature engineering <<<<< %s", PROJECT_STAGE)
    except Exception as e:
        logger.exception(e)
        raise e

# %%
