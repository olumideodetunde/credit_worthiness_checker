'''This module is used to create the machine learning dataset'''
import pandas as pd
from sklearn.model_selection import train_test_split
from src import data_preprocess as dp
from src import src_logger as logger

def read_parent_data(file_paths:list) -> list:
    '''
    This function reads the parent data from the file paths and returns a list of dataframes
    '''
    dataframes = []
    for path in file_paths:
        name = path.split(".")[1] + "_df"
        name = dp.read_data(file_path=path)
        dataframes.append(name)
    return dataframes

def create_feature_df(feature_name:str, parentfeature:dict,
                      feature_recency:str, proposed_datatype:dict,
                      join_method:str = None, common_col:str = None) -> pd.DataFrame:
    '''
    ** of importance is the parentfeature which is a dictionary with the key as progressive 
    number in string (1,2,3 etc) and the value as a tuple of the dataframe and the subset 
    column name.**
    '''
    features = []
    for _, (key, value) in enumerate(parentfeature.items()):
        dataframe, column = value[0], value[1]
        feature = feature_name + key
        feature = dp.get_individual_feature(df=dataframe, col_list=column, recency=feature_recency)
        features.append(feature)
    features_combined = dp.merge_features_into_df(dfs=features, on_col=common_col,
                                                  join_type=join_method)
    feature_df = dp.create_consolicated_df(df=features_combined, new_colname=feature_name)
    feature_df = dp.handle_features_datatype(df=feature_df, col_datatype=proposed_datatype)
    return feature_df

def create_ml_dataset(feature_dfs:list, base_df:pd.DataFrame,
                      common_col:str = None) -> pd.DataFrame:
    '''
    This function takes all the individual feature dataframes and the base dataframe 
    and creates the final dataset for machine learning.
    '''
    ml_df = base_df
    for df in feature_dfs:
        ml_df = pd.merge(ml_df, df, on=common_col, how="outer")
    return ml_df

def split_ml_dataset(df:pd.DataFrame, target_col:str, test_size:float, random_state:int):
    '''
    This function takes the machine learning dataset and splits it into train and test datasets
    '''
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state,
                                        stratify=df[target_col])
    return train_df, test_df

def save_traindev_dataset(dataset:dict, output_path:str) -> None:
    '''
    This function saves the train and dev machine learning datasets to the output path
    '''
    for name,dataframe in dataset.items():
        output_name = output_path + "/" + name + ".parquet"
        dataframe.to_parquet(output_name)

def main(input_paths:list, output_path:str):
    '''
    This function is the main function that calls all the other functions 
    to create the machine learning dataset
    '''
    parent_dfs = read_parent_data(input_paths)
    base_df, static_df, person_df, previous_df = parent_dfs[0], parent_dfs[1],\
                                parent_dfs[2], parent_dfs[3]

    dob_df = create_feature_df(feature_name="dateofbirth",
                                parentfeature={"One":(static_df, ['case_id', "dateofbirth_337D"]),
                                    "Two":(static_df, ['case_id', "dateofbirth_342D"])},
                                feature_recency="last",
                                common_col="case_id",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "dateofbirth":"date"})
    gender_df = create_feature_df(feature_name='gender',
                                parentfeature={"One":(person_df, ['case_id', "sex_738L"])},
                                feature_recency="first",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "gender":"string"})
    child_df = create_feature_df(feature_name='no_of_children',
                                parentfeature={"One":(previous_df, ['case_id', "childnum_21L"]),
                                    "Two":(person_df, ['case_id', "childnum_185L"])},
                                feature_recency="last",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "no_of_children":"integer"})
    marital_df = create_feature_df(feature_name='marital_status',
                                parentfeature={"One":(person_df, ['case_id', "familystate_447L"]),
                                    "Two":(previous_df, ['case_id', "familystate_726L"])},
                                feature_recency="last",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "marital_status":"string"})
    inc_df = create_feature_df(feature_name='income',
                                parentfeature={"One":(person_df, ['case_id',
                                    "mainoccupationinc_384A"])},
                                feature_recency="last",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "income":"float"})
    debt_df = create_feature_df(feature_name='debt',
                                parentfeature={"One":(previous_df, ['case_id',
                                    "outstandingdebt_522A"])},
                                feature_recency="last",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "debt":"float"})
    credit_df = create_feature_df(feature_name='credit_status',
                                parentfeature={"One":(previous_df, ['case_id',
                                    "credacc_status_367L"])},
                                feature_recency="last",
                                join_method="outer",
                                proposed_datatype={"case_id":"integer", "credit_status":"string"})

    ml_data = create_ml_dataset(feature_dfs=[dob_df, gender_df, child_df, marital_df, inc_df,
                                             debt_df, credit_df],
                            base_df=base_df, common_col="case_id")

    ml_data = split_ml_dataset(df=ml_data, target_col="target", test_size=0.2, random_state=42)

    save_traindev_dataset(dataset={"train":ml_data[0], "dev":ml_data[1]}, output_path=output_path)

if __name__ == "__main__":

    PROJECT_STAGE = "Dataset Preparation"
    try:
        logger.info(">>>>>> Started <<<<<< %s", PROJECT_STAGE)
        INPUT_LIST = [
        "artifacts/data/raw/parquet_files/train/train_base.parquet",
        "artifacts/data/raw/parquet_files/train/train_static_cb_0.parquet",
        "artifacts/data/raw/parquet_files/train/train_person_1.parquet",
        "artifacts/data/raw/parquet_files/train/train_applprev_1_1.parquet"
        ]
        OUTPUT_DIR = "artifacts/data/output"
        main(input_paths=INPUT_LIST, output_path=OUTPUT_DIR)
        logger.info(">>>>>> Ended <<<<<< %s", PROJECT_STAGE)

    except Exception as e:
        logger.exception(e)
        raise e

#End of file
