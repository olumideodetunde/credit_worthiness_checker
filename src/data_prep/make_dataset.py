#%%
from src.data_prep import data_preprocess as dp


#%%
def read_parent_data(file_paths:list):
    dataframes = []
    for path in file_paths:
        name = path.split(".")[1] + "_df"
        name = dp.read_data(file_path=path)
        dataframes.append(name)
    return dataframes

base_data, static_data, person_data, previous_application = read_parent_data(["data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet",
                                                                    "data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_static_cb_0.parquet",
                                                                    "data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_person_1.parquet",
                                                                    "data/raw/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_1.parquet"])

#%%

def create_feature_df(feature_name:str, parentfeature_dict:dict, feature_recency:str, join_method:str, proposed_datatype:dict):
    n = len(parentfeature_dict)
    features = []
    for key, value in parentfeature_dict.items(): #include enumerate to handle more than one dataframe
        feature = dp.get_individual_feature(value[0], value[1], feature_recency)
        features.append(feature)
    #features_combined = dp.merge_features_into_df(dfs=features, on_col="case_id", join_type=join_method) #deal with this method to handle more than one dataframe
    feature_df = dp.create_consolicated_df(df=features[0], new_colname=feature_name)
    feature_df = dp.handle_features_datatype(df=feature_df, col_datatype=proposed_datatype)
    return feature_df

dob_df = create_feature_df(feature_name="dateofbirth",
                           parentfeature_dict={"One":(static_data, ['case_id', "dateofbirth_337D"])},
                           feature_recency="last",
                           join_method="outer",
                           proposed_datatype={"case_id":"integer", "dateofbirth":"date"})

#%%

def create_ml_dataset():
    pass


#%%
def split_ml_dataset():
    pass


#%%
def save_traindev_dataset():
    pass


def main():
    pass



if __name__ == "__main__":
    main()










# #%%
# # Tested for the dob feature
# dob_df1 = dp.get_individual_feature(static_data, ["case_id", "dateofbirth_337D"], "last")
# # dob_df2 = get_inidivual_feature(static_data_2, ["case_id", "dateofbirth_342D"], "last")
# # dob_comb = merge_dfs([dob_df1, dob_df2], "case_id", "outer")
# # dob_df = create_consolicated_df(dob_comb, "dateofbirth")
# # dob_df = handle_feature_datatype(dob_df, {"case_id":"integer", "dateofbirth":"date"})

# # %%
# ## Testing the methods on the gender feature - this is just found in one dataframe so no need to merge
# gender = dp.get_individual_feature(person_data, ["case_id","sex_738L"], "first")
# # gender_df = create_consolicated_df(gender, "gender")
# # gender_df = handle_feature_datatype(gender_df, {"case_id":"integer", "gender":"string"})

# #%%
# child1 = dp.get_individual_feature(previous_application, ["case_id", "childnum_21L"], "last")
# # child2 = get_individual_feature(person_data, ["case_id", "childnum_185L"], "last")
# # child_comb = merge_features_into_df([child1, child2], "case_id", "outer")
# # child_df = create_consolicated_df(child_comb, "no_of_children")


# # %%
# # Deal with getting the marital status from the previous_application_df

# fam1 = get_individual_feature(previous_application_df, ["case_id", "familystate_726L"], "last")
# fam2 = get_individual_feature(person_data, ["case_id", "familystate_447L"], "last")
# fam_comb = merge_features_into_df([fam1, fam2], "case_id", "outer")
# fam_df = create_consolicated_df(fam_comb, "marital_status")
# fam_df = handle_feature_datatype(fam_df, {"case_id":"integer", "marital_status":"string"})


# # %%
# #revist the consolidation method to handle integer and float properly

# inc = get_individual_feature(person_data, ["case_id", "mainoccupationinc_384A"], "last")
# inc_df = create_consolicated_df(inc, "main_income")
# inc_df = handle_feature_datatype(inc_df, {"case_id":"integer", "main_income":"integer"})

# # %%
# debt = get_individual_feature(previous_application_df, ["case_id", "outstandingdebt_522A"], "last")
# debt_df = create_consolicated_df(debt, "outstanding_debt")

# #debt_df = handle_feature_datatype(debt_df, {"case_id":"integer", "outstanding_debt":"integer"})


# # %%
# credit = get_individual_feature(previous_application_df, ["case_id", "credacc_status_367L"], "last")
# credit_df = create_consolicated_df(credit, "credit_status")
# credit_df = handle_feature_datatype(credit_df, {"case_id":"integer", "credit_status":"string"})