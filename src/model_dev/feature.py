#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#%%
train_df = pd.read_parquet("artifacts/data_prep/output/train.parquet")
train_df.set_index("case_id", inplace=True)


#Filling the gaps, dropping and deriving new features
#%%
def calculate_age(df:pd.DataFrame):
    df[['date_decision', 'dateofbirth']] = df[['date_decision', 'dateofbirth']].apply(pd.to_datetime)
    df['age'] = (df['date_decision'] - df['dateofbirth']).dt.days // 365
    return df

train_df = calculate_age(train_df)
# %%

def assign_time_of_year(df:pd.DataFrame):
    df['time_of_year'] = df['date_decision'].dt.month.apply(lambda x: 
        'winter' if x in [12, 1, 2] else 'spring' 
        if x in [3, 4, 5] else 'summer' 
        if x in [6, 7, 8] else 'autumn')
    return df

train_df = assign_time_of_year(train_df)


# %%
def impute_no_of_children(df:pd.DataFrame):
    df['no_of_children'] = df['no_of_children'].fillna(0)
    return df

train_df = impute_no_of_children(train_df)

#%%
def bin_family_size(df:pd.DataFrame):
    df['family_size'] = df['no_of_children'].apply(lambda x: 'single' if x == 0 else 'small' if x < 2 else 'large')
    return df

train_df = bin_family_size(train_df)

# %%
def drop_columns(df:pd.DataFrame, cols:list):
    return df.drop(columns=cols)

train_df = drop_columns(train_df, ['credit_status', "debt"])

#%%
#marital status catgories is too noisy, I will group them into 5 categories
#grouping list  = ['Married', 'Single', 'Divorced', 'Widowed',civil_partnership, 'Not disclosed']
grouping_dict = {
                 'Married': ['Married','Marriedmarried','Singlemarried','Widowedmarried','Divorcedmarried','Living_With_Partnermarried'], 
                 'Single':  ['Single', 'Singlesingle','Divorcedsingle','Widowedsingle','Marriedsingle','Living_With_Partnersingle'],
                 'Divorced': ['Divorced', 'Singledivorced', 'Marrieddivorced', 'Divorceddivorced', 'Widoweddivorced', 'Living_With_Partnerdivorced'],
                 'Widowed': ['Widowed', 'Marriedwidowed', 'Singlewidowed', 'Divorcedwidowed', 'Widowedwidowed', 'Living_With_Partnerwidowed'],
                 'Not Supplying': ['Not Disclosed'],
                 'civil_partnership': ['Living_With_Partner','Singleliving_With_Partner','Marriedliving_With_Partner', 'Living_With_Partnerliving_With_Partner', 'Widowedliving_With_Partner', 'Divorcedliving_With_Partner']
}

def group_marital_status(df:pd.DataFrame, grouping_dict:dict):
    df['marital_status'] = df['marital_status'].fillna('Not disclosed')
    df['marital_status'] = df['marital_status'].str.title()
    df['marital_status_new'] = df['marital_status']
    for group, values in grouping_dict.items():
        df.loc[df['marital_status'].isin(values), 'marital_status_new'] = group
    return df

train_df = group_marital_status(train_df, grouping_dict)  

# %%
def drop_na_rows(df:pd.DataFrame, cols:str):
    return df.dropna(subset=cols)

train_df = drop_na_rows(train_df, 'age')

#%%
# Next step is to transform each numerica feature depending on my intuition - this will be used in training to obtain and see the effect of numeric features altered
#Bin the ages & income
def bin_feature(df:pd.DataFrame, col:str, col_bin:list, label:list) -> pd.DataFrame:
    df[str(col) + '_binned'] = pd.cut(df[col], bins=col_bin,
                                      labels=label, include_lowest=True)
    return df

train_df = bin_feature(train_df, col='age', col_bin=[0,25,50,125],
                       label=['young', 'middle_aged', 'old'])
train_df = bin_feature(train_df, col='income', col_bin=[0, 25000, 50000, 200000],
                       label=['low', 'medium', 'high'])

#%%
# Next step is transform the categorical and numerical columns to be ml ready
def one_hot_encode(df:pd.DataFrame, cat_col_list:list) -> pd.DataFrame:
    '''This function transforms the categorical columns to be ml ready'''
    # df = df.select_dtypes(include=['object'])
    df = df[cat_col_list]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
    cat_encoded = encoder.fit_transform(df)
    cat_columns = []
    for i, col in enumerate(df.columns):
        for cat in encoder.categories_[i]:
            cat_columns.append(f"{col}_{cat}")
    cat_df = pd.DataFrame(cat_encoded, columns=cat_columns)
    return cat_df

x = one_hot_encode(train_df, ['time_of_year', 'family_size', 'marital_status_new', 'age_binned', 'income_binned'])

#%%
#Log transform the income 
def log_transform(df:pd.DataFrame, col:str):
    df[str(col)+'_log_transformed'] = np.log(df[col]+1)
    return df
train_df = log_transform(train_df, 'income')

#%%
#Standardise numerical features
def standardise(df: pd.DataFrame, cols: list):
    scaler = StandardScaler()
    for col in cols:
        col_data = df[[col]]  # Extract column data as DataFrame
        df[str(col)+'_standardised'] = scaler.fit_transform(col_data)  # Fit and transform
    return df
train_df = standardise(train_df, ['age', 'income'])

# %%
# Write a normalised function
#Sort the joining of categorical and numerical features to create the ml ready data

