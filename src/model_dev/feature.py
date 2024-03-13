

#%%
import pandas as pd
import numpy as np


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
        'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'autumn')
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
def aggregate_marital_status(df:pd.DataFrame):
    #rename the none values to not disclosed
    df['marital_status'] = df['marital_status'].fillna('not disclosed')
    return df

train_df = aggregate_marital_status(train_df)

# %%
#testing dropping all the rows with no age
y = train_df.copy()
y.dropna(subset=['age'], inplace=True)
# %%
