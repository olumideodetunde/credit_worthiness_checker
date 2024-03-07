#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
train_df = pd.read_parquet("data/output/train.parquet")
dev_df = pd.read_parquet("data/output/dev.parquet")

# %%
# Confirm that all the datatypes of the columns
def _check_dtypes(df):
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

_check_dtypes(train_df)

# %%
#Noticed that the date column is not in datetime format
def convert_date(df:pd.DataFrame, date_cols:list) -> pd.DataFrame:
    df[date_cols] = df[date_cols].apply(pd.to_datetime, format="%Y-%m-%d", errors="coerce")
    return df

convert_date(train_df, ["date_decision", "dateofbirth"])
convert_date(dev_df, ["date_decision", "dateofbirth"])

# %%
# Check data completeness
def check_completeness(df:pd.DataFrame) -> None:
    x = (df.isnull().sum())
    report = pd.DataFrame({
        "column": x.index,
        "missing": x.values,
        "percentage": (x.values / len(df)) * 100,
        }).sort_values(by="percentage", ascending=True)
    plt.figure(figsize=(5, 3))
    sns.barplot(x="column", y="percentage", data=report)
    plt.title("Percentage of missing data")
    plt.xticks(rotation=90)
    plt.show()

check_completeness(train_df)
# %%
#Summary statistics of the columns depending on the data type

#get the numerical columns
def get_numeric_summary(df:pd.DataFrame, numeric_cols:list) -> pd.DataFrame:
    num_df = df[numeric_cols]
    x = num_df.describe().T
    return x

a = get_numeric_summary(train_df, ["no_of_children", "debt", "income"])
a

# %%
def display_cat_summary(df:pd.DataFrame, cat_cols:list) -> None:
    cat_df = df[cat_cols]
    figs, axes = plt.subplots(3,1)
    
    for i, column in enumerate(cat_df.columns):
        counts = cat_df[column].value_counts()
        counts.plot(
            kind ="bar",
            ax = axes.flatten()[i],
            fontsize = "large"
        ).set_title(column)
    figs.set_size_inches(15, 12)
    #plt.tight_layout()
    plt.show()

display_cat_summary(train_df, cat_cols=['gender', 'marital_status', 'credit_status'])

# %%
#Look at date related columns, create new feature - age
def create_age(df:pd.DataFrame, date_cols:list) -> pd.DataFrame:
    df["age"] = df[date_cols[0]].dt.year - df[date_cols[1]].dt.year
    return df

train_df = create_age(train_df, ["date_decision", "dateofbirth"])
dev_df = create_age(dev_df, ["date_decision", "dateofbirth"])

# %%
#Exploring relationships between feautures guided by questions/business understanding & target variable

#Does the age affect the 