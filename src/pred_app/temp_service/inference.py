#%%
import pandas as pd
import mlflow.sklearn

#%%
#load the model
model = mlflow.sklearn.load_model("artifacts/model_dev/model")

#%%
#create a sample data
sample_data = {
    "gender": "M",
    "age": 30,
    "income": 100000,
    "no_of_children": 0,
}

#create the dataframe and transform the data
sample_df = pd.DataFrame(sample_data, index=[0])
sample_df
# %%
