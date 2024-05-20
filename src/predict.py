#Setting up the inference pipeline

#%%
import pandas as pd
import mlflow.sklearn
from src.feature import main as feature_main

#%%
test = pd.read_parquet("artifacts/data/output/dev.parquet")
test = feature_main(
    input_filepath="artifacts/data/output/dev.parquet",
    output_filepath="artifacts/data/output/assumed_test.parquet",
    ml_datatype="test"
)
test = pd.read_parquet("artifacts/data/output/assumed_test.parquet")

# %%
one_line = test.iloc[:1, :]
one_line

# %%
#load the model
model = mlflow.sklearn.load_model("artifacts/model/model")

#%%
#make a prediction for one line of the test data
x = one_line.drop(columns=["target"])[['gender_F', 'gender_M', 'age_standardised', 'income', 'family_size_large', 
                                       'family_size_single', 'family_size_small', 'no_of_children']]
y = one_line["target"]
y_pred = model.predict(x)
y_pred

# %%
#make predictions for the entire test data
x = test.drop(columns=["target"])[['gender_F', 'gender_M', 'age_standardised', 'income', 'family_size_large', 
                                       'family_size_single', 'family_size_small', 'no_of_children']]
y = test["target"]
y_pred = model.predict(x)
comparison = pd.DataFrame({"actual": y, "predicted": y_pred})
comparison
# %%
