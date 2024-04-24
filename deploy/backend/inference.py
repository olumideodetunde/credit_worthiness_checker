#%%
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from backend.transform import Transform

def make_prediction(data, model):
    y_pred = model.predict(data)
    return y_pred

app = FastAPI()

class InputData(BaseModel):
    gender:str
    age:int
    income:int
    no_of_children:int

class PredResponse(BaseModel):
    prediction: int

@app.get("/")
def test():
    return {"message": "Welcome to the Credit Worthiness Prediction API"}

@app.post("/predict", response_model=PredResponse, summary="Make a prediction")
def predict(data:InputData):
    model = mlflow.sklearn.load_model("")
    feature = Transform(pd.DataFrame(data.dict(), index=[0]))
    feature.engineer_family_size(
        col="no_of_children",
        new_col_name="family_size",
        bins=[0, 1, 2, 10],
        labels=["single", "small", "large"],
    )
    feature.standardise_feature(["age", "income"], "test",
                                "")
    feature.one_hot_encode(
    cat_col_list=["gender",
                    "family_size"],
    path="")
    feature = feature.create_inference_df(required_features=['gender_F', 'gender_M',
                                                      'age_standardised', 'income', 
                                                      'family_size_large', 
                                                      'family_size_single', 'family_size_small', 
                                                      'no_of_children'])
    prediction = make_prediction(data=feature, model=model)
    output = {
        "prediction": prediction[0]
    }
    return output