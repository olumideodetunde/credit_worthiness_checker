#%%
import pandas as pd
import mlflow.sklearn
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from fastapi import FastAPI, Path, Response, Request
from pydantic import BaseModel

 # %%
class Transform:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def engineer_family_size(self, col: str, new_col_name: str, bins: list, labels: str) -> pd.DataFrame:
        """This method calculates the family size based on the number of children"""
        self.df[str(col) + "_binned"] = pd.cut(self.df[col], bins=bins, labels=labels,
                                                include_lowest=True)
        self.df.rename(columns={col + "_binned": new_col_name}, inplace=True)
        return self.df

    def standardise_feature(
        self, cols: list, mldatatype: str, path: str) -> pd.DataFrame:
        """This method standardises the numerical features in the dataframe"""
        scaler = StandardScaler()
        if mldatatype == "test":
            for col in cols:
                col_data = self.df[[col]]
                scaler.fit(col_data)
                self.df[str(col) + "_standardised"] = scaler.transform(col_data)
            return self.df
        else:
            raise ValueError("Invalid mldatatype")
                
    def one_hot_encode(self, cat_col_list: list, path:str) -> pd.DataFrame:
        """This method transforms the categorical columns to be ml ready"""
        encoder = OneHotEncoder(sparse_output=False)
        cat_encoded = encoder.fit_transform(self.df[cat_col_list])
        cat_columns = []
        for i, col in enumerate(cat_col_list):
            for category in encoder.categories_[i]:
                cat_columns.append(f"{col}_{category}")
        cat_encoded = pd.DataFrame(cat_encoded, columns=cat_columns)
        self.df = pd.concat([self.df, cat_encoded], axis=1)
        return self.df
    
    def create_inference_df(self, required_features:list) -> pd.DataFrame:
        """This method creates a dataframe with the required features for inference"""
        for col in required_features:
            if col not in self.df.columns:
                self.df[col] = 0
        return self.df[required_features]

def make_prediction(data, model):
    y_pred = model.predict(data)
    return y_pred


#%%
#Create the complete ML endpoint

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

@app.get("/get_input_data", response_model=InputData, summary="Get the input data for the model")
def get_gender(response: InputData = None):
    input_data = InputData(
        gender="M",
        age=30,
        income=100000,
        no_of_children=0
    )
    input_data = input_data.dict()
    return input_data

@app.post("/predict", response_model=PredResponse, summary="Make a prediction")
def predict(data:InputData):
    model = mlflow.sklearn.load_model("artifacts/model_dev/model")
    feature = Transform(pd.DataFrame(data.dict(), index=[0]))
    feature.engineer_family_size(
        col="no_of_children",
        new_col_name="family_size",
        bins=[0, 1, 2, 10],
        labels=["single", "small", "large"],
    )
    feature.standardise_feature(["age", "income"], "test", "artifacts/model_dev")
    feature.one_hot_encode(
    cat_col_list=["gender",
                    "family_size"],
    path="artifacts/model_dev")
    feature = feature.create_inference_df(required_features=['gender_F', 'gender_M', 
                                                      'age_standardised', 'income', 
                                                      'family_size_large', 
                                                      'family_size_single', 'family_size_small', 
                                                      'no_of_children'])
    prediction = make_prediction(data=feature, model=model)
    return {"prediction": prediction[0]}