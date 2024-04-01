#%%
import pandas as pd
import numpy as np
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from eval import eval_with_auc_and_pr_curve
from eval import eval_with_average_precision_score
from eval import eval_with_f_beta_score

class ModelTrainer:
    def __init__(self, input_path:str) -> None:
        self.input_path = input_path
        self.train_data = None
        self.dev_data = None
        self.model = None
        self.metrics = None
        self.run_id = None

    def _split_data_into_features_target(self, data):
        features = data.drop(columns=["target"])
        target = data["target"]
        return features, target

    def initialise_training(self) -> None:
        mlflow.set_experiment("CreditWorthinessModelTraining")
        mlflow.start_run()
        self.run_id = mlflow.active_run().info.run_id
        return None
   
    def load_data(self, train_name:str, dev_name:str) -> pd.DataFrame:
        try:
            with mlflow.start_run(run_id=self.run_id):
                self.train_data = pd.read_parquet(self.input_path + '/' + train_name)
                self.dev_data = pd.read_parquet(self.input_path + '/' + dev_name)
                mlflow.log_param("train_data", self.train_data.describe())
                mlflow.log_param("dev_data", self.dev_data.describe())
            return self.train_data, self.dev_data
        except FileNotFoundError:
            raise FileNotFoundError("File not found")

    def select_model(self, algorithm:str, hyperparameters:dict=None) -> object:
        with mlflow.start_run(run_id=self.run_id):
            if algorithm == "RandomForest":
                self.model = RandomForestClassifier(**hyperparameters)
            elif algorithm == "LogisticRegression":
                self.model = LogisticRegression(**hyperparameters)
            elif algorithm == "Baseline":
                self.model = DummyClassifier(strategy="most_frequent")
            else:
                raise ValueError("Invalid algorithm")
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_param("hyperparameters", hyperparameters)
        return self.model

    def train_model(self) -> object:
        with mlflow.start_run(run_id=self.run_id):
            train_features, train_target = self._split_data_into_features_target(self.train_data)
            self.model.fit(train_features, train_target)
            mlflow.sklearn.log_model(self.model, "model")
        return self.model

    def predict(self, data) -> tuple:
        with mlflow.start_run(run_id=self.run_id):
            features, target = self._split_data_into_features_target(data)
            predict = self.model.predict(features)
            y_score = self.model.predict_proba(features)[:,1]
        return predict, y_score

    def evaluate_model(self) -> dict:
        with mlflow.start_run(run_id=self.run_id):
            predict, y_score = self.predict(self.dev_data)
            _, dev_target = self._split_data_into_features_target(self.dev_data)
            accuracy = accuracy_score(dev_target, predict)
            precision = precision_score(dev_target, predict)
            recall = recall_score(dev_target, predict)
            auc_pr = eval_with_auc_and_pr_curve(dev_target, y_score)
            auc_pr2 = eval_with_average_precision_score(dev_target, y_score)
            f_beta_score = eval_with_f_beta_score(dev_target, predict)
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc_pr": auc_pr,
                "auc_pr2": auc_pr2,
                "f_beta_score": f_beta_score
            }
            mlflow.log_metrics(self.metrics)
        return self.metrics

    def end_training(self) -> None:
        mlflow.end_run()
        return None

#%%
if __name__ == "__main__":
        trainer = ModelTrainer(input_path="artifacts/data_prep/output")
        trainer.initialise_training()
        trainer.load_data("train_data.parquet", "dev_data.parquet")
        trainer.select_model("RandomForest")
        trainer.train_model()
        trainer.evaluate_model()
        trainer.end_training()

# %%
