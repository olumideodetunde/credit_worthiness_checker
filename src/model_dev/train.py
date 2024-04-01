#%%
import mlflow
import pandas as pd
from typing import Tuple
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

    def _split_data_into_features_target(self, data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = data.drop(columns=["target"])
        target = data["target"]
        return features, target

    def _make_prediction(self, data) -> Tuple[pd.Series, pd.Series]:
        features, target = self._split_data_into_features_target(data)
        predict = self.model.predict(features.iloc[:, 15:])
        y_score = self.model.predict_proba(features.iloc[:, 15:])[:,1]
        return predict, y_score

    def load_data(self, train_name:str, dev_name:str) -> pd.DataFrame:
        try:
            self.train_data = pd.read_parquet(self.input_path + '/' + train_name)
            self.dev_data = pd.read_parquet(self.input_path + '/' + dev_name)
            return self.train_data, self.dev_data
        except FileNotFoundError:
            raise FileNotFoundError("File not found")

    def select_model(self, algorithm:str, hyperparameters:dict=None) -> object:
        if algorithm == "RandomForest":
            self.model = RandomForestClassifier()
        elif algorithm == "LogisticRegression":
            self.model = LogisticRegression()
        elif algorithm == "Baseline":
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError("Invalid algorithm")
        return self.model

    def train_model(self) -> object:
        train_features, train_target = self._split_data_into_features_target(self.train_data)
        self.model.fit(train_features.iloc[:,15:], train_target)
        return self.model

    def evaluate_model(self) -> dict:
        predict, y_score = self._make_prediction(self.dev_data)
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
        return self.metrics

    def end_training(self) -> None:
        mlflow.end_run()
        return None

def main():
    mlflow.set_experiment("CreditWorthinessTraining")
    with mlflow.start_run():
        trainer = ModelTrainer(input_path="artifacts/data_prep/output")
        trainer.load_data("ml_train.parquet", "ml_dev.parquet")
        mlflow.log_param("train_data", trainer.train_data.describe())
        mlflow.log_param("dev_data", trainer.dev_data.describe())
        trainer.select_model("LogisticRegression")
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("hyperparameters", trainer.model.get_params())
        trainer.train_model()
        mlflow.log_param("model", trainer.model)
        metrics = trainer.evaluate_model()
        mlflow.log_metrics(metrics)
        trainer.end_training()
    return None
if __name__ == "__main__":
    main()

# %%
