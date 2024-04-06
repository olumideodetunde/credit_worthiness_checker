#%%
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
    
    def load_data(self, train_name:str, dev_name:str) -> pd.DataFrame:
        try:
            self.train_data = pd.read_parquet(self.input_path + '/' + train_name)
            self.dev_data = pd.read_parquet(self.input_path + '/' + dev_name)
            return self.train_data, self.dev_data
        except FileNotFoundError:
            raise FileNotFoundError("File not found")

    def select_model(self, algorithm:str, hyperparameters:dict=None) -> object:
        if algorithm == "RandomForest":
            if hyperparameters is None:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestClassifier(**hyperparameters)
        elif algorithm == "LogisticRegression":
            if hyperparameters is None:
                self.model = LogisticRegression()
            else:
                self.model = LogisticRegression(**hyperparameters)
        elif algorithm == "XGBoost":
            if hyperparameters is None:
                self.model = GradientBoostingClassifier()
            else:
                self.model = GradientBoostingClassifier(**hyperparameters)
        elif algorithm == "RandomForest":
            if hyperparameters is None:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestClassifier(**hyperparameters)
        elif algorithm == "Baseline":
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError("Invalid algorithm")
        return self.model

    def train_model(self, selected_features:list=None) -> object:
        train_features, train_target = self._split_data_into_features_target(self.train_data)
        self.model = self.model.fit(train_features[selected_features], train_target)
        #self.model.fit(train_features.iloc[:,15:], train_target)
        return self.model

    def evaluate_model(self, selected_features:str) -> dict:
        # predict, y_score = self._make_prediction(self.dev_data, selected_features)
        dev_features, dev_target = self._split_data_into_features_target(self.dev_data)
        predict = self.model.predict(dev_features[selected_features])
        y_score = self.model.predict_proba(dev_features[selected_features])[:,1]
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
        training_features = ["gender_F", "gender_M", "age_standardised", 
                             "income",'family_size_large', 
                             'family_size_single','family_size_small',
                             "no_of_children",'time_of_year_autumn',
                             'time_of_year_spring', 'time_of_year_summer', 
                             'time_of_year_winter',]
        trainer = ModelTrainer(input_path="artifacts/data_prep/output")
        trainer.load_data("ml_train.parquet", "ml_dev.parquet")
        # mlflow.log_param("train_data", str(trainer.train_data))
        # mlflow.log_param("dev_data", str(trainer.dev_data))
        trainer.select_model("RandomForest")
        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("hyperparameters", trainer.model.get_params())
        trainer.train_model(selected_features=training_features)
        mlflow.log_param("selected_features", training_features)
        mlflow.log_param("model", trainer.model)
        metrics = trainer.evaluate_model(selected_features=training_features)
        mlflow.log_metrics(metrics)
        trainer.end_training()
    return None

#%%
if __name__ == "__main__":
    main()
# %%
