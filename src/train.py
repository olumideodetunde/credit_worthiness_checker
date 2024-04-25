#%%
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow.sklearn
from src.eval import eval_with_auc_and_pr_curve
from src.eval import eval_with_average_precision_score
from src.eval import eval_with_f_beta_score

#%%
class ModelTrainer:
    def __init__(self, input_path:str) -> None:
        self.input_path = input_path
        self.train_data = None
        self.dev_data = None
        self.model = None
        self.metrics = None
        
    def _test_manual_downsampling(self, data:pd.DataFrame) -> pd.DataFrame:
        data_majority = data[data["target"] == 0]
        data_minortiy = data[data["target"] == 1]
        df_majority_downsampled = resample(data_majority,
                                           replace=False,
                                           n_samples = 3*len(data_minortiy),
                                           random_state=42)
        df = pd.concat([df_majority_downsampled, data_minortiy])
        return df
    
    def _test_manual_upsampling(self, data:pd.DataFrame) -> pd.DataFrame:
        data_majority = data[data["target"] == 0]
        data_minortiy = data[data["target"] == 1]
        df_minortiy_upsampled = resample(data_minortiy,
                                         replace=True,
                                         n_samples = len(data_majority),
                                         random_state=42)
        df = pd.concat([data_majority, df_minortiy_upsampled])
        return df

    def _split_data_into_features_target(self, data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        features = data.drop(columns=["target"])
        target = data["target"]
        return features, target

    def load_data(self, train_name:str, dev_name:str) -> pd.DataFrame:
        try:
            self.train_data = pd.read_parquet(self.input_path + '/' + train_name)
            self.dev_data = pd.read_parquet(self.input_path + '/' + dev_name)
            #self.train_data = self._test_manual_downsampling(self.train_data)
            self.train_data = self._test_manual_upsampling(self.train_data)
            # self.dev_data = self._test_manual_downsampling(self.dev_data)
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
        elif algorithm == "DecisionTree":
            if hyperparameters is None:
                self.model = DecisionTreeClassifier()
            else:
                self.model = DecisionTreeClassifier(**hyperparameters)
        elif algorithm == "SVM":
            if hyperparameters is None:
                self.model = SVC(probability=True)
            else:
                self.model = SVC(**hyperparameters)
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
    
    def export_model(self, model_path:str) -> None:
        mlflow.sklearn.save_model(self.model, model_path)
        return None
    
    def end_training(self) -> None:
        mlflow.end_run()
        return None
    

def main():
    mlflow.set_experiment("CreditWorthinessTraining")
    with mlflow.start_run():
        training_features = ['gender_F', 'gender_M', 'age_standardised', 'income',
                             'family_size_large', 'family_size_single', 'family_size_small',
                             'no_of_children',]
        trainer = ModelTrainer(input_path="artifacts/data_prep/output")
        trainer.load_data("ml_train.parquet", "ml_dev.parquet")
        mlflow.log_param("data_downsampling_parameter", ["data_minority","replace=False",
                                           "n_samples = len(data_majority)",
                                           "random_state=42"])
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
        trainer.export_model("artifacts/model_dev/model")
        trainer.end_training()
    return None

#%%
if __name__ == "__main__":
    main()