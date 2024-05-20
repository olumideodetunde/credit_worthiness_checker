'''This script is used to train a model, evaluate it, and save it to output dir'''
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, \
balanced_accuracy_score,average_precision_score, accuracy_score, fbeta_score

class ModelTrainer:
    '''This class is used to train, evaluate, and export a model.'''

    def __init__(self, input_path:str) -> None:
        self.input_path = input_path
        self.train_data = None
        self.dev_data = None
        self.model = None
        self.metrics = None

    def _test_manual_upsampling(self, data:pd.DataFrame) -> pd.DataFrame:
        '''This function is used to upsample the minority class in the dataset.'''
        data_majority = data[data["target"] == 0]
        data_minority = data[data["target"] == 1]
        df_minority_upsampled = resample(data_minority,
                                         replace=True,
                                         n_samples = len(data_majority),
                                         random_state=42)
        df = pd.concat([data_majority, df_minority_upsampled])
        return df

    def _split_data_into_features_target(self, data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        '''This function is used to split the data into features and target.'''
        features = data.drop(columns=["target"])
        target = data["target"]
        return features, target

    def load_data(self, train_name:str, dev_name:str) -> pd.DataFrame:
        '''This function is used to load the train and dev data.It upsamples the train data.'''
        try:
            self.train_data = pd.read_parquet(self.input_path + '/' + train_name)
            self.dev_data = pd.read_parquet(self.input_path + '/' + dev_name)
            self.train_data = self._test_manual_upsampling(self.train_data)
            return self.train_data, self.dev_data
        except FileNotFoundError as e:
            raise FileNotFoundError("File not found") from e

    def select_model(self, algorithm:str, hyperparameters:dict=None) -> object:
        '''This function is used to select the model. It currently supports only XGBoost.'''
        if algorithm == "XGBoost":
            if hyperparameters is None:
                self.model = GradientBoostingClassifier()
            else:
                self.model = GradientBoostingClassifier(**hyperparameters)
        else:
            raise ValueError("Invalid algorithm")
        return self.model

    def train_model(self, selected_features:list=None) -> object:
        '''This function is used to train the model. It uses the selected features.'''
        train_features, train_target = self._split_data_into_features_target(self.train_data)
        self.model = self.model.fit(train_features[selected_features], train_target)
        return self.model

    def evaluate_model(self, selected_features:str) -> dict:
        '''This function is used to evaluate the model on the dev data'''
        dev_features, dev_target = self._split_data_into_features_target(self.dev_data)
        predict = self.model.predict(dev_features[selected_features])
        y_score = self.model.predict_proba(dev_features[selected_features])[:,1]
        balanced_acc = balanced_accuracy_score(dev_target, predict)
        accuracy = accuracy_score(dev_target, predict)
        precision = precision_score(dev_target, predict)
        recall = recall_score(dev_target, predict)
        auc_pr = average_precision_score(dev_target, y_score)
        f_beta_score = fbeta_score(dev_target, predict, beta=0.5)
        self.metrics = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "auc_pr": auc_pr,
            "f_beta_score": f_beta_score
        }
        return self.metrics

    def export_model(self, model_path:str) -> None:
        '''This function is used to export the model. It uses MLflow to save the model.'''
        mlflow.sklearn.save_model(self.model, model_path)
        mlflow.end_run()

def main():
    '''This is the main function that trains the model, logs the metrics to MLflow and saves
    the model to the deploy folder.'''
    mlflow.set_experiment("CreditWorthinessTraining")
    with mlflow.start_run(run_name="XGBoost Training - to be deployed"):
        training_features = ['gender_F', 'gender_M', 'age_standardised', 'income_standardised',
                             'family_size_large', 'family_size_single', 'family_size_small',
                             'no_of_children',]
        trainer = ModelTrainer(input_path="artifacts/data/output")
        trainer.load_data("ml_train.parquet", "ml_dev.parquet")
        mlflow.log_param("data_downsampling_parameter", ["data_minority","replace=False",
                                           "n_samples = len(data_majority)",
                                           "random_state=42"])
        trainer.select_model("XGBoost", hyperparameters={"n_estimators": 1000})
        mlflow.log_param("algorithm", "XGBoost")
        mlflow.log_param("hyperparameters", trainer.model.get_params())
        trainer.train_model(selected_features=training_features)
        mlflow.log_param("selected_features", training_features)
        mlflow.log_param("model", trainer.model)
        metrics = trainer.evaluate_model(selected_features=training_features)
        mlflow.log_metrics(metrics)
        trainer.export_model("deploy/backend/model")

if __name__ == "__main__":
    main()
#EOF
