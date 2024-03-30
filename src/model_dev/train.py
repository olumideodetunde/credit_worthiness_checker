#%%

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from eval import eval_with_auc_and_pr_curve
from eval import eval_with_average_precision_score
from eval import eval_with_f_beta_score

#%%
ml_traindf = pd.read_parquet("artifacts/data_prep/output/ml_train.parquet")
ml_devdf = pd.read_parquet("artifacts/data_prep/output/ml_dev.parquet")
train_feature = ml_traindf.drop(columns=["target"])
train_targe = ml_traindf["target"]
dev_feature = ml_devdf.drop(columns=["target"])
dev_targe = ml_devdf["target"]

#%%
#Establish a baseline
def train_baseline(train_features, train_target, dev_features, dev_target):
    
    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_feature, train_targe)
    predict = model.predict(dev_feature)
    accuracy = accuracy_score(dev_targe, predict)
    precision = precision_score(dev_targe, predict)
    recall = recall_score(dev_targe, predict)
    y_score = model.predict_proba(dev_feature)[:,1]
    auc_pr = eval_with_auc_and_pr_curve(dev_targe, y_score)
    auc_pr2 = eval_with_average_precision_score(dev_targe, y_score)
    f_beta_score = eval_with_f_beta_score(dev_targe, predict)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_pr": auc_pr,
        "auc_pr2": auc_pr2,
        "f_beta_score": f_beta_score
    
    }

x = train_baseline(train_feature, train_targe, dev_feature, dev_targe)

#%%
#Implement experiment tracking and logging

#%%
#Set up to test different models

#%%
#Set up to test different hyperparameters

#%%
#Set up to test different features

#%%
#Log each results with eval.py

#%%
#Identitfy the best model

#%%
#Export the best model