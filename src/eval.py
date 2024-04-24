"""This module contains functions to evaluate the model"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score


def eval_with_average_precision_score(
    y_true: Union[np.ndarray, pd.Series], y_score: Union[np.ndarray, pd.Series]
) -> float:
    """This function calculates the average precision score of the model"""
    average_precision = average_precision_score(y_true, y_score)
    return average_precision


def eval_with_auc_and_pr_curve(
    y_true: Union[np.ndarray, pd.Series], y_score: Union[np.ndarray, pd.Series]
) -> float:
    """This function calculates the area under the precision-recall curve of the model"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall


def eval_with_f_beta_score(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    beta: float = 0.5,
) -> float:
    """This function calculates the f-beta score of the model"""
    f_beta = fbeta_score(y_true, y_pred, beta=beta)
    return f_beta
