#%%
#The goal is to come up with one whole metric that ties neatly into the business metric.

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score

# What if theere are 3 layers of metrics? - REVIEW THE business metric again



# %%

#The following functions give me the 
def eval_with_average_precision_score(y_true, y_score):
    average_precision = average_precision_score(y_true, y_score)
    return average_precision

def eval_with_auc_and_pr_curve(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_precision_recall = auc(recall, precision)
    return auc_precision_recall

