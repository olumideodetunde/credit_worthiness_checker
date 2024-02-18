#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
base_credit = pd.read_csv("data/raw/home-credit-credit-risk-model-stability/csv_files/train/train_base.csv")
base_credit
# %%
appl = pd.read_csv("data/raw/home-credit-credit-risk-model-stability/csv_files/train/train_applprev_1_0.csv")
appl
# %%
