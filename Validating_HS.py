import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score

######################## VARIABLES ########################
leafs = np.array([2,4,8,12,15,20,24,28,30,32])
metric_r2 = pd.DataFrame(columns=[2,4,8,12,15,20,24,28,30,32])
metric_mse = pd.DataFrame(columns=[2,4,8,12,15,20,24,28,30,32])

######################## MODELS ########################
HS = HSTreeRegressor()
DT = DecisionTreeRegressor()

# ######################## DATASET (DIABETES) ########################
# diabetes = datasets.load_diabetes()
# X = diabetes.data
# Y = diabetes.target

######################## DATASET (RED WINE) ########################
# Download the dataset
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
# Check the files in the directory
files = os.listdir(path)
file_path = os.path.join(path, "winequality-red.csv")

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)
X = df.drop(columns=['quality'])
Y = df['quality']


######################## DT ########################
for i in range(0, 10):
    r2_scores_dt = []
    mse_dt = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    for num_leaves in leafs:
        DT = DecisionTreeRegressor(max_leaf_nodes=num_leaves)
        DT.fit(train_x, train_y)
        predictions = DT.predict(test_x)
        r2 = r2_score(test_y, predictions)
        mse1 = mean_squared_error(test_y, predictions)
        r2_scores_dt.append(r2)
        mse_dt.append(mse1)

    metric_r2.loc[i] = r2_scores_dt
    metric_mse.loc[i] = mse_dt

######################## HS ########################
for i in range(0, 10):
    r2_scores_hs = []
    mse_hs = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=i)
    parameters = [{'reg_param': [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]}]

    gs_dt = GridSearchCV(HS, param_grid=parameters, scoring='r2', cv=k_fold)
    gs_dt.fit(train_x, train_y)
    best_lambda = gs_dt.best_params_['reg_param']

    for num_leaves in leafs:
        HS = HSTreeRegressor(reg_param=best_lambda, max_leaf_nodes=num_leaves)
        HS.fit(train_x, train_y)
        predictions = HS.predict(test_x)
        r2_hs = r2_score(test_y, predictions)
        mse2 = mean_squared_error(test_y, predictions)
        r2_scores_hs.append(r2_hs)
        mse_hs.append(mse2)


    metric_r2.loc[i] = r2_scores_hs
    metric_mse.loc[i] = mse_hs


######################## DT AVERAGE METRICS  ########################
data_r2_dt = []
data_mse_dt = []
for cols in metric_r2.columns:
    data_r2_dt.append(metric_r2[cols].mean())
    data_mse_dt.append(metric_mse[cols].mean())

######################## HS AVERAGE METRICS  ########################
data_r2_hs = []
data_mse_hs = []
for cols in metric_r2.columns:
    data_r2_hs.append(metric_r2[cols].mean())
    data_mse_hs.append(metric_mse[cols].mean())



################################################
#
# PLOTS
#
################################################

######################## R2 SCORE ########################
plt.figure(figsize=(10,6))
plt.plot(leafs, data_r2_hs, marker='o', linestyle='-', color='red', label='HS R2 Score')
plt.plot(leafs, data_r2_dt, marker='o', linestyle='-', color='b', label="DT R2 Score")
plt.xlabel("Number of Leaves")
plt.ylabel("R2 Score")
plt.grid(True)
plt.title("R2 Score", fontsize=20)
plt.legend()
plt.savefig("RedWine_Reg_R2")
plt.show()

######################## MSE ########################
plt.figure(figsize=(10, 6))
plt.plot(leafs, data_mse_hs, marker='o', linestyle='-', color='red', label='HS MSE')
plt.plot(leafs, data_mse_dt, marker='o', linestyle='-', color='b', label='DT MSE')
plt.xlabel('Number of Leaves')
plt.ylabel('MSE')
plt.grid(True)
plt.title("Mean Squared Error", fontsize=20)
plt.legend()
plt.savefig("RedWine_Reg_MSE")
plt.show()







