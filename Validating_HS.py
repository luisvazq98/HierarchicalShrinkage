import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor, HSTreeClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV, cross_val_score

######################## VARIABLES ########################
leafs = np.array([2,4,8,12,15,20,24,28,30,32])
metric_auc = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
metric_acc = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])

######################## MODELS ########################
HS = HSTreeRegressor()
DT = DecisionTreeRegressor()



################################################
#
# REGRESSION DATASETS
#
################################################

# ######################## DATASET (DIABETES (REGRESSION) ) ########################
# diabetes = datasets.load_diabetes()
# X = diabetes.data
# Y = diabetes.target
#
#
# ######################## DATASET ( RED WINE ) ########################
# # Download the dataset
# path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
# # Check the files in the directory
# files = os.listdir(path)
# file_path = os.path.join(path, "winequality-red.csv")
#
# # Load the dataset into a DataFrame
# df = pd.read_csv(file_path)
# X = df.drop(columns=['quality'])
# Y = df['quality']



################################################
#
# CLASSIFICATION DATASETS
#
################################################

######################## DATASET ( DIABETES (CLASSIFICATION) ) ########################
# # Download latest version
# path = kagglehub.dataset_download("mathchi/diabetes-data-set")
# files = os.listdir(path)
# file_path = os.path.join(path, "diabetes.csv")
#
# df = pd.read_csv(file_path)
# X = df.drop(columns=['Outcome'])
# Y = df['Outcome']


# ######################## DATASET ( BREAST CANCER ) ########################
# # fetch dataset
# breast_cancer = fetch_ucirepo(id=14)
#
# df = breast_cancer.data
#
# # data (as pandas dataframes)
# X = breast_cancer.data.features
# Y = breast_cancer.data.targets
#
# # Identify and encode categorical columns
# categorical_columns = X.select_dtypes(include=["object"]).columns
# label_encoders = {}
#
# for col in categorical_columns:
#     le = LabelEncoder()
#     X.loc[:, col] = le.fit_transform(X[col])  # Use .loc for assignment
#     label_encoders[col] = le
#
# # Encode the target variable Y
# y_label_encoder = LabelEncoder()
# Y = y_label_encoder.fit_transform(Y)

######################## DATASET ( HABERMAN ) ########################
# fetch dataset
haberman_s_survival = fetch_ucirepo(id=43)

# data (as pandas dataframes)
X = haberman_s_survival.data.features
Y = haberman_s_survival.data.targets



################################################
#
# TRAINING MODELS
#
################################################

######################## DT ########################
for i in range(0, 10):
    auc_scores_dt = []
    acc_scores_dt = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    for num_leaves in leafs:
        DT = DecisionTreeClassifier(max_leaf_nodes=num_leaves)
        DT.fit(train_x, train_y)
        predictions = DT.predict(test_x)

        auc_dt = roc_auc_score(test_y, predictions)
        acc_dt = accuracy_score(test_y, predictions)

        auc_scores_dt.append(auc_dt)
        acc_scores_dt.append(acc_dt)

    metric_auc.loc[i] = auc_scores_dt
    metric_acc.loc[i] = acc_scores_dt


######################## HS ########################
for i in range(0, 10):
    auc_scores_hs = []
    acc_scores_hs = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    k_fold = KFold(n_splits=3, shuffle=True, random_state=i)
    parameters = [{'reg_param': [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]}]

    gs_dt = GridSearchCV(HS, param_grid=parameters, scoring='r2', cv=k_fold)
    gs_dt.fit(train_x, train_y)
    best_lambda = gs_dt.best_params_['reg_param']

    for num_leaves in leafs:
        HS = HSTreeClassifier(reg_param=best_lambda, max_leaf_nodes=num_leaves)
        HS.fit(train_x, train_y)
        predictions = HS.predict(test_x)

        auc_hs = roc_auc_score(test_y, predictions)
        acc_hs = accuracy_score(test_y, predictions)

        auc_scores_hs.append(auc_hs)
        acc_scores_hs.append(acc_hs)


    metric_auc.loc[i] = auc_scores_hs
    metric_acc.loc[i] = acc_scores_hs


######################## DT AVERAGE METRICS  ########################
data_auc_dt = []
data_acc_dt = []
for cols in metric_auc.columns:
    data_auc_dt.append(metric_auc[cols].mean())
    data_acc_dt.append(metric_acc[cols].mean())

######################## HS AVERAGE METRICS  ########################
data_auc_hs = []
data_acc_hs = []
for cols in metric_auc.columns:
    data_auc_hs.append(metric_auc[cols].mean())
    data_acc_hs.append(metric_acc[cols].mean())



################################################
#
# PLOTS
#
################################################

######################## AUC SCORE ########################
plt.figure(figsize=(10,6))
plt.plot(leafs, data_auc_hs, marker='o', linestyle='-', color='red', label='HS AUC')
plt.plot(leafs, data_auc_dt, marker='o', linestyle='-', color='b', label="DT AUC")
plt.xlabel("Number of Leaves")
plt.ylabel("AUC")
plt.grid(True)
#plt.title("R2 Score", fontsize=20)
plt.legend()
plt.savefig("haberman_class_auc")
plt.show()

######################## ACCURACY ########################
plt.figure(figsize=(10, 6))
plt.plot(leafs, data_acc_hs, marker='o', linestyle='-', color='red', label='HS ACCURACY')
plt.plot(leafs, data_acc_dt, marker='o', linestyle='-', color='b', label='DT ACCURACY')
plt.xlabel('Number of Leaves')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.title("Mean Squared Error", fontsize=20)
plt.legend()
plt.savefig("haberman_class_acc")
plt.show()






