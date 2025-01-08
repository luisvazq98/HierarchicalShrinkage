import pandas as pd
import numpy as np
import imodels as im
import subprocess
import os
import json
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor, HSTreeClassifier
from imodels import get_clean_dataset
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV

#import imodels as im

######################## VARIABLES ########################
LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
METRIC_AUC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
METRIC_ACC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
DATASET = "recidivism"
SOURCE = "imodels"

# MODELS
HS = HSTreeClassifier()
DT = DecisionTreeClassifier()


# Getting clean dataset from iModels
X, Y, features = get_clean_dataset(DATASET, SOURCE, return_target_col_names=True)


################################################
#
# REGRESSION DATASETS
#
################################################

# ######################## DIABETES (REGRESSION) ########################
# diabetes = datasets.load_diabetes()
# X = diabetes.data
# Y = diabetes.target
#
#
# ######################## RED WINE ########################
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
#
#
#
# ################################################
# #
# # CLASSIFICATION DATASETS
# #
# ################################################

# ######################## DIABETES (CLASSIFICATION) ########################
# # Download latest version
# path = kagglehub.dataset_download("mathchi/diabetes-data-set")
# files = os.listdir(path)
# file_path = os.path.join(path, "diabetes.csv")
#
# df = pd.read_csv(file_path)
# X = df.drop(columns=['Outcome'])
# Y = df['Outcome']


######################## BREAST CANCER ########################
# # Fetch dataset
# breast_cancer = fetch_ucirepo(id=14)
# df = breast_cancer.data
#
# # Data (as pandas dataframes)
# X = breast_cancer.data.features
# Y = breast_cancer.data.targets
#
# # Dropping rows with NaN values
# Y = Y.drop(index=X[X.isna().any(axis=1)].index)
# X = X.dropna(axis=0)
#
# # Reset the indices of X and Y
# X.reset_index(drop=True, inplace=True)
# Y.reset_index(drop=True, inplace=True)
#
#
# # Identify and encode categorical columns
# categorical_columns = X.select_dtypes(include=["object"]).columns
# label_encoders = {}
#
# for col in categorical_columns:
#     le = OneHotEncoder(sparse_output=False)
#     transformed = le.fit_transform(X[col].values.reshape(-1, 1))
#
#     # Convert transformed data into a DataFrame and join back with X
#     transformed_df = pd.DataFrame(transformed, columns=le.categories_[0])
#     X = X.drop(columns=[col]).join(transformed_df, lsuffix='_left', rsuffix='_right')
#     label_encoders[col] = le
#
# # Encode the target variable Y
# y_label_encoder = LabelEncoder()
# Y = y_label_encoder.fit_transform(Y)


# ######################## HABERMAN ########################
# # fetch dataset
# haberman_s_survival = fetch_ucirepo(id=43)
#
# # data (as pandas dataframes)
# X = haberman_s_survival.data.features
# Y = haberman_s_survival.data.targets


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

    for num_leaves in LEAVES:
        DT = DecisionTreeClassifier(max_leaf_nodes=num_leaves)
        DT.fit(train_x, train_y)
        predictions = DT.predict(test_x)

        auc_dt = roc_auc_score(test_y, predictions)
        acc_dt = accuracy_score(test_y, predictions)

        auc_scores_dt.append(auc_dt)
        acc_scores_dt.append(acc_dt)

    METRIC_AUC.loc[i] = auc_scores_dt
    METRIC_ACC.loc[i] = acc_scores_dt


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

    for num_leaves in LEAVES:
        HS = HSTreeClassifier(reg_param=best_lambda, max_leaf_nodes=num_leaves)
        HS.fit(train_x, train_y)
        predictions = HS.predict(test_x)

        auc_hs = roc_auc_score(test_y, predictions)
        acc_hs = accuracy_score(test_y, predictions)

        auc_scores_hs.append(auc_hs)
        acc_scores_hs.append(acc_hs)


    METRIC_AUC.loc[i] = auc_scores_hs
    METRIC_ACC.loc[i] = acc_scores_hs



################################################
#
# METRICS
#
################################################

######################## DT AVERAGE METRICS  ########################
data_auc_dt = []
data_acc_dt = []
for cols in METRIC_AUC.columns:
    data_auc_dt.append(METRIC_AUC[cols].mean())
    data_acc_dt.append(METRIC_ACC[cols].mean())

######################## HS AVERAGE METRICS  ########################
data_auc_hs = []
data_acc_hs = []
for cols in METRIC_AUC.columns:
    data_auc_hs.append(METRIC_AUC[cols].mean())
    data_acc_hs.append(METRIC_ACC[cols].mean())



################################################
#
# PLOTS
#
################################################

######################## AUC SCORE ########################
plt.figure(figsize=(10,6))
plt.plot(LEAVES, data_auc_hs, marker='o', linestyle='-', color='red', label='HS AUC')
plt.plot(LEAVES, data_auc_dt, marker='o', linestyle='-', color='b', label="DT AUC")
plt.xlabel("Number of Leaves")
plt.ylabel("AUC")
plt.grid(True)
plt.title("Juvenile", fontsize=20)
plt.legend()
plt.savefig("juvenile_class_auc")
plt.show()

######################## ACCURACY ########################
plt.figure(figsize=(10, 6))
plt.plot(LEAVES, data_acc_hs, marker='o', linestyle='-', color='red', label='HS ACCURACY')
plt.plot(LEAVES, data_acc_dt, marker='o', linestyle='-', color='b', label='DT ACCURACY')
plt.xlabel('Number of Leaves')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title("Juvenile", fontsize=20)
plt.legend()
plt.savefig("juvenile_class_acc")
plt.show()
