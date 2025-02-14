import pandas as pd
import numpy as np
import subprocess
import os
import json
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import get_clean_dataset
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
import sys
sys.path.append("/Users/luisvazquez/imodelsExperiments")
import os
from hierarchical_shrinkage import HSTreeClassifierCV
from Marina_models import ESTIMATORS_CLASSIFICATION
#from ucimlrepo import fetch_ucirepo
#from hierarchical_shrinkage import HSTreeClassifierCV
#from sklearn.model_selection import GridSearchCV
#from config.shrinkage.models import ESTIMATORS_CLASSIFICATION


######################## VARIABLES ########################
LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
METRIC_AUC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
METRIC_ACC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
DATASET = "haberman"
SOURCE = "imodels"


################################################
#
# MODELS
#
################################################

# HS_model = HSTreeClassifierCV()
# DT = DecisionTreeClassifier()

# cart_hscart_estimators = [
#     model for model_group in ESTIMATORS_CLASSIFICATION
#     for model in model_group
#     if model.name in ['CART', 'HSCART']
# ]

models = []
for i in LEAVES:
    models.append(HSTreeClassifierCV(estimator_=DecisionTreeClassifier(max_leaf_nodes=i)))  # Appending a new instance of HSTreeClassifierCV

for i in  LEAVES:
    models.append(DecisionTreeClassifier(max_leaf_nodes=i))  # Appending a new instance of DecisionTreeClassifier

print(models)  # This will contain 5 HS models and 5 DT models



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
test = hs_models[0]

################################################################################################################################################################################################
hs = []
cart = []
hs_models = models[:10]
cart_models = models[10:]
results = []
for i in range(0, 10):
    auc_scores_dt = []
    acc_scores_dt = []
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

    for model in hs_models:
        model.fit(train_x, train_y)
        y_pred_proba = model.predict_proba(test_x)[:,1]
        auc_cart = roc_auc_score(test_y, y_pred_proba)

        # Append CART results
        hs.append({
            'Model': 'HSCART',
            'AUC': auc_cart,
            'Split Seed': i
        })
    for model in cart_models:
        model.fit(train_x, train_y)
        y_pred_proba = model.predict_proba(test_x)[:, 1]
        auc_hscart = roc_auc_score(test_y, y_pred_proba)
        cart.append({
            'Model': 'CART',
            'AUC': auc_hscart,
            'Split Seed': i
        })

hs = pd.DataFrame(hs)
cart = pd.DataFrame(cart)

################################################################################################################################################################################################
################################################
#
# NEW WAY
#
################################################
# results = []
# for i in range(0, 10):
#     auc_scores_dt = []
#     acc_scores_dt = []
#     train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)
#
#     for model_config in cart_hscart_estimators:
#         model_name = model_config.name
#         model_class = model_config.cls
#         model_kwargs = model_config.kwargs.copy()
#
#         if model_name == "CART":
#             cart_model = model_class(**model_kwargs)
#             cart_model.fit(train_x, train_y)
#             y_pred_proba = cart_model.predict_proba(test_x)[:,1]
#             auc_cart = roc_auc_score(test_y, y_pred_proba)
#
#             # Append CART results
#             results.append({
#                 'Model': 'CART',
#                 'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
#                 'Lambda': None,  # CART does not use lambda
#                 'AUC': auc_cart,
#                 'Split Seed': i
#             })
#         elif model_name == "HSCART":
#             model = model_class(**model_kwargs)
#             model.fit(train_x, train_y)
#             y_pred_proba = model.predict_proba(test_x)[:, 1]
#             auc_hscart = roc_auc_score(test_y, y_pred_proba)
#             results.append({
#                 'Model': 'HSCART',
#                 'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
#                 'Lambda': model.reg_param,  # Save the selected lambda
#                 'AUC': auc_hscart,
#                 'Split Seed': i
#             })


# results_df = pd.DataFrame(results)
#results_df.to_csv(f"{specific_dataset_name}_cart_hscart_results_combined_GTC.csv", index=False)



# cart = results_df[results_df['Model']=="CART"]
# hscart = results_df[results_df['Model']=='HSCART']
# Group by 'Max Leaves' and calculate the average AUC
average_auc = cart.groupby('Max Leaves')['AUC'].mean().reset_index()
avg_auc_hs  = hs.groupby('Max Leaves')['AUC'].mean().reset_index()



######################## DT ########################
# for i in range(0, 10):
#     auc_scores_dt = []
#     acc_scores_dt = []
#     train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)
#
#     for num_leaves in LEAVES:
#         DT = DecisionTreeClassifier(max_leaf_nodes=num_leaves)
#         DT.fit(train_x, train_y)
#         #predictions = DT.predict(test_x)
#         y_pred_proba = DT.predict_proba(test_x)[:, 1]
#
#         auc_dt = roc_auc_score(test_y, y_pred_proba)
#         #acc_dt = accuracy_score(test_y, predictions)
#
#         auc_scores_dt.append(auc_dt)
#         #acc_scores_dt.append(acc_dt)
#
#     METRIC_AUC.loc[i] = auc_scores_dt
#     #METRIC_ACC.loc[i] = acc_scores_dt


# ######################## HS ########################
# for i in range(0, 10):
#     auc_scores_hs = []
#     acc_scores_hs = []
#     train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)
#
#     # k_fold = KFold(n_splits=3, shuffle=True, random_state=i)
#     # parameters = [{'reg_param': [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]}]
#     #
#     # gs_dt = GridSearchCV(HS_model, param_grid=parameters, scoring='roc_auc', cv=k_fold)
#     # gs_dt.fit(train_x, train_y)
#     # best_lambda = gs_dt.best_params_['reg_param']
#
#     for num_leaves in LEAVES:
#         HS = HSTreeClassifierCV(reg_param=best_lambda, max_leaf_nodes=num_leaves)
#         HS.fit(train_x, train_y)
#         predictions = HS.predict(test_x)
#         #y_pred_proba = HS.predict_proba(test_x)[:, 1]
#
#         auc_hs = roc_auc_score(test_y, predictions)
#         #acc_hs = accuracy_score(test_y, predictions)
#
#         auc_scores_hs.append(auc_hs)
#         #acc_scores_hs.append(acc_hs)
#
#
#     METRIC_AUC.loc[i] = auc_scores_hs
#     #METRIC_ACC.loc[i] = acc_scores_hs
#


################################################
#
# METRICS
#
################################################

# ######################## DT AVERAGE METRICS  ########################
# data_auc_dt = []
# data_acc_dt = []
# for cols in METRIC_AUC.columns:
#     data_auc_dt.append(METRIC_AUC[cols].mean())
#     data_acc_dt.append(METRIC_ACC[cols].mean())
#
# ######################## HS AVERAGE METRICS  ########################
# data_auc_hs = []
# data_acc_hs = []
# for cols in METRIC_AUC.columns:
#     data_auc_hs.append(METRIC_AUC[cols].mean())
#     data_acc_hs.append(METRIC_ACC[cols].mean())








################################################
#
# PLOTS
#
################################################

######################## AUC SCORE ########################
plt.figure(figsize=(10,6))
plt.plot(average_auc['Max Leaves'], average_auc['AUC'], marker='o', linestyle='-', color='red', label='CART')
plt.plot(avg_auc_hs['Max Leaves'], avg_auc_hs['AUC'], marker='o', linestyle='-', color='b', label="HSCART")
plt.xlabel("Number of Leaves")
plt.ylabel("AUC")
plt.grid(True)
#plt.title("Juvenile", fontsize=20)
plt.legend()
#plt.savefig("juvenile_class_auc")
plt.show()

# ######################## ACCURACY ########################
# plt.figure(figsize=(10, 6))
# plt.plot(LEAVES, data_acc_hs, marker='o', linestyle='-', color='red', label='HS ACCURACY')
# plt.plot(LEAVES, data_acc_dt, marker='o', linestyle='-', color='b', label='DT ACCURACY')
# plt.xlabel('Number of Leaves')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.title("Juvenile", fontsize=20)
# plt.legend()
# plt.savefig("juvenile_class_acc")
# plt.show()
