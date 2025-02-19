import pandas as pd
import numpy as np
import sys
import imodels as im
import subprocess
import os
import json
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor, HSTreeClassifier, HSTreeClassifierCV
from imodels import get_clean_dataset
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
from ucimlrepo import fetch_ucirepo
sys.path.insert(0, "/Users/luisvazquez")
sys.path.insert(0, "/Users/luisvazquez/imodelsExperiments")
from imodelsExperiments.config.shrinkage.models import ESTIMATORS_CLASSIFICATION
from sklearn.model_selection import GridSearchCV



################################################
#
# MODELS
#
################################################
def get_models():
    LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    models = []
    for i in LEAVES:
        models.append(HSTreeClassifierCV(estimator_=DecisionTreeClassifier(max_leaf_nodes=i)))

    for i in  LEAVES:
        models.append(DecisionTreeClassifier(max_leaf_nodes=i))

    return models


################################################
#
# REGRESSION DATASETS
#
################################################
def get_regression_dataset(DATASET):
    if DATASET == "diabetes":
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        Y = diabetes.target
        return X, Y
    elif DATASET == "red wine":
        # Download the dataset
        path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
        # Check the files in the directory
        files = os.listdir(path)
        file_path = os.path.join(path, "winequality-red.csv")

        # Load the dataset into a DataFrame
        df = pd.read_csv(file_path)
        X = df.drop(columns=['quality'])
        Y = df['quality']
        return X, Y
    else:
        print("Error. Please enter a valid dataset")



# ################################################
# #
# # CLASSIFICATION DATASETS
# #
# ################################################
def get_classification_dataset(DATASET, SOURCE):
    if SOURCE == 'imodels':
        X, Y, features = get_clean_dataset(DATASET, SOURCE, return_target_col_names=True)
        return X, Y, features
    else:
        if DATASET == "diabetes":
            # Download latest version
            path = kagglehub.dataset_download("mathchi/diabetes-data-set")
            files = os.listdir(path)
            file_path = os.path.join(path, "diabetes.csv")

            df = pd.read_csv(file_path)
            X = df.drop(columns=['Outcome'])
            Y = df['Outcome']
            return X, Y

        elif DATASET == "breast cancer":
            # Fetch dataset
            breast_cancer = fetch_ucirepo(id=14)
            df = breast_cancer.data

            # Data (as pandas dataframes)
            X = breast_cancer.data.features
            Y = breast_cancer.data.targets

            # Dropping rows with NaN values
            Y = Y.drop(index=X[X.isna().any(axis=1)].index)
            X = X.dropna(axis=0)

            # Reset the indices of X and Y
            X.reset_index(drop=True, inplace=True)
            Y.reset_index(drop=True, inplace=True)

            # Identify and encode categorical columns
            categorical_columns = X.select_dtypes(include=["object"]).columns
            label_encoders = {}

            for col in categorical_columns:
                le = OneHotEncoder(sparse_output=False)
                transformed = le.fit_transform(X[col].values.reshape(-1, 1))

                # Convert transformed data into a DataFrame and join back with X
                transformed_df = pd.DataFrame(transformed, columns=le.categories_[0])
                X = X.drop(columns=[col]).join(transformed_df, lsuffix='_left', rsuffix='_right')
                label_encoders[col] = le

            # Encode the target variable Y
            y_label_encoder = LabelEncoder()
            Y = y_label_encoder.fit_transform(Y)
            return X, Y

        elif DATASET == "haberman":
            # fetch dataset
            haberman_s_survival = fetch_ucirepo(id=43)

            # data (as pandas dataframes)
            X = haberman_s_survival.data.features
            Y = haberman_s_survival.data.targets

            return X, Y

################################################
#
# TRAINING MODELS
#
################################################
def training_models(X, Y, MODELS):
    results = []
    for i in range(0, 10):
        auc_scores_dt = []
        acc_scores_dt = []
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1 / 3, random_state=i)

        for model_config in MODELS:
            model_name = model_config.name
            model_class = model_config.cls
            model_kwargs = model_config.kwargs.copy()

            if model_name == "CART":
                cart_model = model_class(**model_kwargs)
                cart_model.fit(train_x, train_y)
                y_pred_proba = cart_model.predict_proba(test_x)[:,1]
                auc_cart = roc_auc_score(test_y, y_pred_proba)

                # Append CART results
                results.append({
                    'Model': 'CART',
                    'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
                    'Lambda': None,  # CART does not use lambda
                    'AUC': auc_cart,
                    'Split Seed': i
                })
            elif model_name == "HSCART":
                model = model_class(**model_kwargs)
                model.fit(train_x, train_y)
                y_pred_proba = model.predict_proba(test_x)[:, 1]
                auc_hscart = roc_auc_score(test_y, y_pred_proba)
                results.append({
                    'Model': 'HSCART',
                    'Max Leaves': model_kwargs['max_leaf_nodes'],  # Directly taken from ModelConfig
                    'Lambda': model.reg_param,  # Save the selected lambda
                    'AUC': auc_hscart,
                    'Split Seed': i
                })


    results_df = pd.DataFrame(results)
    return results_df




if __name__ == "__main__":

    ######################## VARIABLES ########################
    #METRIC_AUC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    #METRIC_ACC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    DATASET = "heart"
    SOURCE = "imodels"

    ######################## MODELS ########################
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in ['CART', 'HSCART']
    ]

    # Use the following function only if above does not work
    # models = get_models()

    ######################## DATASET ########################
    # X, Y = get_regression_dataset(DATASET)
    X, Y = get_classification_dataset(DATASET, SOURCE)


    ######################## TRAINING MODELS ########################
    results = training_models(X, Y, cart_hscart_estimators)

    ######################## GETTING METRICS ########################
    cart = results[results['Model']=="CART"]
    hscart = results[results['Model']=='HSCART']
    # Group by 'Max Leaves' and calculate the average AUC
    average_auc = cart.groupby('Max Leaves')['AUC'].mean().reset_index()
    avg_auc_hs = hscart.groupby('Max Leaves')['AUC'].mean().reset_index()

    ######################## PLOTS ########################

    # AUC Score
    plt.figure(figsize=(10,6))
    plt.plot(LEAVES, data_auc_hs, marker='o', linestyle='-', color='red', label='HS AUC')
    plt.plot(LEAVES, data_auc_dt, marker='o', linestyle='-', color='b', label="DT AUC")
    plt.xlabel("Number of Leaves")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.title(DATASET, fontsize=20)
    plt.legend()
    #plt.savefig("juvenile_class_au")
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(LEAVES, data_acc_hs, marker='o', linestyle='-', color='red', label='HS ACCURACY')
    plt.plot(LEAVES, data_acc_dt, marker='o', linestyle='-', color='b', label='DT ACCURACY')
    plt.xlabel('Number of Leaves')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.title(DATASET, fontsize=20)
    plt.legend()
    #plt.savefig("juvenile_class_acc")
    plt.show()
