import pandas as pd
import numpy as np
import sys
import imodels as im
import time
import os
import json
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor, HSTreeClassifier, HSTreeClassifierCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from imodels import get_clean_dataset
sys.path.insert(0, "/Users/luisvazquez")
sys.path.insert(0, "/Users/luisvazquez/imodelsExperiments")
from imodelsExperiments.config.shrinkage.models import ESTIMATORS_CLASSIFICATION

######################## VARIABLES ########################
# METRIC_AUC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
# METRIC_ACC = pd.DataFrame(columns=[2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
dataset_list = ['credit_card_clean', 'diabetes', 'breast_cancer', 'haberman', 'gait', 'student performance'
                'student dropout', 'titanic']
DATASET = "student performance"
SOURCE = "kaggle"
PCA_VALUE = "no"

######################## MODELS ########################
def get_models():
    LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    models = []
    for i in LEAVES:
        models.append(HSTreeClassifierCV(estimator_=DecisionTreeClassifier(max_leaf_nodes=i)))

    for i in  LEAVES:
        models.append(DecisionTreeClassifier(max_leaf_nodes=i))

    return models


######################## REGRESSION DATASETS ########################
def get_regression_dataset(dataset):
    if dataset == "diabetes":
        diabetes = datasets.load_diabetes()
        x = diabetes.data
        y = diabetes.target
        return x, y
    elif dataset == "red wine":
        # Download the dataset
        path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
        # Check the files in the directory
        files = os.listdir(path)
        file_path = os.path.join(path, "winequality-red.csv")

        # Load the dataset into a DataFrame
        df = pd.read_csv(file_path)
        x = df.drop(columns=['quality'])
        y = df['quality']
        return x, y
    else:
        print("Error. Please enter a valid dataset")


######################## CLASSIFICATION DATASETS ########################
def get_classification_dataset(dataset, source):
    if SOURCE == 'imodels':
        x, y, feature = get_clean_dataset(dataset, source, return_target_col_names=True)
        return x, y, feature
    elif SOURCE == 'uci':
        dataset_ids = {
            "breast cancer": 14,
            "haberman": 43,
            "gait": 760,
            "student dropout": 697
        }

        # Fetch the dataset using its corresponding ID
        data = fetch_ucirepo(dataset_ids[DATASET])
        x = data.data.features
        y = data.data.targets

        # Dropping rows with NaN values
        y = y.drop(index=x[x.isna().any(axis=1)].index)
        x = x.dropna(axis=0)

        # Reset the indices of X and Y
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if DATASET == "breast cancer":
            categorical_columns = x.select_dtypes(include=["object"]).columns
            label_encoders = {}

            for col in categorical_columns:
                le = OneHotEncoder(sparse_output=False)
                transformed = le.fit_transform(x[col].values.reshape(-1, 1))

                # Convert transformed data into a DataFrame and join back with X
                transformed_df = pd.DataFrame(transformed, columns=le.categories_[0])
                x = x.drop(columns=[col]).join(transformed_df, lsuffix='_left', rsuffix='_right')
                label_encoders[col] = le

            # Encode the target variable Y
            y_label_encoder = LabelEncoder()
            y = y_label_encoder.fit_transform(y)
            return x, y
        elif DATASET == "gait":
            x.dropna(columns=['condition'], inplace=True)
            y = x['condition']
            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        elif DATASET == "student dropout":
            # Convert categorical variables to numerical using Label Encoding
            categorical_columns = y.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                y.loc[:, col] = le.fit_transform(y[col])

            y = y.squeeze()
            y = y.astype(int)
            y = y.dropna()

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        else:
            return x, y

    elif SOURCE == 'kaggle':
        dataset_paths = {
            "diabetes": {"filename": 'diabetes.csv', "path": "mathchi/diabetes-data-set"},
            "student performance": {"filename": 'Student_performance_data _.csv', "path": "rabieelkharoua/students-performance-dataset"},
        }

        path = kagglehub.dataset_download(dataset_paths[DATASET]['path'])
        file_path = os.path.join(path, dataset_paths[DATASET]['filename'])
        df = pd.read_csv(file_path)
        if DATASET == "diabetes":
            x = df.drop(columns=['Outcome'])
            y = df['Outcome']
            return x, y
        elif DATASET == "student performance":
            x = df.drop(columns=['GradeClass'])
            y = df['GradeClass']
            return x, y

    else:
        if dataset == 'adult income':
            ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income']

            data_adult = pd.read_csv(ADULT_URL, names=ADULT_COLS, sep=',\s', engine='python')

            # Remove rows with missing values
            data_adult.replace('?', np.nan, inplace=True)
            data_adult.dropna(inplace=True)

            # Separate features and target variable
            x = data_adult.drop('income', axis=1)
            y = data_adult['income']

            # Convert categorical variables to numerical using Label Encoding
            categorical_columns = x.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col])

            return x, y

        elif dataset == 'titanic':
            TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
            data_titanic = pd.read_csv(TITANIC_URL)
            data_titanic.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace=True)
            data_titanic['Age'] = data_titanic['Age'].fillna(data_titanic['Age'].median())
            categorical_columns = data_titanic.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                data_titanic[col] = le.fit_transform(data_titanic[col])

            x = data_titanic.drop('Survived', axis=1)
            y = data_titanic['Survived']

            return x, y


######################## TRAINING MODELS ########################
def training_models(x, y, models):
    results = []
    for i in range(0, 10):
        if SOURCE == 'imodels':
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1 / 3, random_state=i)
        else:
            # Splitting dataset
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1 / 3, random_state=i)

            train_x = train_x.reset_index(drop=True).to_numpy()
            train_y = train_y.reset_index(drop=True).to_numpy()
            test_x = test_x.reset_index(drop=True).to_numpy()
            test_y = test_y.reset_index(drop=True).to_numpy()

            # Standard Scaler
            # scaler = StandardScaler()
            # train_x = scaler.fit_transform(train_x)
            # test_x = scaler.transform(test_x)

            # PCA
            if PCA_VALUE == 'yes':
                pca = PCA(0.99)
                train_x = pca.fit_transform(train_x)
                test_x = pca.transform(test_x)


        for model_config in models:
            model_name = model_config.name
            model_class = model_config.cls
            model_kwargs = model_config.kwargs.copy()

            if model_name == "CART":
                # Training model
                cart_model = model_class(**model_kwargs)
                start_time = time.time()
                cart_model.fit(train_x, train_y)
                end_time = time.time()

                # Getting metrics
                y_pred_proba = cart_model.predict_proba(test_x)[:, 1]
                predictions = cart_model.predict(test_x)
                auc_cart = roc_auc_score(test_y, y_pred_proba)
                accuracy = accuracy_score(test_y, predictions)

                # Append CART results
                results.append({
                    'Model': 'CART',
                    'Max Leaves': model_kwargs['max_leaf_nodes'],
                    'Max Depth': cart_model.tree_.max_depth,
                    'Node Count': cart_model.tree_.node_count,
                    'Lambda': None,
                    'AUC': auc_cart,
                    'Accuracy': accuracy,
                    'Time (min)': (end_time - start_time) / 60,
                    'Split Seed': i
                })
            elif model_name == "HSCART":
                # Training model
                hs_model = model_class(**model_kwargs)
                start_time = time.time()
                hs_model.fit(train_x, train_y)
                end_time = time.time()

                # Getting metrics
                y_pred_proba = hs_model.predict_proba(test_x)[:, 1]
                predictions = hs_model.predict(test_x)
                auc_hscart = roc_auc_score(test_y, y_pred_proba)
                accuracy = accuracy_score(test_y, predictions)

                # Append HSCART results
                results.append({
                    'Model': 'HSCART',
                    'Max Leaves': model_kwargs['max_leaf_nodes'],
                    'Max Depth': hs_model.estimator_.tree_.max_depth,
                    'Node Count': hs_model.estimator_.tree_.node_count,
                    'Lambda': hs_model.reg_param,
                    'AUC': auc_hscart,
                    'Accuracy': accuracy,
                    'Time (min)': (end_time - start_time) / 60,
                    'Split Seed': i
                })


    model_results = pd.DataFrame(results)
    return model_results





if __name__ == "__main__":

    ######################## MODELS ########################
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in ['CART', 'HSCART']
    ]

    # Use the following function only if above does not work
    # models = get_models()

    ######################## DATASET ########################
    # x, y = get_regression_dataset(DATASET)
    X, Y = get_classification_dataset(DATASET, SOURCE)


    ######################## TRAINING MODELS ########################
    results_df = training_models(X, Y, cart_hscart_estimators)

    ######################## GETTING METRICS ########################
    cart = results_df[results_df['Model']=="CART"]
    hscart = results_df[results_df['Model']=='HSCART']

    # Group by 'Max Leaves' and calculate the average AUC
    cart_avg_auc = cart.groupby('Max Leaves')['AUC'].mean().reset_index()
    hs_avg_auc = hscart.groupby('Max Leaves')['AUC'].mean().reset_index()

    # Group by 'Max Leaves' and calculate the highest accuracy
    cart_max_acc = cart.groupby('Max Leaves')['Accuracy'].mean().max()
    hs_max_acc = hscart.groupby('Max Leaves')['Accuracy'].mean().max()
    print("CART Accuracy:", (cart_max_acc*100))
    print("HSCART Accuracy:", (hs_max_acc*100))


    ######################## PLOTS ########################

    # AUC Score
    plt.figure(figsize=(10,6))
    plt.plot(cart_avg_auc['Max Leaves'], cart_avg_auc['AUC'], marker='o', linestyle='-', color='blue', label='CART AUC')
    plt.plot(hs_avg_auc['Max Leaves'], hs_avg_auc['AUC'], marker='o', linestyle='-', color='red', label="HSCART AUC")
    plt.xlabel("Number of Leaves")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.title(DATASET, fontsize=20)
    plt.legend()
    # plt.savefig("juvenile_class_au")
    plt.show()

    # Accuracy
    # plt.figure(figsize=(10, 6))
    # plt.plot(LEAVES, data_acc_hs, marker='o', linestyle='-', color='red', label='HS ACCURACY')
    # plt.plot(LEAVES, data_acc_dt, marker='o', linestyle='-', color='b', label='DT ACCURACY')
    # plt.xlabel('Number of Leaves')
    # plt.ylabel('Accuracy')
    # plt.grid(True)
    # plt.title(DATASET, fontsize=20)
    # plt.legend()
    # # plt.savefig("juvenile_class_acc")
    # plt.show()
