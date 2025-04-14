# region Libraries
import pandas as pd
import numpy as np
import sys
import imodels as im
import time
import os
import json
import kagglehub
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
import seaborn as sns
import networkx as nx
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
from cleanlab.filter import find_label_issues
sys.path.insert(0, "/Users/luisvazquez")
sys.path.insert(0, "/Users/luisvazquez/imodelsExperiments")
from imodelsExperiments.config.shrinkage.models import ESTIMATORS_CLASSIFICATION
# endregion

"""
-------------------------------------------------------------------------------------------- 
                                        VARIABLES 
--------------------------------------------------------------------------------------------
Variables used throughout code.

-  DATASET_DIC: Dictionary of dataset names and their respective datasets
*- DATASET: Name of desired dataset to test
-  SOURCE: Source of desired dataset to test
*- PCA_VALUE: If you want to apply PCA  
*- FILE_NAME: Name of excel file to save results
*- VANILLA_MODEL & HS_MODEL: 
        - CART 
        - HSCART
        - Random_Forest
        - HSEnsemble
-  NOISE_LEVELS: List of desired noise levels for confident learning exp. 
--------------------------------------------------------------------------------------------
"""

DATASET_DIC = {
    # These are HS paper datasets
    "breast cancer": {
        "source": 'uci',
        "id": 14
    },

    "haberman": {
        "source": 'uci',
        "id": 43
    },

    "diabetes": {
        "filename": 'diabetes.csv',
        "path": "mathchi/diabetes-data-set",
        "source": ''
    },

    # These are selected datasets for experiments
    "cifar": {
        "source": ''
    },

    "fashion minst": {
        "source": ''
    },

    "oxford pets": {
        "source": ''
    },

    "adult income": {
        "source": ''
    },

    "titanic": {
        "source": ''
    },

    "credit_card_clean": {
        "source": 'imodels'
    },

    "student dropout": {
        "source": 'uci',
        "id": 697
    },

    "student performance": {
        "filename": 'Student_performance_data _.csv',
        "path": "rabieelkharoua/students-performance-dataset",
        "source": 'kaggle'
    },

    "gait": {
        "source": 'uci',
        "id": 760
    },

    "musae": {
        "filename": '',
        "path": 'rozemberczki/musae-github-social-network',
        "source": 'kaggle'
    },

    "internet ads": {
        "filename": 'add.csv',
        "path": 'uciml/internet-advertisements-data-set',
        "source": 'kaggle'
    }
}
DATASET = "adult income"
SOURCE = DATASET_DIC[DATASET]['source']
PCA_VALUE = "no"
FILE_NAME = "RF_Results"
VANILLA_MODEL = "CART"
HS_MODEL = "HSCART"
NOISE_LEVELS = [0.01, 0.05, 0.10, 0.15, 0.30, 0.45, 0.49]

print(f"Dataset: {DATASET}\nSource: {SOURCE}\nPCA: {PCA_VALUE}")





"""
-------------------------------------------------------------------------------------------- 
                                        FUNCTIONS 
--------------------------------------------------------------------------------------------
Functions used. 
- MISCELLANEOUS FUNCTION: Background code
- get_regression_dataset: Getting regression datasets
- training_models: Trains models
- training_models_denoised: Use ONLY for confident learning exp. 
- confident_learning: Performs confident learning exp. (Needs some debugging!) 
- save_to_excel: Saves results to excel

--------------------------------------------------------------------------------------------
"""

#region MISCELLANEOUS FUNCTIONS

# ----------------------------------------------------------
# Another way to load the 20 models (10 CART, 10 HSCART)
# ----------------------------------------------------------
def get_models():
    LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    models = []
    for i in LEAVES:
        models.append(HSTreeClassifierCV(estimator_=DecisionTreeClassifier(max_leaf_nodes=i)))

    for i in LEAVES:
        models.append(DecisionTreeClassifier(max_leaf_nodes=i))

    return models

# ----------------------------------------------------------
# Functions for the GitHub MUSAE dataset
# ----------------------------------------------------------
def transform_features_to_sparse(table):
    table["weight"] = 1
    table = table.values.tolist()
    index_1 = [row[0] for row in table]
    index_2 = [row[1] for row in table]
    values = [row[2] for row in table]
    count_1, count_2 = max(index_1)+1, max(index_2)+1
    sp_m = sparse.csr_matrix(sparse.coo_matrix((values,(index_1,index_2)),shape=(count_1,count_2),dtype=np.float32))
    return sp_m
def normalize_adjacency(raw_edges):
    raw_edges_t = pd.DataFrame()
    raw_edges_t["id_1"] = raw_edges["id_2"]
    raw_edges_t["id_2"] = raw_edges["id_1"]
    raw_edges = pd.concat([raw_edges,raw_edges_t])
    edges = raw_edges.values.tolist()
    graph = nx.from_edgelist(edges)
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    A = transform_features_to_sparse(raw_edges)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs, (ind, ind)), shape=A.shape,dtype=np.float32))
    A = A.dot(degs)
    return A


# ----------------------------------------------------------
# Introducing noise into data for confident learning exp.
# ----------------------------------------------------------
def introduce_label_noise(y, noise_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    indices = np.random.choice(n_samples, size=n_noisy, replace=False)
    unique_labels = np.unique(y)

    for idx in indices:
        if len(unique_labels) == 2:
            # For binary classification, flip the label.
            y_noisy[idx] = 1 - y_noisy[idx]
        else:
            # For multi-class, randomly choose a label that is not the current one.
            possible_labels = [label for label in unique_labels if label != y[idx]]
            y_noisy[idx] = np.random.choice(possible_labels)
    return y_noisy, indices

#endregion

def get_regression_dataset(dataset):
    print("------- Getting regression dataset -------")
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
def get_classification_dataset(dataset, source):
    print("------- Getting classification dataset -------")
    if SOURCE == 'imodels':
        x, y, feature = get_clean_dataset(dataset, source, return_target_col_names=True)
        return x, y, feature

    elif SOURCE == 'uci':
        # Fetch the dataset using its corresponding ID
        data = fetch_ucirepo(id=DATASET_DIC[DATASET]['id'])
        x = data.data.features
        y = data.data.targets

        if y is None:
            y = pd.DataFrame()

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

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        elif DATASET == "gait":
            y = x['condition']
            x.drop(columns=['condition'], inplace=True)
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
            y.reset_index(inplace=True, drop=True)

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        else:
            x = x.to_numpy()
            y = y.to_numpy()
            return x, y

    elif SOURCE == 'kaggle':
        path = kagglehub.dataset_download(DATASET_DIC[DATASET]['path'])
        file_path = os.path.join(path, DATASET_DIC[DATASET]['filename'])

        if DATASET == "diabetes":
            df = pd.read_csv(file_path)
            x = df.drop(columns=['Outcome'])
            y = df['Outcome']

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        elif DATASET == "student performance":
            df = pd.read_csv(file_path)
            x = df.drop(columns=['GradeClass'])
            y = df['GradeClass']

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
        elif DATASET == "musae":
            edges_path = os.path.join(file_path, 'musae_git_edges.csv')
            features_path = os.path.join(file_path, 'musae_git_features.csv')
            target_path = os.path.join(file_path, 'musae_git_target.csv')

            # Dataframes
            edges = pd.read_csv(edges_path)
            features = pd.read_csv(features_path)
            target = pd.read_csv(target_path)

            y = np.array(target["ml_target"])
            A = normalize_adjacency(edges)
            X_sparse = transform_features_to_sparse(features)
            # X_tilde = A.dot(X)

            model = TruncatedSVD(n_components=16, random_state=0)
            W = model.fit_transform(X_sparse)
            model = TruncatedSVD(n_components=16, random_state=0)
            W_tilde = model.fit_transform(A)

            concatenated_features = np.concatenate([W, W_tilde], axis=1)

            return concatenated_features, y

        elif DATASET == 'internet ads':
            # Reading .csv file into dataframe
            df = pd.read_csv(file_path, low_memory=False)
            df = df[['0', '1', '2', '3', '1558']]

            # Setting missing values as NaN instead of '?'
            dfn = df.map(lambda x: np.nan if isinstance(x, str) and '?' in x else x)

            dfn['0'] = dfn['0'].astype(float)
            dfn['1'] = dfn['1'].astype(float)
            dfn['2'] = dfn['2'].astype(float)

            dfn.iloc[:, 0:3] = dfn.iloc[:, 0:3].fillna(dfn.iloc[:, 0:3].dropna().mean())
            dfn.dropna(inplace=True)
            dfn['3'] = dfn['3'].astype(int)
            #dfn.drop_duplicates(inplace=True)
            dfn.reset_index(drop=True, inplace=True)

            categorical_columns = dfn.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                dfn[col] = le.fit_transform(dfn[col])

            x = dfn.drop(columns=['1558'])
            y = dfn['1558']

            x = x.to_numpy()
            y = y.to_numpy()

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
            data_adult.reset_index(inplace=True, drop=True)

            # Separate features and target variable
            x = data_adult.drop('income', axis=1)
            y = data_adult['income']

            # Convert categorical variables to numerical using Label Encoding
            categorical_columns = x.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col])

            x = x.to_numpy()
            y = y.to_numpy()

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

            x = x.to_numpy()
            y = y.to_numpy()

            return x, y
def training_models(x, y, models):
    print("------- Starting to train model -------")
    results = []
    for i in range(0, 10):
        print(f"Currently on seed: {i}")
        if SOURCE == 'imodels':
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1/3, random_state=i)
        else:
            # Splitting dataset
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1/3, random_state=i)

            # Standard Scaler
            # scaler = StandardScaler()
            # train_x = scaler.fit_transform(train_x)
            # test_x = scaler.transform(test_x)

        # PCA
        if PCA_VALUE == 'yes':
            # Performing PCA ONLY on the training set. Then transforming the testing set
            pca = PCA(0.99)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)


        for model_config in models:
            model_name = model_config.name
            model_class = model_config.cls
            model_kwargs = model_config.kwargs.copy()

            if model_name == VANILLA_MODEL:
                # Training model
                cart_model = model_class(**model_kwargs)
                start_time = time.time()
                cart_model.fit(train_x, train_y)
                end_time = time.time()

                if len(np.unique(test_y)) == 2:
                    y_pred_proba = cart_model.predict_proba(test_x)[:, 1]
                    auc_cart = roc_auc_score(test_y, y_pred_proba)
                else:
                    y_pred_proba = cart_model.predict_proba(test_x)
                    auc_cart = roc_auc_score(test_y, y_pred_proba, multi_class='ovr', average='macro')

                # Getting metrics
                predictions = cart_model.predict(test_x)
                accuracy = accuracy_score(test_y, predictions)

                # Append CART results
                results.append({
                    'Model': VANILLA_MODEL,
                    'Max Leaves': model_kwargs['max_leaf_nodes'],
                    'Max Depth': cart_model.tree_.max_depth,
                    'Node Count': cart_model.tree_.node_count,
                    'Lambda': None,
                    'AUC': auc_cart,
                    'Accuracy': accuracy,
                    'Time (min)': (end_time - start_time) / 60,
                    'Split Seed': i
                })
            elif model_name == HS_MODEL:
                # Training model
                hs_model = model_class(**model_kwargs)
                start_time = time.time()
                hs_model.fit(train_x, train_y)
                end_time = time.time()

                if len(np.unique(test_y)) == 2:
                    y_pred_proba = hs_model.predict_proba(test_x)[:, 1]
                    auc_hscart = roc_auc_score(test_y, y_pred_proba)
                else:
                    y_pred_proba = hs_model.predict_proba(test_x)
                    auc_hscart = roc_auc_score(test_y, y_pred_proba, multi_class='ovr', average='macro')

                # Getting metrics
                predictions = hs_model.predict(test_x)
                accuracy = accuracy_score(test_y, predictions)

                # Append HSCART results
                results.append({
                    'Model': HS_MODEL,
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
    print("------- Finished training model -------")
    return model_results
#region CONFIDENT LEARNING
def training_models_denoised(train_x, train_y, test_x, test_y, models):
    results = []
    for model_config in models:
        model_name = model_config.name
        model_class = model_config.cls
        model_kwargs = model_config.kwargs.copy()

        if model_name == "CART":
            model = model_class(**model_kwargs)
            start_time = time.time()
            model.fit(train_x, train_y)
            end_time = time.time()
            if len(np.unique(test_y)) == 2:
                y_pred_proba = model.predict_proba(test_x)[:, 1]
                auc = roc_auc_score(test_y, y_pred_proba)
            else:
                y_pred_proba = model.predict_proba(test_x)
                auc = roc_auc_score(test_y, y_pred_proba, multi_class='ovr', average='macro')
            predictions = model.predict(test_x)
            accuracy = accuracy_score(test_y, predictions)
            results.append({
                'Model': 'CART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],
                'Max Depth': model.tree_.max_depth,
                'Node Count': model.tree_.node_count,
                'Lambda': None,
                'AUC': auc,
                'Accuracy': accuracy,
                'Time (min)': (end_time - start_time) / 60
            })
        elif model_name == "HSCART":
            model = model_class(**model_kwargs)
            start_time = time.time()
            model.fit(train_x, train_y)
            end_time = time.time()
            if len(np.unique(test_y)) == 2:
                y_pred_proba = model.predict_proba(test_x)[:, 1]
                auc = roc_auc_score(test_y, y_pred_proba)
            else:
                y_pred_proba = model.predict_proba(test_x)
                auc = roc_auc_score(test_y, y_pred_proba, multi_class='ovr', average='macro')
            predictions = model.predict(test_x)
            accuracy = accuracy_score(test_y, predictions)
            results.append({
                'Model': 'HSCART',
                'Max Leaves': model_kwargs['max_leaf_nodes'],
                'Max Depth': model.estimator_.tree_.max_depth,
                'Node Count': model.estimator_.tree_.node_count,
                'Lambda': model.reg_param,
                'AUC': auc,
                'Accuracy': accuracy,
                'Time (min)': (end_time - start_time) / 60
            })
    return pd.DataFrame(results)
def confident_learning(x, y, models):

    results_by_noise_noisy = {'CART': {}, 'HSCART': {}}
    results_by_noise_denoised = {'CART': {}, 'HSCART': {}}


    for noise in NOISE_LEVELS:
        # -------------------------------------------------------------------------------------
        # Experiment 1: Train on Noisy Data
        # -------------------------------------------------------------------------------------
        print("------- Starting Confident Learning Experiments -------")
        y_noisy, noisy_indices = introduce_label_noise(y, noise_ratio=noise)
        print(f"Introduced noise in {len(noisy_indices)} out of {len(Y)} samples.")
        print("Training on noisy data...")
        results_noisy = training_models(X, y_noisy, cart_hscart_estimators)

        # Group the results by Max Leaves for each model (using Accuracy)
        for model in ['CART', 'HSCART']:
            df = results_noisy[results_noisy['Model'] == model].groupby('Max Leaves')['Accuracy'].mean().reset_index()
            results_by_noise_noisy[model][noise] = df

        # -------------------------------------------------------------------------------------
        # Intermediate Step: Use Cleanlab to filter out probable label issues.
        # -------------------------------------------------------------------------------------
        print("Performing denoising using Cleanlab...")
        train_x_full, _, train_y_noisy, _ = train_test_split(x, y_noisy, test_size=1/3, random_state=0)
        # if SOURCE != 'imodels':
        #     train_x_full = train_x_full.reset_index(drop=True).to_numpy()
        #     train_y_noisy = train_y_noisy.reset_index(drop=True).to_numpy()

        # Convert labels to integers for Cleanlab
        train_y_noisy = train_y_noisy.astype(int)

        # Train a simple DecisionTreeClassifier as a base model
        base_model = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)
        base_model.fit(train_x_full, train_y_noisy)
        probas = base_model.predict_proba(train_x_full)

        # Identify mislabeled indices using Cleanlab
        noise_indices_est = find_label_issues(labels=train_y_noisy, pred_probs=probas,
                                              return_indices_ranked_by='normalized_margin')
        print(f"Cleanlab flagged {len(noise_indices_est)} samples as potential label issues.")

        # Create a denoised version of training data by removing flagged samples
        mask = np.ones(len(train_y_noisy), dtype=bool)
        mask[noise_indices_est] = False
        train_x_denoised = train_x_full[mask]
        train_y_denoised = train_y_noisy[mask]


        # -------------------------------------------------------------------------------------
        # Experiment 2: Train on Denoised Data
        # -------------------------------------------------------------------------------------
        _, test_x, _, test_y = train_test_split(x, y_noisy, test_size=1 / 3, random_state=0)
        # if SOURCE != 'imodels':
        #     test_x = test_x.reset_index(drop=True).to_numpy()
        #     test_y = test_y.reset_index(drop=True).to_numpy()

        print("Retraining on denoised data...")
        results_denoised = training_models_denoised(train_x_denoised, train_y_denoised, test_x, test_y,
                                                    cart_hscart_estimators)

        for model in ['CART', 'HSCART']:
            df = results_denoised[results_denoised['Model'] == model].groupby('Max Leaves')['Accuracy'].mean().reset_index()
            results_by_noise_denoised[model][noise] = df


    return results_by_noise_noisy, results_by_noise_denoised
def plot_noise(results_by_noise_noisy, results_by_noise_denoised):
    # --------- Plotting ---------
    # Create two figures: one for Noisy Data and one for Denoised Data.
    # In each figure, we have two subplots: left for CART and right for HSCART.
    # In each subplot, each line (color/marker) corresponds to one noise level.

    # Colors and markers for distinguishing noise levels
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    # Plot for Noisy Data
    fig1, axs1 = plt.subplots(ncols=2, figsize=(12, 6))
    for idx, noise in enumerate(NOISE_LEVELS):
        # For CART
        df_cart = results_by_noise_noisy['CART'][noise]
        axs1[0].plot(df_cart['Max Leaves'], df_cart['Accuracy'], marker=markers[idx],
                     color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
        axs1[0].set_title("CART - Noisy Data", fontsize=25)
        axs1[0].set_xlabel("Max Leaves", fontsize=20)
        axs1[0].set_ylabel("Accuracy", fontsize=20)
        axs1[0].grid(True)

        # For HSCART
        df_hs = results_by_noise_noisy['HSCART'][noise]
        axs1[1].plot(df_hs['Max Leaves'], df_hs['Accuracy'], marker=markers[idx],
                     color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
        axs1[1].set_title("HSCART - Noisy Data", fontsize=25)
        axs1[1].set_xlabel("Max Leaves", fontsize=20)
        axs1[1].set_ylabel("Accuracy", fontsize=20)
        axs1[1].grid(True)

    # Create legends for the noisy data figure
    axs1[0].legend(title="Noise Level", loc='center')
    axs1[1].legend(title="Noise Level")
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("student_noisy_acc")
    plt.show()

    # Plot for Denoised Data
    fig2, axs2 = plt.subplots(ncols=2, figsize=(12, 6))
    for idx, noise in enumerate(NOISE_LEVELS):
        # For CART
        df_cart = results_by_noise_denoised['CART'][noise]
        axs2[0].plot(df_cart['Max Leaves'], df_cart['Accuracy'], marker=markers[idx],
                     color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
        axs2[0].set_title("CART - Denoised Data", fontsize=25)
        axs2[0].set_xlabel("Max Leaves", fontsize=20)
        axs2[0].set_ylabel("Accuracy", fontsize=20)
        axs2[0].grid(True)

        # For HSCART
        df_hs = results_by_noise_denoised['HSCART'][noise]
        axs2[1].plot(df_hs['Max Leaves'], df_hs['Accuracy'], marker=markers[idx],
                     color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
        axs2[1].set_title("HSCART - Denoised Data", fontsize=25)
        axs2[1].set_xlabel("Max Leaves", fontsize=20)
        axs2[1].set_ylabel("Accuracy", fontsize=20)
        axs2[1].grid(True)

    # Create legends for the denoised data figure
    axs2[0].legend(title="Noise Level")
    axs2[1].legend(title="Noise Level")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig("student_denoised_acc")
    plt.show()
#endregion
def save_to_excel(cart_acc, hs_acc, std_cart, std_hs):
    if os.path.exists(FILE_NAME):
        if PCA_VALUE == 'no':
            existing_df = pd.read_excel(f'{FILE_NAME}.xlsx')
            df = existing_df.copy()
            dataset_name = DATASET

            # Check if the row for this dataset already exists. (In
            if dataset_name in df['DATASET'].values:
                # Update the existing row with new PCA values.
                df.loc[df['DATASET'] == dataset_name, ['DT', 'HS-DT']] = [f"{cart_acc:.2f} ({std_cart:.4f})",
                                                                          f"{hs_acc:.2f} ({std_hs:.4f})"]
            else:
                # If the row doesn't exist, create it. You can also include other values as needed.
                new_row = {
                    'DATASET': dataset_name,
                    'DT': f"{cart_acc:.2f} ({std_cart:.4f})",
                    'HS-DT': f"{hs_acc:.2f} ({std_hs:.4f})",
                    'PCA-DT': [""],
                    'PCA-HS-DT': [""]
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated DataFrame back to the Excel file.
            df.to_excel(f'{FILE_NAME}.xlsx', index=False)

        elif PCA_VALUE == 'yes':
            existing_df = pd.read_excel(f'{FILE_NAME}.xlsx')
            df = existing_df.copy()
            dataset_name = DATASET

            # Check if the row for this dataset already exists.
            if dataset_name in df['DATASET'].values:
                # Update the existing row with new PCA values.
                df.loc[df['DATASET'] == dataset_name, ['PCA-DT', 'PCA-HS-DT']] = [f"{cart_acc:.2f} ({std_cart:.4f})",
                                                                                  f"{hs_acc:.2f} ({std_hs:.4f})"]
            else:
                # If the row doesn't exist, create it. You can also include other values as needed.
                new_row = {
                    'DATASET': dataset_name,
                    'DT': [""],  # or provide a value if available
                    'HS-DT': [""],  # or provide a value if available
                    'PCA-DT': [f"{cart_acc:.2f} ({std_cart:.4f})"],
                    'PCA-HS-DT': [f"{hs_acc:.2f} ({std_hs:.4f})"]
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated DataFrame back to the Excel file.
            df.to_excel(f'{FILE_NAME}.xlsx', index=False)

    else:
        if PCA_VALUE == 'no':
            # Read existing data
            try:
                existing_df = pd.read_excel(f'{FILE_NAME}.xlsx')
            except FileNotFoundError:
                existing_df = pd.DataFrame(columns=['DATASET', 'DT', 'HS-DT', 'PCA-DT', 'PCA-HS-DT'])

            df = existing_df.copy()
            # Specify your dataset name and new values.
            dataset_name = DATASET

            # Check if the row for this dataset already exists.
            if dataset_name in df['DATASET'].values:
                # Update the existing row with new PCA values.
                df.loc[df['DATASET'] == dataset_name, ['DT', 'HS-DT']] = [f"{cart_acc:.2f} ({std_cart:.4f})",
                                                                          f"{hs_acc:.2f} ({std_hs:.4f})"]
            else:
                # If the row doesn't exist, create it. You can also include other values as needed.
                new_row = {
                    'DATASET': dataset_name,
                    'DT': f"{cart_acc:.2f} ({std_cart:.4f})",
                    'HS-DT': f"{hs_acc:.2f} ({std_hs:.4f})",
                    'PCA-DT': [""],
                    'PCA-HS-DT': [""]
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated DataFrame back to the Excel file.
            df.to_excel(f'{FILE_NAME}.xlsx', index=False)

        elif PCA_VALUE == 'yes':
            # Read existing data
            try:
                existing_df = pd.read_excel(f'{FILE_NAME}.xlsx')
            except FileNotFoundError:
                existing_df = pd.DataFrame(columns=['DATASET', 'DT', 'HS-DT', 'PCA-DT', 'PCA-HS-DT'])

            df = existing_df.copy()

            # Specify your dataset name and new values.
            dataset_name = DATASET

            # Check if the row for this dataset already exists.
            if dataset_name in df['DATASET'].values:
                # Update the existing row with new PCA values.
                df.loc[df['DATASET'] == dataset_name, ['PCA-DT', 'PCA-HS-DT']] = [f"{cart_acc:.2f} ({std_cart:.4f})",
                                                                                  f"{hs_acc:.2f} ({std_hs:.4f})"]
            else:
                # If the row doesn't exist, create it. You can also include other values as needed.
                new_row = {
                    'DATASET': dataset_name,
                    'DT': [""],  # or provide a value if available
                    'HS-DT': [""],  # or provide a value if available
                    'PCA-DT': [f"{cart_acc:.2f} ({std_cart:.4f})"],
                    'PCA-HS-DT': [f"{hs_acc:.2f} ({std_hs:.4f})"]
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated DataFrame back to the Excel file.
            df.to_excel(f'{FILE_NAME}.xlsx', index=False)

    print("Finished saving results to excel file!")

def plot(cart, hscart, metric):
    # Group by 'Max Leaves' and calculate the average AUC
    cart_avg_auc = cart.groupby('Max Leaves')[metric].mean().reset_index()
    hs_avg_auc = hscart.groupby('Max Leaves')[metric].mean().reset_index()

    # Group by 'Max Leaves' and calculate the average Accuracy
    cart_avg_acc = cart.groupby('Max Leaves')['Accuracy'].mean().reset_index()
    hs_avg_acc = hscart.groupby('Max Leaves')['Accuracy'].mean().reset_index()

    # AUC Score
    plt.figure(figsize=(10, 6))
    plt.plot(cart_avg_auc['Max Leaves'], cart_avg_auc[metric], marker='o', linestyle='-', color='blue',
             label='CART AUC')
    plt.plot(hs_avg_auc['Max Leaves'], hs_avg_auc[metric], marker='o', linestyle='-', color='red',
             label="HSCART AUC")
    plt.xlabel("Number of Leaves")
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.title(DATASET.upper(), fontsize=20)
    plt.legend()
    # plt.savefig(f"{DATASET.upper()}_AUC.png", bbox_inches='tight')
    plt.show()
def violin_plot_cart_max_depth(cart):
    """
    Creates a violin plot showing the distribution of the maximum depth
    for the CART model from the provided results DataFrame.

    Parameters:
        cart (pd.DataFrame): DataFrame containing CART model results with a column 'Max Depth'.
    """
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=cart['Max Depth'])
    plt.title("Violin Plot of CART Model Max Depth")
    plt.grid(True)
    plt.ylabel("Max Depth")
    plt.xlabel("")
    plt.show()


"""
-------------------------------------------------------------------------------------------- 
                                        MAIN 
--------------------------------------------------------------------------------------------                   
Main function. 
--------------------------------------------------------------------------------------------
"""
if __name__ == "__main__":
    ######################## MODELS ########################
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in [VANILLA_MODEL, HS_MODEL]
    ]
    # Use the following function only if above does not work
    # models = get_models()

    ######################## DATASET ########################
    #X, Y = get_regression_dataset(DATASET)
    X, Y, *extra = get_classification_dataset(DATASET, SOURCE)
    # features = extra[0] if extra else None

    ######################## CONFIDENT LEARNING ########################
    # results1, results2 = confident_learning(X, Y, cart_hscart_estimators)
    # plot_noise(results1, results2)

    ######################## TRAINING MODELS ########################
    results_df = training_models(X, Y, cart_hscart_estimators)

    ######################## SEPARATING MODELS USED  ########################
    cart = results_df[results_df['Model']==VANILLA_MODEL]
    hscart = results_df[results_df['Model']==HS_MODEL]

    ######################## PLOT DESIRED METRIC  ########################
    plot(cart, hscart, 'AUC')
    violin_plot_cart_max_depth(cart)



    ######################## GETTING MAX ACCURACY AND CORRESPONDING STD  ########################
    # Group by 'Max Leaves' and calculate the highest accuracy
    cart_max_acc = cart.groupby('Max Leaves')['Accuracy'].mean().max()
    hs_max_acc = hscart.groupby('Max Leaves')['Accuracy'].mean().max()
    print("CART Accuracy:", (cart_max_acc*100))
    print("HSCART Accuracy:", (hs_max_acc*100))


    # Group by 'Max Leaves' and calculate the STD for the group that corresponds with the highest accuracy
    cart_mean_series = cart.groupby('Max Leaves')['Accuracy'].mean()
    cart_best_group = cart_mean_series.idxmax()
    cart_std = cart[cart['Max Leaves'] == cart_best_group]['Accuracy'].std()

    hs_mean_series = hscart.groupby('Max Leaves')['Accuracy'].mean()
    hs_best_group = hs_mean_series.idxmax()
    hs_std = hscart[hscart['Max Leaves'] == hs_best_group]['Accuracy'].std()

    # Writing to Excel file
    save_to_excel((cart_max_acc*100), (hs_max_acc*100), cart_std, hs_std)
