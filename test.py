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
sys.path.insert(0, "/Users/luisvazquez")
sys.path.insert(0, "/Users/luisvazquez/imodelsExperiments")
from imodelsExperiments.config.shrinkage.models import ESTIMATORS_CLASSIFICATION
# endregion


######################## VARIABLES ########################
DATASET_DIC = {
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

VANILLA_MODEL = "CART"
HS_MODEL = "HSCART"


######################## MISCELLANEOUS FUNCTIONS ########################
#region MISCELLANEOUS FUNCTIONS

def get_models():
    """Another way to load the 20 models (10 CART, 10 HSCART) if needed."""
    LEAVES = np.array([2, 4, 8, 12, 15, 20, 24, 28, 30, 32])
    models = []
    for i in LEAVES:
        models.append(HSTreeClassifierCV(estimator_=DecisionTreeClassifier(max_leaf_nodes=i)))

    for i in LEAVES:
        models.append(DecisionTreeClassifier(max_leaf_nodes=i))

    return models


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

#endregion


######################## REGRESSION DATASETS ########################
def get_regression_dataset(dataset):
    print("------- Getting regression dataset -------")
    if dataset == "diabetes":
        diabetes = datasets.load_diabetes()
        x = diabetes.data
        y = diabetes.target
        return x, y
    elif dataset == "red wine":
        path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
        file_path = os.path.join(path, "winequality-red.csv")
        df = pd.read_csv(file_path)
        x = df.drop(columns=['quality'])
        y = df['quality']
        return x, y
    else:
        print("Error. Please enter a valid dataset")


######################## CLASSIFICATION DATASETS ########################
def get_classification_dataset(dataset, source, dataset_dic):
    """Loads classification data for the given dataset."""
    print(f"------- Getting classification dataset: {dataset} -------")
    if source == 'imodels':
        x, y, feature = get_clean_dataset(dataset, source, return_target_col_names=True)
        return x, y, feature

    elif source == 'uci':
        data = fetch_ucirepo(id=dataset_dic[dataset]['id'])
        x = data.data.features
        y = data.data.targets

        if y is None:
            y = pd.DataFrame()

        # Dropping rows with NaN values
        y = y.drop(index=x[x.isna().any(axis=1)].index)
        x = x.dropna(axis=0)

        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if dataset == "breast cancer":
            categorical_columns = x.select_dtypes(include=["object"]).columns
            label_encoders = {}

            for col in categorical_columns:
                le = OneHotEncoder(sparse_output=False)
                transformed = le.fit_transform(x[col].values.reshape(-1, 1))
                transformed_df = pd.DataFrame(transformed, columns=le.categories_[0])
                x = x.drop(columns=[col]).join(transformed_df, lsuffix='_left', rsuffix='_right')
                label_encoders[col] = le

            y_label_encoder = LabelEncoder()
            y = y_label_encoder.fit_transform(y)
            x = x.to_numpy()
            y = y.to_numpy()
            return x, y

        elif dataset == "gait":
            y = x['condition']
            x.drop(columns=['condition'], inplace=True)
            x = x.to_numpy()
            y = y.to_numpy()
            return x, y

        elif dataset == "student dropout":
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

    elif source == 'kaggle':
        path = kagglehub.dataset_download(dataset_dic[dataset]['path'])
        file_path = os.path.join(path, dataset_dic[dataset]['filename'])

        if dataset == "diabetes":
            df = pd.read_csv(file_path)
            x = df.drop(columns=['Outcome'])
            y = df['Outcome']
            return x.to_numpy(), y.to_numpy()

        elif dataset == "student performance":
            df = pd.read_csv(file_path)
            x = df.drop(columns=['GradeClass'])
            y = df['GradeClass']
            return x.to_numpy(), y.to_numpy()

        elif dataset == "musae":
            edges_path = os.path.join(file_path, 'musae_git_edges.csv')
            features_path = os.path.join(file_path, 'musae_git_features.csv')
            target_path = os.path.join(file_path, 'musae_git_target.csv')

            edges = pd.read_csv(edges_path)
            features = pd.read_csv(features_path)
            target = pd.read_csv(target_path)

            y = np.array(target["ml_target"])
            A = normalize_adjacency(edges)
            X_sparse = transform_features_to_sparse(features)

            model = TruncatedSVD(n_components=16, random_state=0)
            W = model.fit_transform(X_sparse)

            model = TruncatedSVD(n_components=16, random_state=0)
            W_tilde = model.fit_transform(A)

            concatenated_features = np.concatenate([W, W_tilde], axis=1)
            return concatenated_features, y

        elif dataset == 'internet ads':
            df = pd.read_csv(file_path, low_memory=False)
            df = df[['0', '1', '2', '3', '1558']]

            dfn = df.map(lambda x: np.nan if isinstance(x, str) and '?' in x else x)
            dfn['0'] = dfn['0'].astype(float)
            dfn['1'] = dfn['1'].astype(float)
            dfn['2'] = dfn['2'].astype(float)

            dfn.iloc[:, 0:3] = dfn.iloc[:, 0:3].fillna(dfn.iloc[:, 0:3].dropna().mean())
            dfn.dropna(inplace=True)
            dfn['3'] = dfn['3'].astype(int)
            dfn.reset_index(drop=True, inplace=True)

            categorical_columns = dfn.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                dfn[col] = le.fit_transform(dfn[col])

            x = dfn.drop(columns=['1558'])
            y = dfn['1558']
            return x.to_numpy(), y.to_numpy()

    else:
        # If source == '' or something else
        if dataset == 'adult income':
            ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'sex',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income']
            data_adult = pd.read_csv(ADULT_URL, names=ADULT_COLS, sep=',\s', engine='python')
            data_adult.replace('?', np.nan, inplace=True)
            data_adult.dropna(inplace=True)
            data_adult.reset_index(inplace=True, drop=True)

            x = data_adult.drop('income', axis=1)
            y = data_adult['income']

            categorical_columns = x.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col])

            return x.to_numpy(), y.to_numpy()

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
            return x.to_numpy(), y.to_numpy()


######################## TRAINING MODELS ########################
def training_models(x, y, models, source, pca_value="no"):
    print("------- Starting to train model -------")
    results = []
    for i in range(0, 10):
        print(f"Currently on seed: {i}")
        # Splitting dataset
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1/3, random_state=i)

        # Optional PCA
        if pca_value == 'yes':
            pca = PCA(0.99)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)

        for model_config in models:
            model_name = model_config.name
            model_class = model_config.cls
            model_kwargs = model_config.kwargs.copy()

            if model_name == VANILLA_MODEL:
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

                predictions = cart_model.predict(test_x)
                accuracy = accuracy_score(test_y, predictions)

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

                predictions = hs_model.predict(test_x)
                accuracy = accuracy_score(test_y, predictions)

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


######################## MAIN SCRIPT ########################
if __name__ == "__main__":

    # 1) Define the datasets you want to loop over
    dataset_names = [
        "adult income",
        "titanic",
        "credit_card_clean",
        "student dropout",
        "student performance",
        "gait",
        "musae",
        "internet ads"
    ]

    # 2) We'll choose whether to do PCA or not, and which model(s) we want to keep
    PCA_VALUE = "no"  # or "yes"
    chosen_model_for_plot = "CART"  # or "HSCART"

    # 3) Load the estimators you want from your config
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in [VANILLA_MODEL, HS_MODEL]
    ]

    # 4) Collect results across all datasets
    combined_results = []

    for ds in dataset_names:
        print(f"\n\n======================== DATASET: {ds} ========================")
        SOURCE = DATASET_DIC[ds]['source']

        # -- Load data (Classification in this example) --
        data = get_classification_dataset(ds, SOURCE, DATASET_DIC)
        if len(data) == 3:
            X, Y, _ = data
        else:
            X, Y = data

        # -- Train models --
        results_df = training_models(X, Y, cart_hscart_estimators, source=SOURCE, pca_value=PCA_VALUE)

        # -- Add a 'Dataset' column so we know which dataset the results came from
        results_df['Dataset'] = ds

        # -- Collect
        combined_results.append(results_df)

    # 5) Concatenate all dataset results into a single DataFrame
    all_results = pd.concat(combined_results, ignore_index=True)

    # 6) Filter the DataFrame to keep only the chosen model (CART or HSCART)
    model_results = all_results[all_results["Model"] == chosen_model_for_plot]

    # 7) Create a violin plot of "Max Depth" vs "Dataset"
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=model_results,
                   x="Dataset",
                   y="Max Depth",
                   inner="box",
                   color="skyblue")

    plt.xticks(rotation=45, ha='right')
    plt.title(f"Violin Plot of Maximum Depth for {chosen_model_for_plot.upper()}", fontsize=16)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Max Depth", fontsize=12)
    plt.tight_layout()
    plt.show()
