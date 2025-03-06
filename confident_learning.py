import pandas as pd
import numpy as np
import sys
import imodels as im
import time
import os
import json
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from imodels import HSTreeRegressorCV, HSTreeRegressor, HSTreeClassifier, HSTreeClassifierCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from imodels import get_clean_dataset
from cleanlab.filter import find_label_issues  # Cleanlab import for filtering noisy labels

sys.path.insert(0, "/Users/luisvazquez")
sys.path.insert(0, "/Users/luisvazquez/imodelsExperiments")
from imodelsExperiments.config.shrinkage.models import ESTIMATORS_CLASSIFICATION

######################## VARIABLES ########################
dataset_list = ['credit_card_clean', 'diabetes', 'breast_cancer', 'haberman', 'gait', 'student performance',
                'student dropout', 'titanic']
DATASET = "student performance"  # Example multi-class dataset
SOURCE = "kaggle"
PCA_VALUE = "no"
NOISE_RATIO = 0.49


######################## NOISE INJECTION ########################
def introduce_label_noise(y, noise_ratio=0.2, random_state=42):
    """
    For binary classification:
      Flips labels (0->1 and 1->0).

    For multi-class classification:
      Randomly replaces a fraction of labels with a different class.
    """
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


######################## DATA LOADING ########################
def get_classification_dataset(dataset, source):
    if source == 'imodels':
        x, y, features = get_clean_dataset(dataset, source, return_target_col_names=True)
        # If y is not numeric, encode it.
        if y.dtype == 'O' or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        return x, y, features
    elif source == 'uci':
        dataset_ids = {
            "breast cancer": 14,
            "haberman": 43,
            "gait": 760,
            "student dropout": 697
        }
        data = fetch_ucirepo(dataset_ids[DATASET])
        x = data.data.features
        y = data.data.targets
        y = y.drop(index=x[x.isna().any(axis=1)].index)
        x = x.dropna(axis=0)
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        if DATASET == "breast cancer":
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
            return x, y, None
        elif DATASET == "gait":
            x.dropna(columns=['condition'], inplace=True)
            y = x['condition']
            x = x.to_numpy()
            y = y.to_numpy()
            return x, y, None
        elif DATASET == "student dropout":
            categorical_columns = y.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                y.loc[:, col] = le.fit_transform(y[col])
            y = y.squeeze().astype(int).dropna()
            x = x.to_numpy()
            y = y.to_numpy()
            return x, y, None
        else:
            return x, y, None
    elif source == 'kaggle':
        dataset_paths = {
            "diabetes": {"filename": 'diabetes.csv', "path": "mathchi/diabetes-data-set"},
            "student performance": {"filename": 'Student_performance_data _.csv',
                                    "path": "rabieelkharoua/students-performance-dataset"},
        }
        path = kagglehub.dataset_download(dataset_paths[DATASET]['path'])
        file_path = os.path.join(path, dataset_paths[DATASET]['filename'])
        df = pd.read_csv(file_path)
        if DATASET == "diabetes":
            x = df.drop(columns=['Outcome'])
            y = df['Outcome']
            return x, y, None
        elif DATASET == "student performance":
            x = df.drop(columns=['GradeClass'])
            y = df['GradeClass']
            # Encode if necessary:
            # if y.dtype == 'O' or not np.issubdtype(y.dtype, np.number):
            #     le = LabelEncoder()
            #     y = le.fit_transform(y)
            return x, y, None
    else:
        # Fallback for other datasets such as 'adult income' or 'titanic'
        if dataset == 'adult income':
            ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                          'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income']
            data_adult = pd.read_csv(ADULT_URL, names=ADULT_COLS, sep=',\s', engine='python')
            data_adult.replace('?', np.nan, inplace=True)
            data_adult.dropna(inplace=True)
            x = data_adult.drop('income', axis=1)
            y = data_adult['income']
            categorical_columns = x.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col])
            return x, y, None
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
            return x, y, None


######################## TRAINING MODELS ########################
def training_models(x, y, models):
    results = []
    for i in range(0, 10):
        # Split the dataset
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1 / 3, random_state=i)

        # Convert to numpy arrays if not using imodels source
        if SOURCE != 'imodels':
            train_x = train_x.reset_index(drop=True).to_numpy()
            train_y = train_y.reset_index(drop=True).to_numpy()
            test_x = test_x.reset_index(drop=True).to_numpy()
            test_y = test_y.reset_index(drop=True).to_numpy()
            if PCA_VALUE == 'yes':
                pca = PCA(0.99)
                train_x = pca.fit_transform(train_x)
                test_x = pca.transform(test_x)

        for model_config in models:
            model_name = model_config.name
            model_class = model_config.cls
            model_kwargs = model_config.kwargs.copy()

            if model_name == "CART":
                model = model_class(**model_kwargs)
                start_time = time.time()
                model.fit(train_x, train_y)
                end_time = time.time()
                # Compute AUC differently for binary vs. multi-class
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
                    'Time (min)': (end_time - start_time) / 60,
                    'Split Seed': i
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
                    'Time (min)': (end_time - start_time) / 60,
                    'Split Seed': i
                })
    return pd.DataFrame(results)


######################## MAIN EXPERIMENT ########################
if __name__ == "__main__":
    # Get estimators for CART and HSCART
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in ['CART', 'HSCART']
    ]

    # Load the clean dataset first
    X, Y, features = get_classification_dataset(DATASET, SOURCE)

    # Introduce noise to the labels (works for both binary and multi-class)
    Y_noisy, noisy_indices = introduce_label_noise(Y, noise_ratio=NOISE_RATIO)
    print(f"Introduced noise in {len(noisy_indices)} out of {len(Y)} samples.")

    # -------------------------
    # Experiment Part 1: Train on Noisy Data
    # -------------------------
    print("Training on noisy data...")
    results_noisy = training_models(X, Y_noisy, cart_hscart_estimators)

    # -------------------------
    # Denoising Step: Use Cleanlab to filter out probable label issues from training data
    print("Performing denoising using Cleanlab...")
    train_x_full, _, train_y_noisy, _ = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    if SOURCE != 'imodels':
        train_x_full = train_x_full.reset_index(drop=True).to_numpy()
        train_y_noisy = train_y_noisy.reset_index(drop=True).to_numpy()

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


    # -------------------------
    # Experiment Part 2: Retrain Models on Denoised Data
    # -------------------------
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


    _, test_x, _, test_y = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    if SOURCE != 'imodels':
        test_x = test_x.reset_index(drop=True).to_numpy()
        test_y = test_y.reset_index(drop=True).to_numpy()

    print("Retraining on denoised data...")
    results_denoised = training_models_denoised(train_x_denoised, train_y_denoised, test_x, test_y,
                                                cart_hscart_estimators)

    # -------------------------
    # Print and Plot Comparison Metrics
    # -------------------------
    print("Noisy Data Results:")
    print(results_noisy.groupby('Model')[['AUC', 'Accuracy']].mean())
    print("Denoised Data Results:")
    print(results_denoised.groupby('Model')[['AUC', 'Accuracy']].mean())

    # Plot comparison. For multi-class problems, we are using AUC computed with multi_class='ovr'.
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for dataset_results, title, ax in zip([results_noisy, results_denoised],
                                          ["Noisy Data", "Denoised Data"],
                                          ax.flatten()):
        cart_res = dataset_results[dataset_results['Model'] == 'CART'].groupby('Max Leaves')['AUC'].mean().reset_index()
        hs_res = dataset_results[dataset_results['Model'] == 'HSCART'].groupby('Max Leaves')['AUC'].mean().reset_index()
        ax.plot(cart_res['Max Leaves'], cart_res['AUC'], marker='o', linestyle='-', color='blue', label='CART AUC')
        ax.plot(hs_res['Max Leaves'], hs_res['AUC'], marker='o', linestyle='-', color='red', label='HSCART AUC')
        ax.set_title(title)
        ax.set_xlabel("Number of Leaves")
        ax.set_ylabel("AUC")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.title("student performance 49%")
    plt.savefig("student performance_49")
    plt.show()