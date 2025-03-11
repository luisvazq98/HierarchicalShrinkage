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
    # print("Noisy Data Results:")
    # print(results_noisy.groupby('Model')[['AUC', 'Accuracy']].mean())
    # print("Denoised Data Results:")
    # print(results_denoised.groupby('Model')[['AUC', 'Accuracy']].mean())
    #
    # # Plot comparison. For multi-class problems, we are using AUC computed with multi_class='ovr'.
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # for dataset_results, title, ax in zip([results_noisy, results_denoised],
    #                                       ["Noisy Data", "Denoised Data"],
    #                                       ax.flatten()):
    #     cart_res = dataset_results[dataset_results['Model'] == 'CART'].groupby('Max Leaves')['AUC'].mean().reset_index()
    #     hs_res = dataset_results[dataset_results['Model'] == 'HSCART'].groupby('Max Leaves')['AUC'].mean().reset_index()
    #     ax.plot(cart_res['Max Leaves'], cart_res['AUC'], marker='o', linestyle='-', color='blue', label='CART AUC')
    #     ax.plot(hs_res['Max Leaves'], hs_res['AUC'], marker='o', linestyle='-', color='red', label='HSCART AUC')
    #     ax.set_title(title)
    #     ax.set_xlabel("Number of Leaves")
    #     ax.set_ylabel("AUC")
    #     ax.grid(True)
    #     ax.legend()
    # plt.tight_layout()
    # plt.title("student performance 49%")
    # plt.savefig("student performance_49")
    # plt.show()

    # Define noise levels as fractions (for 1%, 5%, 10%, 15%, 30%, 45%, 49%)
    # noise_levels = [0.01, 0.05, 0.10, 0.15, 0.30, 0.45, 0.49]
    #
    # # Dictionaries to store results for each noise level:
    # # For each condition (noisy, denoised) and each model, we store a DataFrame keyed by noise level.
    # results_by_noise_noisy = {'CART': {}, 'HSCART': {}}
    # results_by_noise_denoised = {'CART': {}, 'HSCART': {}}
    #
    # for noise in noise_levels:
    #     # ----- Experiment for Noisy Data -----
    #     Y_noisy, _ = introduce_label_noise(Y, noise_ratio=noise)
    #     results_noisy = training_models(X, Y_noisy, cart_hscart_estimators)
    #     # Group the results by Max Leaves for each model
    #     for model in ['CART', 'HSCART']:
    #         df = results_noisy[results_noisy['Model'] == model].groupby('Max Leaves')['AUC'].mean().reset_index()
    #         results_by_noise_noisy[model][noise] = df
    #
    #     # ----- Experiment for Denoised Data (after confident learning) -----
    #     # Split training and test sets based on the noisy labels
    #     train_x_full, _, train_y_noisy, _ = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    #     if SOURCE != 'imodels':
    #         train_x_full = train_x_full.reset_index(drop=True).to_numpy()
    #         train_y_noisy = train_y_noisy.reset_index(drop=True).to_numpy()
    #     train_y_noisy = train_y_noisy.astype(int)
    #
    #     # Train a base DecisionTreeClassifier for confident learning (Cleanlab)
    #     base_model = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)
    #     base_model.fit(train_x_full, train_y_noisy)
    #     probas = base_model.predict_proba(train_x_full)
    #     noise_indices_est = find_label_issues(labels=train_y_noisy, pred_probs=probas,
    #                                           return_indices_ranked_by='normalized_margin')
    #     mask = np.ones(len(train_y_noisy), dtype=bool)
    #     mask[noise_indices_est] = False
    #     train_x_denoised = train_x_full[mask]
    #     train_y_denoised = train_y_noisy[mask]
    #
    #     # Create a test set from the noisy data
    #     _, test_x, _, test_y = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
    #     if SOURCE != 'imodels':
    #         test_x = test_x.reset_index(drop=True).to_numpy()
    #         test_y = test_y.reset_index(drop=True).to_numpy()
    #
    #     results_denoised = training_models_denoised(train_x_denoised, train_y_denoised, test_x, test_y,
    #                                                 cart_hscart_estimators)
    #     for model in ['CART', 'HSCART']:
    #         df = results_denoised[results_denoised['Model'] == model].groupby('Max Leaves')['AUC'].mean().reset_index()
    #         results_by_noise_denoised[model][noise] = df
    #
    # # --------- Plotting ---------
    #
    # # Create two figures: one for Noisy Data and one for Denoised Data.
    # # In each figure, we have two subplots: left for CART and right for HSCART.
    # # In each subplot, each line (color/marker) corresponds to one noise level.
    #
    # # Colors and markers for distinguishing noise levels
    # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    # markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    #
    # # Plot for Noisy Data
    # fig1, axs1 = plt.subplots(ncols=2, figsize=(12, 6))
    # for idx, noise in enumerate(noise_levels):
    #     # For CART
    #     df_cart = results_by_noise_noisy['CART'][noise]
    #     axs1[0].plot(df_cart['Max Leaves'], df_cart['AUC'], marker=markers[idx],
    #                  color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
    #     axs1[0].set_title("CART - Noisy Data")
    #     axs1[0].set_xlabel("Max Leaves")
    #     axs1[0].set_ylabel("AUC")
    #     axs1[0].grid(True)
    #
    #     # For HSCART
    #     df_hs = results_by_noise_noisy['HSCART'][noise]
    #     axs1[1].plot(df_hs['Max Leaves'], df_hs['AUC'], marker=markers[idx],
    #                  color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
    #     axs1[1].set_title("HSCART - Noisy Data")
    #     axs1[1].set_xlabel("Max Leaves")
    #     axs1[1].set_ylabel("AUC")
    #     axs1[1].grid(True)
    #
    # # Create legends for the noisy data figure
    # axs1[0].legend(title="Noise Level")
    # axs1[1].legend(title="Noise Level")
    # fig1.suptitle("Credit Card on Noisy Data", fontsize=16)
    # fig1.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig("credit_card_noisy")
    # plt.show()
    #
    # # Plot for Denoised Data
    # fig2, axs2 = plt.subplots(ncols=2, figsize=(12, 6))
    # for idx, noise in enumerate(noise_levels):
    #     # For CART
    #     df_cart = results_by_noise_denoised['CART'][noise]
    #     axs2[0].plot(df_cart['Max Leaves'], df_cart['AUC'], marker=markers[idx],
    #                  color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
    #     axs2[0].set_title("CART - Denoised Data")
    #     axs2[0].set_xlabel("Max Leaves")
    #     axs2[0].set_ylabel("AUC")
    #     axs2[0].grid(True)
    #
    #     # For HSCART
    #     df_hs = results_by_noise_denoised['HSCART'][noise]
    #     axs2[1].plot(df_hs['Max Leaves'], df_hs['AUC'], marker=markers[idx],
    #                  color=colors[idx], linestyle='-', label=f"{int(noise * 100)}% noise")
    #     axs2[1].set_title("HSCART - Denoised Data")
    #     axs2[1].set_xlabel("Max Leaves")
    #     axs2[1].set_ylabel("AUC")
    #     axs2[1].grid(True)
    #
    # # Create legends for the denoised data figure
    # axs2[0].legend(title="Noise Level")
    # axs2[1].legend(title="Noise Level")
    # fig2.suptitle("Credit Card on Denoised Data", fontsize=16)
    # fig2.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig("credit_card_denoised")
    # plt.show()




    ###############################

    # Noise levels to test (as fractions)
    noise_levels_to_plot = [0.01, 0.15, 0.30]

    # Dictionaries to collect accuracy lists
    # For each model and each noise level, we store the list of accuracy values (from multiple splits)
    accuracies_noisy = {'CART': {}, 'HSCART': {}}
    accuracies_denoised = {'CART': {}, 'HSCART': {}}

    for noise in noise_levels_to_plot:
        # --- Noisy Data Experiment ---
        Y_noisy, _ = introduce_label_noise(Y, noise_ratio=noise)
        results_noisy = training_models(X, Y_noisy, cart_hscart_estimators)
        # Save accuracies for each model
        for model in ['CART', 'HSCART']:
            accs = results_noisy[results_noisy['Model'] == model]['Accuracy'].tolist()
            accuracies_noisy[model][noise] = accs

        # --- Denoised Data Experiment (after confident learning) ---
        # Split training data from noisy labels
        train_x_full, _, train_y_noisy, _ = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
        if SOURCE != 'imodels':
            train_x_full = train_x_full.reset_index(drop=True).to_numpy()
            train_y_noisy = train_y_noisy.reset_index(drop=True).to_numpy()
        train_y_noisy = train_y_noisy.astype(int)

        # Train a base classifier to flag label issues
        base_model = DecisionTreeClassifier(max_leaf_nodes=30, random_state=0)
        base_model.fit(train_x_full, train_y_noisy)
        probas = base_model.predict_proba(train_x_full)
        noise_indices_est = find_label_issues(labels=train_y_noisy, pred_probs=probas,
                                              return_indices_ranked_by='normalized_margin')
        mask = np.ones(len(train_y_noisy), dtype=bool)
        mask[noise_indices_est] = False
        train_x_denoised = train_x_full[mask]
        train_y_denoised = train_y_noisy[mask]

        # Create a test set from the noisy data
        _, test_x, _, test_y = train_test_split(X, Y_noisy, test_size=1 / 3, random_state=0)
        if SOURCE != 'imodels':
            test_x = test_x.reset_index(drop=True).to_numpy()
            test_y = test_y.reset_index(drop=True).to_numpy()

        results_denoised = training_models_denoised(train_x_denoised, train_y_denoised, test_x, test_y,
                                                    cart_hscart_estimators)
        for model in ['CART', 'HSCART']:
            accs = results_denoised[results_denoised['Model'] == model]['Accuracy'].tolist()
            accuracies_denoised[model][noise] = accs

    # ---------- Plotting Boxplots ----------

    # For each condition (noisy and denoised), we will create a figure with two subplots:
    # one for CART and one for HSCART. Each subplot will show boxplots for the three noise levels.

    # Define labels for the noise levels (as percentages)
    noise_labels = [f"{int(noise * 100)}%" for noise in noise_levels_to_plot]

    # --- Boxplots for Noisy Data ---
    fig_noisy, axs_noisy = plt.subplots(ncols=2, figsize=(12, 6))

    for idx, model in enumerate(['CART', 'HSCART']):
        # Collect accuracy data for each noise level for the current model
        data = [accuracies_noisy[model][noise] for noise in noise_levels_to_plot]
        axs_noisy[idx].boxplot(data, labels=noise_labels, patch_artist=True)
        axs_noisy[idx].set_title(f"{model} Accuracies on Noisy Data")
        axs_noisy[idx].set_xlabel("Noise Level")
        axs_noisy[idx].set_ylabel("Accuracy")
        axs_noisy[idx].grid(True)

    fig_noisy.suptitle("Boxplots of Model Accuracies (Noisy Data)")
    fig_noisy.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("student_performance_box_noisy")
    plt.show()

    # --- Boxplots for Denoised Data ---
    fig_denoised, axs_denoised = plt.subplots(ncols=2, figsize=(12, 6))

    for idx, model in enumerate(['CART', 'HSCART']):
        data = [accuracies_denoised[model][noise] for noise in noise_levels_to_plot]
        axs_denoised[idx].boxplot(data, labels=noise_labels, patch_artist=True)
        axs_denoised[idx].set_title(f"{model} Accuracies on Denoised Data")
        axs_denoised[idx].set_xlabel("Noise Level")
        axs_denoised[idx].set_ylabel("Accuracy")
        axs_denoised[idx].grid(True)

    fig_denoised.suptitle("Boxplots of Model Accuracies (Denoised Data)")
    fig_denoised.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("student_performance_box_denoised")
    plt.show()
