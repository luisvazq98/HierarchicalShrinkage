import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import plot_tree
from ucimlrepo import fetch_ucirepo
import kagglehub
from config.shrinkage.models import ESTIMATORS_CLASSIFICATION

# Function to split datasets
def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# Function to load and preprocess the Gait dataset
def load_gait_dataset():
    multivariate_gait_data = fetch_ucirepo(id=760)
    X = multivariate_gait_data.data.features
    y = multivariate_gait_data.data.targets['condition']
    return X.drop(columns=['condition']).to_numpy(), y.to_numpy()

# Function to load and preprocess the Student Performance dataset
def load_student_performance_dataset():
    dataset_path = kagglehub.dataset_download("rabieelkharoua/students-performance-dataset")
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset folder.")
    csv_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(csv_path)
    return df.drop(columns=['GradeClass']).to_numpy(), df['GradeClass'].to_numpy()

# Function to load and preprocess the Student Dropout dataset
def load_student_dropout_dataset():
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)
    X = predict_students_dropout_and_academic_success.data.features
    y = predict_students_dropout_and_academic_success.data.targets
    categorical_columns = y.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        y.loc[:, col] = le.fit_transform(y[col])
    y = y.squeeze().astype(int).dropna()
    return X.to_numpy(), y.to_numpy()

# Function to train and evaluate models
def train_and_evaluate_models(models, x_train, y_train, x_test, y_test, dataset_name):
    results = []
    for model_config in models:
        model_name = model_config.name
        model_class = model_config.cls
        model_kwargs = model_config.kwargs.copy()

        model = model_class(**model_kwargs)
        start_time = time.time()
        model.fit(x_train, y_train)
        end_time = time.time()

        y_pred_proba = model.predict_proba(x_test)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        result = {
            'Dataset': dataset_name,
            'Model': model_name,
            'Max Leaves': model_kwargs.get('max_leaf_nodes', None),
            'Lambda': getattr(model, 'reg_param', None),
            'Accuracy': accuracy,
            'Training Time (s)': end_time - start_time,
            'Split Seed': "na"
        }

        if model_name == 'CART':
            result['Tree Size'] = model.tree_.node_count
            result['Tree Depth'] = model.tree_.max_depth

        results.append(result)

    return pd.DataFrame(results)

# Function to plot decision trees
def plot_decision_tree(model, feature_names, title, filename=None):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=feature_names)
    plt.title(title, fontsize=35)
    if filename:
        plt.savefig(filename)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load datasets
    x_gait, y_gait = load_gait_dataset()
    x_student_perf, y_student_perf = load_student_performance_dataset()
    x_student_dropout, y_student_dropout = load_student_dropout_dataset()

    # Split datasets
    x_train_gait, x_test_gait, y_train_gait, y_test_gait = split_data(x_gait, y_gait)
    x_train_student_perf, x_test_student_perf, y_train_student_perf, y_test_student_perf = split_data(x_student_perf, y_student_perf)
    x_train_student_dropout, x_test_student_dropout, y_train_student_dropout, y_test_student_dropout = split_data(x_student_dropout, y_student_dropout)

    # Initialize models
    cart_hscart_estimators = [
        model for model_group in ESTIMATORS_CLASSIFICATION
        for model in model_group
        if model.name in ['CART', 'HSCART']
    ]

    # Train and evaluate models
    results_gait = train_and_evaluate_models(cart_hscart_estimators, x_train_gait, y_train_gait, x_test_gait, y_test_gait, "Gait")
    results_student_perf = train_and_evaluate_models(cart_hscart_estimators, x_train_student_perf, y_train_student_perf, x_test_student_perf, y_test_student_perf, "Student Performance")
    results_student_dropout = train_and_evaluate_models(cart_hscart_estimators, x_train_student_dropout, y_train_student_dropout, x_test_student_dropout, y_test_student_dropout, "Student Dropout")

    # Combine results
    results_df = pd.concat([results_gait, results_student_perf, results_student_dropout], ignore_index=True)

    # Plot decision trees
    plot_decision_tree(cart_hscart_estimators[0].cls(**cart_hscart_estimators[0].kwargs), features, "CART: Gait", 'CART_gait.png')
    plot_decision_tree(cart_hscart_estimators[1].cls(**cart_hscart_estimators[1].kwargs), features, "HSCART: Gait", 'HSCART_gait.png')