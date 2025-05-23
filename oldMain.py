###################################################################
#
# LIBRARIES
#
###################################################################
import csv
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, fashion_mnist
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from imodels import HSTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time
import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################
#
# URL'S FOR DATASETS
#
###################################################################
TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
CREDIT_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

# OXFORD_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
# OXFORD_DIR = tf.keras.utils.get_file('oxford_pets', origin=OXFORD_URL, extract=True)
# OXFORD_DIR = pathlib.Path(OXFORD_DIR).parent / "images"


###################################################################
#
# MODELS
#
###################################################################
# MODEL = DecisionTreeClassifier()
# MODEL = RandomForestClassifier()
# MODEL = HSTreeClassifier() # Note: Does not work well with Pandas dataframes. Use Numpy arrays


ENSEMBLE = DecisionTreeClassifier()
MODEL = HSTreeClassifier(estimator_=ENSEMBLE)


METHOD = 'PCA-HS-DT (CREDIT CARD)'


###################################################################
#
# FUNCTIONS
#
###################################################################
# Loading tabular datasets
def load_data_tabular(url):
    if url.endswith('.csv'):
        df = pd.read_csv(url)
    else:
        df = pd.read_excel(url, header=1)

    return df

# Loading image datasets
def load_data_image(dataset):
    (x_train_images, y_train_images), (x_test_images, y_test_images) = dataset.load_data()

    return x_train_images, y_train_images, x_test_images, y_test_images

# Splitting datasets
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

###################################################################
#
# CIFAR-10
#
###################################################################
# with open('Temp.csv', 'a', newline='') as csvfile:
#     #writer = csv.writer(csvfile)
#     #writer.writerow([METHOD])
#     #writer.writerow(['Average Accuracy', 'Time (min)'])
#     runs = pd.DataFrame(columns=['Accuracy', 'Time (min)'])
#     for run in range(0, 100):
#         # Loading dataset
#         x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_data_image(cifar10)
#         print("Length of testing: ", len(y_test_cifar))
#
#         # Flatten images
#         x_train_cifar_flat = x_train_cifar.reshape(x_train_cifar.shape[0], -1)
#         x_test_cifar_flat = x_test_cifar.reshape(x_test_cifar.shape[0], -1)
#
#         # Normalize images
#         x_train_cifar_flat = x_train_cifar_flat / 255.0
#         x_test_cifar_flat = x_test_cifar_flat / 255.0
#
#         # PCA
#         # pca = PCA(0.99)
#         # pca.fit(x_train_cifar_flat)
#         # x_train_cifar_flat = pca.transform(x_train_cifar_flat)
#         # x_test_cifar_flat = pca.transform(x_test_cifar_flat)
#
#         # Training model
#         start_time = time.time()
#         MODEL.fit(x_train_cifar_flat, y_train_cifar.ravel())
#         end_time = time.time()
#
#         predictions = MODEL.predict(x_test_cifar_flat)
#         accuracy = (accuracy_score(y_test_cifar, predictions)) * 100
#         runs.loc[run] = accuracy, ((end_time - start_time) / 60)
#
#     writer.writerow([runs['Accuracy'].mean(), runs['Time (min)'].mean()])
#     writer.writerow([runs['Accuracy'].std(), runs['Time (min)'].std()])



###################################################################
#
# FASHION-MINST
#
###################################################################
with open('Temp.csv', 'a', newline='') as csvfile:
    #writer = csv.writer(csvfile)
    #writer.writerow([])
    #writer.writerow([METHOD])
    #writer.writerow(['Run', 'Accuracy', 'Time (s)'])
    for run in range(1, 6):
        # Loading dataset
        x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion = load_data_image(fashion_mnist)
        print("Length of testing: ", len(y_test_fashion))

        # Flatten images
        x_train_fashion_flat = x_train_fashion.reshape(x_train_fashion.shape[0], -1)
        x_test_fashion_flat = x_test_fashion.reshape(x_test_fashion.shape[0], -1)

        # Normalize images
        x_train_fashion_flat = x_train_fashion_flat / 255.0
        x_test_fashion_flat = x_test_fashion_flat / 255.0

        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_fashion_flat)
        x_train_fashion_flat = pca.transform(x_train_fashion_flat)
        x_test_fashion_flat = pca.transform(x_test_fashion_flat)

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_fashion_flat, y_train_fashion.ravel())
        end_time = time.time()

        predictions = MODEL.predict(x_test_fashion_flat)
        accuracy = accuracy_score(y_test_fashion, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])



###################################################################
#
# ADULT INCOME
#
###################################################################
with open('Temp.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow([METHOD])
    writer.writerow(['Average Accuracy', 'Time (min)'])
    runs = pd.DataFrame(columns=['Accuracy', 'Time (min)'])
    for run in range(0, 100):
        # Loading dataset
        data_adult = pd.read_csv(ADULT_URL, names=ADULT_COLS, sep=',\s', engine='python')

        # Remove rows with missing values
        data_adult.replace('?', np.nan, inplace=True)
        data_adult.dropna(inplace=True)

        # Separate features and target variable
        x_adult = data_adult.drop('income', axis=1)
        y_adult = data_adult['income']

        # Convert categorical variables to numerical using Label Encoding
        categorical_columns = x_adult.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            x_adult[col] = le.fit_transform(x_adult[col])

        # Splitting dataset
        x_train_adult, x_test_adult, y_train_adult, y_test_adult = split_data(x_adult, y_adult)

        x_train_adult = x_train_adult.reset_index(drop=True).to_numpy()
        y_train_adult = y_train_adult.reset_index(drop=True).to_numpy()
        x_test_adult = x_test_adult.reset_index(drop=True).to_numpy()
        y_test_adult = y_test_adult.reset_index(drop=True).to_numpy()

        # Standard Scaler
        scaler = StandardScaler()
        x_train_adult = scaler.fit_transform(x_train_adult)
        x_test_adult = scaler.transform(x_test_adult)

        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_adult)
        x_train_adult = pca.transform(x_train_adult)
        x_test_adult = pca.transform(x_test_adult)

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_adult, y_train_adult)
        end_time = time.time()

        predictions = MODEL.predict(x_test_adult)
        accuracy = (accuracy_score(y_test_adult, predictions)) * 100
        runs.loc[run] = accuracy, ((end_time - start_time) / 60)

    writer.writerow([runs['Accuracy'].mean(), runs['Time (min)'].mean()])
    writer.writerow([runs['Accuracy'].std(), runs['Time (min)'].std()])


###################################################################
#
# TITANIC
#
###################################################################
with open('Temp.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow([METHOD])
    writer.writerow(['Average Accuracy', 'Time (min)'])
    runs = pd.DataFrame(columns=['Accuracy', 'Time (min)'])
    for run in range(0, 100):
        # Loading dataset
        data_titanic = load_data_tabular(TITANIC_URL)
        data_titanic.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace=True)
        data_titanic['Age'] = data_titanic['Age'].fillna(data_titanic['Age'].median())
        categorical_columns = data_titanic.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            data_titanic[col] = le.fit_transform(data_titanic[col])

        x_titanic = data_titanic.drop('Survived', axis=1)
        y_titanic = data_titanic['Survived']

        # Splitting dataset
        x_train_titanic, x_test_titanic, y_train_titanic, y_test_titanic = split_data(x_titanic, y_titanic)

        # Convert to numpy array if needed
        x_train_titanic = x_train_titanic.reset_index(drop=True).to_numpy()
        x_test_titanic = x_test_titanic.reset_index(drop=True).to_numpy()
        y_train_titanic = y_train_titanic.reset_index(drop=True).to_numpy()
        y_test_titanic = y_test_titanic.reset_index(drop=True).to_numpy()

        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_titanic)
        x_train_titanic = pca.transform(x_train_titanic)
        x_test_titanic = pca.transform(x_test_titanic)

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_titanic, y_train_titanic)
        end_time = time.time()

        predictions = MODEL.predict(x_test_titanic)
        accuracy = (accuracy_score(y_test_titanic, predictions)) * 100
        runs.loc[run] = accuracy, ((end_time - start_time) / 60)


    writer.writerow([runs['Accuracy'].mean(), runs['Time (min)'].mean()])
    writer.writerow([runs['Accuracy'].std(), runs['Time (min)'].std()])



###################################################################
#
# CREDIT CARD
#
###################################################################
with open('Temp.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow([METHOD])
    writer.writerow(['Average Accuracy', 'Time (min)'])
    runs = pd.DataFrame(columns=['Accuracy', 'Time (min)'])
    for run in range(0, 100):
        # Loading dataset
        data_credit = load_data_tabular(CREDIT_URL)
        data_credit.drop(columns=['ID', 'SEX'], inplace=True)
        x_credit = data_credit.drop('default payment next month', axis=1)
        y_credit = data_credit['default payment next month']

        # Splitting dataset
        x_train_credit, x_test_credit, y_train_credit, y_test_credit = split_data(x_credit, y_credit)


        # Converting pandas to numpy array
        x_train_credit = x_train_credit.reset_index(drop=True).to_numpy()
        x_test_credit = x_test_credit.reset_index(drop=True).to_numpy()
        y_train_credit = y_train_credit.reset_index(drop=True).to_numpy()
        y_test_credit = y_test_credit.reset_index(drop=True).to_numpy()


        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_credit)
        x_train_credit = pca.transform(x_train_credit)
        x_test_credit = pca.transform(x_test_credit)


        # Training model
        start_time = time.time()
        MODEL.fit(x_train_credit, y_train_credit)
        end_time = time.time()

        predictions = MODEL.predict(x_test_credit)
        accuracy = (accuracy_score(y_test_credit, predictions)) * 100
        runs.loc[run] = accuracy, ((end_time - start_time) / 60)


    writer.writerow([runs['Accuracy'].mean(), runs['Time (min)'].mean()])
    writer.writerow([runs['Accuracy'].std(), runs['Time (min)'].std()])



###################################################################
#
# OXFORD PETS
#
###################################################################
# Loading dataset
# batch_size = 32
# img_height = 128
# img_width = 128
# dataset = image_dataset_from_directory(
#     OXFORD_DIR,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     label_mode='int'  # Labels are categorical integers (0-36 for the breeds)
# )
#
# # List all image files
# image_files = [f for f in os.listdir(OXFORD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
#
# # Load images
# images = []
# for file in image_files:
#     img_path = os.path.join(OXFORD_DIR, file)
#     img = load_img(img_path, target_size=(128, 128))  # Resize images to 128x128 pixels
#     img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
#     images.append(img_array)
#
# # Convert to a numpy array
# images = np.array(images)
#
# print(f"Loaded {len(images)} images.")
# print(images.shape)
#
# PCA
# pca = PCA(0.99)
# pca.fit(x_train_credit)
# x_train_credit = pca.transform(x_train_credit)
# x_test_credit = pca.transform(x_test_credit)




###################################################################
#
# NEW DATASETS
#
###################################################################


# fetch dataset
multivariate_gait_data = fetch_ucirepo(id=760)

# data (as pandas dataframes)
X = multivariate_gait_data.data.features
y = multivariate_gait_data.data.targets

features = X.drop(columns=['condition'])
features = features.columns

x = X.drop(columns=['condition'])
y = X['condition']

x = x.to_numpy()
y = y.to_numpy()

x_train_gait, x_test_gait, y_train_gait, y_test_gait = split_data(x, y)



##########################################################################################
# # Download dataset
dataset_path = kagglehub.dataset_download("rabieelkharoua/students-performance-dataset")

# List all files in the dataset directory
dataset_files = os.listdir(dataset_path)

# Find the CSV file(s)
csv_files = [f for f in dataset_files if f.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError("No CSV file found in dataset folder.")

# Load the first CSV file
csv_path = os.path.join(dataset_path, csv_files[0])
df = pd.read_csv(csv_path)

# Display the first few rows of the DataFrame
print(df.head())
X = df.drop(columns=['GradeClass'])  # Drop the target column to get features
y = df['GradeClass']  # Target is the GradeClass column



pca = PCA(0.99)
pca.fit(X)
X = pca.transform(X)



features = df.drop(columns=['GradeClass'])
features = features.columns



##########################################################################################


# # Fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# Data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

# Convert categorical variables to numerical using Label Encoding
categorical_columns = y.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    y.loc[:, col] = le.fit_transform(y[col])  # Fix SettingWithCopyWarning




y = y.squeeze()  # Converts DataFrame with one column to Series
y = y.astype(int)
y = y.dropna()  # Remove NaNs

X = X.to_numpy()
y = y.to_numpy()

# # Split data
x_train_student, x_test_student, y_train_student, y_test_student = split_data(X, y)





















###################################################################
#
# TRAINING MODELS
#
###################################################################

# Initialize a list of classifiers that consists only of CART and HSCART classifiers
cart_hscart_estimators = [
    model for model_group in ESTIMATORS_CLASSIFICATION
    for model in model_group
    if model.name in ['CART', 'HSCART']
]

results = []

for model_config in cart_hscart_estimators:  # Loop through CART and HSCART models
    model_name = model_config.name
    model_class = model_config.cls
    model_kwargs = model_config.kwargs.copy()  # Copy to safely modify

    if model_name == 'CART':
        # Train CART
        cart_model = model_class(**model_kwargs)
        cart_model.fit(x_train_gait, y_train_gait)
        y_pred_proba = cart_model.predict_proba(x_test_gait)[:, 1]
        predictions = cart_model.predict(x_test_gait)
        accuracy = accuracy_score(y_test_gait, predictions)
        #print(f"Shape of y_pred_proba: {y_pred_proba.shape}")

        # Calculate AUC for CART
        #auc_cart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')

        # Get tree size (number of nodes) and depth
        tree_size = cart_model.tree_.node_count
        tree_depth = cart_model.tree_.max_depth
        if model_kwargs['max_leaf_nodes'] == 20:
            plt.figure(figsize=(20, 10))
            plot_tree(cart_model)
            plt.title('CART: Gait', fontsize=35)
            plt.savefig('cart_gait')
            plt.show()


        # Append CART results
        results.append({
            'Dataset': "Adult",
            'Model': 'CART',
            'Max Leaves': model_kwargs['max_leaf_nodes'],
            'Lambda': None,  # CART does not use lambda
            #'AUC': auc_cart,
            'Tree Size': tree_size,
            'Tree Depth': tree_depth,
            'Split Seed': "na",
            'Accuracy': accuracy
        })

    elif model_name == 'HSCART':
        # Train HSCART
        hscart_model = model_class(**model_kwargs)
        start_time = time.time()
        hscart_model.fit(x_train_gait, y_train_gait)
        end_time = time.time()
        #print(f"Total time {model_kwargs['max_leaf_nodes']}: {(end_time - start_time) / 60}")

        # Predict and calculate AUC for HSCART
        y_pred_proba = hscart_model.predict_proba(x_test_gait)
        predictions = hscart_model.predict(x_test_gait)
        accuracy = accuracy_score(y_test_gait, predictions)
        #auc_hscart = roc_auc_score(y_test_student, y_pred_proba, multi_class='ovo')

        # Get tree size (number of nodes) and depth
        if hasattr(hscart_model, 'estimator_'):
            decision_tree = hscart_model.estimator_
            tree_size = decision_tree.tree_.node_count
            tree_depth = decision_tree.tree_.max_depth
        else:
            raise AttributeError("HSCART model does not contain an attribute 'estimator_' for tree visualization.")

        # Append HSCART results
        results.append({
            'Dataset': "Adult",
            'Model': 'HSCART',
            'Max Leaves': model_kwargs['max_leaf_nodes'],
            'Lambda': hscart_model.reg_param,  # Save the selected lambda
            #'AUC': auc_hscart,
            'Tree Size': tree_size,
            'Tree Depth': tree_depth,
            'Split Seed': "na",
            'Accuracy': accuracy
        })


# Convert results to DataFrame and save
results_df = pd.DataFrame(results)


###################################################################
#
# PLOTS
#
###################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Manually input the times for each dataset and model (in seconds)
time_data = {
    'CIFAR-10': [[242.1991422, 241.5421028, 242.4317679, 240.107276, 239.3795531],
                 [219.521008, 219.398421, 217.6687739, 223.4067731, 237.2773688],
                 [547.0455508, 544.337544, 542.9848099, 545.2337132, 545.7758899],
                 [301.5395989, 305.0823662, 304.612591, 305.078649, 303.4675562]],

    'FASHION-MINST': [[85.07053995, 82.27442884, 86.89681196, 82.37714314, 81.27386975],
                      [212.085438, 211.337795, 211.327845, 211.390029, 209.868541],
                      [103.372396, 103.1430721, 102.9896231, 103.0515089, 103.5272992],
                      [263.8968761, 264.0881269, 264.7886379, 264.333806, 264.1657717]],

    'ADULT INCOME': [[2.232364893, 2.258303165, 2.254941225, 2.237604141, 2.246879339],
                     [9.25217485, 9.00513578, 8.85312295, 8.83711314, 8.85314393],
                     [0.715862036, 0.708870888, 0.707540989, 0.71313262, 0.715147972],
                     [2.79746294, 2.772037983, 2.781803131, 2.767557144, 2.761221886]],

    'TITANIC': [[0.158365965, 0.153714895, 0.167060137, 0.170248032, 0.146846771],
                [0.174762964, 0.174715996, 0.178793907, 0.177582026, 0.169512987],
                [0.031949043, 0.026435852, 0.024561167, 0.02679491, 0.025578976],
                [0.053377151, 0.026199102, 0.028033018, 0.025827885, 0.027102947]],

    'CREDIT CARD': [[7.067342043, 6.986837149, 7.351256847, 7.211401939, 6.985906363],
                    [8.923218966, 8.954082966, 8.976350307, 9.06367588, 8.916848898],
                    [2.170444965, 2.165299892, 2.173708916, 2.174731016, 2.199632883],
                    [1.542072773, 1.548459053, 1.551234007, 1.541230202, 1.541714907]]
}

# Convert seconds to minutes by dividing by 60
#time_data_minutes = {dataset: [[t / 60 for t in times] for times in models] for dataset, models in time_data.items()}

# Prepare the data for seaborn
datasets = []
models = []
times = []

for dataset, data in time_data_minutes.items():
    for i, model_times in enumerate(data):
        model_name = ['RF', 'PCA-RF', 'HS-RF', 'PCA-HS-RF'][i]
        datasets.extend([dataset] * len(model_times))
        models.extend([model_name] * len(model_times))
        times.extend(model_times)

# Create a DataFrame
df = pd.DataFrame({
    'Dataset': datasets,
    'Model': models,
    'Time (min)': times
})

# Plot individual boxplots for each dataset
sns.set(style="whitegrid")

# Unique datasets
unique_datasets = df['Dataset'].unique()

# Loop through each dataset and create individual plots
for dataset in unique_datasets:
    plt.figure(figsize=(5, 7))  # Adjust figure size to make plots visually appealing

    # Filter data for the current dataset
    data_filtered = df[df['Dataset'] == dataset]

    # Create the boxplot
    sns.boxplot(x='Model', y='Time (min)', data=data_filtered, width=0.6)  # Wider boxes

    # Set the title
    plt.title(f'Boxplot of Time (min) for {dataset}', fontsize=16)

    # Adjust y-axis limits (optional, remove or modify as necessary)
    plt.ylim(0, None)

    # Improve spacing
    plt.tight_layout()

    # Show each plot separately
    plt.show()



###################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Manually input the times for each dataset and model (in seconds)
time_data = {
    'CIFAR-10': [[242.1991422, 241.5421028, 242.4317679, 240.107276, 239.3795531],
                 [219.521008, 219.398421, 217.6687739, 223.4067731, 237.2773688],
                 [547.0455508, 544.337544, 542.9848099, 545.2337132, 545.7758899],
                 [301.5395989, 305.0823662, 304.612591, 305.078649, 303.4675562]],

    'FASHION-MINST': [[85.07053995, 82.27442884, 86.89681196, 82.37714314, 81.27386975],
                      [212.085438, 211.337795, 211.327845, 211.390029, 209.868541],
                      [103.372396, 103.1430721, 102.9896231, 103.0515089, 103.5272992],
                      [263.8968761, 264.0881269, 264.7886379, 264.333806, 264.1657717]],

    'ADULT INCOME': [[2.232364893, 2.258303165, 2.254941225, 2.237604141, 2.246879339],
                     [9.25217485, 9.00513578, 8.85312295, 8.83711314, 8.85314393],
                     [0.715862036, 0.708870888, 0.707540989, 0.71313262, 0.715147972],
                     [2.79746294, 2.772037983, 2.781803131, 2.767557144, 2.761221886]],

    'TITANIC': [[0.158365965, 0.153714895, 0.167060137, 0.170248032, 0.146846771],
                [0.174762964, 0.174715996, 0.178793907, 0.177582026, 0.169512987],
                [0.031949043, 0.026435852, 0.024561167, 0.02679491, 0.025578976],
                [0.053377151, 0.026199102, 0.028033018, 0.025827885, 0.027102947]],

    'CREDIT CARD': [[7.067342043, 6.986837149, 7.351256847, 7.211401939, 6.985906363],
                    [8.923218966, 8.954082966, 8.976350307, 9.06367588, 8.916848898],
                    [2.170444965, 2.165299892, 2.173708916, 2.174731016, 2.199632883],
                    [1.542072773, 1.548459053, 1.551234007, 1.541230202, 1.541714907]]
}

# Convert seconds to minutes by dividing by 60
time_data_minutes = {dataset: [[t / 60 for t in times] for times in models] for dataset, models in time_data.items()}

# Prepare the data for seaborn
datasets = []
models = []
times = []

for dataset, data in time_data_minutes.items():
    for i, model_times in enumerate(data):
        model_name = ['RF', 'PCA-RF', 'HS-RF', 'PCA-HS-RF'][i]
        datasets.extend([dataset] * len(model_times))
        models.extend([model_name] * len(model_times))
        times.extend(model_times)

# Create a DataFrame
df = pd.DataFrame({
    'Dataset': datasets,
    'Model': models,
    'Time (min)': times
})

# Plot boxplots for each dataset
sns.set(style="whitegrid")
g = sns.catplot(x="Model", y="Time (min)", col="Dataset", data=df, kind="box", height=4, aspect=0.7)
g.set_titles("{col_name}")

# Display the plot
plt.tight_layout()
plt.savefig("One_Fig")
plt.show()



###################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Manually input the times for each dataset and model (in seconds)
time_data = {
    'CIFAR-10': [[242.1991422, 241.5421028, 242.4317679, 240.107276, 239.3795531],
                 [219.521008, 219.398421, 217.6687739, 223.4067731, 237.2773688],
                 [547.0455508, 544.337544, 542.9848099, 545.2337132, 545.7758899],
                 [301.5395989, 305.0823662, 304.612591, 305.078649, 303.4675562]],

    'FASHION-MINST': [[85.07053995, 82.27442884, 86.89681196, 82.37714314, 81.27386975],
                      [212.085438, 211.337795, 211.327845, 211.390029, 209.868541],
                      [103.372396, 103.1430721, 102.9896231, 103.0515089, 103.5272992],
                      [263.8968761, 264.0881269, 264.7886379, 264.333806, 264.1657717]],

    'ADULT INCOME': [[2.232364893, 2.258303165, 2.254941225, 2.237604141, 2.246879339],
                     [9.25217485, 9.00513578, 8.85312295, 8.83711314, 8.85314393],
                     [0.715862036, 0.708870888, 0.707540989, 0.71313262, 0.715147972],
                     [2.79746294, 2.772037983, 2.781803131, 2.767557144, 2.761221886]],

    'TITANIC': [[0.158365965, 0.153714895, 0.167060137, 0.170248032, 0.146846771],
                [0.174762964, 0.174715996, 0.178793907, 0.177582026, 0.169512987],
                [0.031949043, 0.026435852, 0.024561167, 0.02679491, 0.025578976],
                [0.053377151, 0.026199102, 0.028033018, 0.025827885, 0.027102947]],

    'CREDIT CARD': [[7.067342043, 6.986837149, 7.351256847, 7.211401939, 6.985906363],
                    [8.923218966, 8.954082966, 8.976350307, 9.06367588, 8.916848898],
                    [2.170444965, 2.165299892, 2.173708916, 2.174731016, 2.199632883],
                    [1.542072773, 1.548459053, 1.551234007, 1.541230202, 1.541714907]]
}

# Convert seconds to minutes by dividing by 60
time_data_minutes = {dataset: [[t / 60 for t in times] for times in models] for dataset, models in time_data.items()}

# Prepare the data for seaborn
datasets = []
models = []
times = []

for dataset, data in time_data_minutes.items():
    for i, model_times in enumerate(data):
        model_name = ['RF', 'PCA-RF', 'HS-RF', 'PCA-HS-RF'][i]
        datasets.extend([dataset] * len(model_times))
        models.extend([model_name] * len(model_times))
        times.extend(model_times)

# Create a DataFrame
df = pd.DataFrame({
    'Dataset': datasets,
    'Model': models,
    'Time (min)': times
})

# Plot individual boxplots for each dataset
sns.set(style="whitegrid")

# Unique datasets
unique_datasets = df['Dataset'].unique()

# Loop through each dataset and create individual plots
for dataset in unique_datasets:
    plt.figure(figsize=(8, 6))  # Adjust figure size to make plots visually appealing

    # Filter data for the current dataset
    data_filtered = df[df['Dataset'] == dataset]

    # Create the boxplot
    sns.boxplot(x='Model', y='Time (min)', data=data_filtered, width=0.6)  # Wider boxes

    # Set the title
    plt.title(f'Boxplot of Time (min) for {dataset}', fontsize=16)

    # Adjust y-axis limits (optional, remove or modify as necessary)
    plt.ylim(0, None)

    # Improve spacing
    plt.tight_layout()

    # Show each plot separately
    plt.savefig(f"Figure_{dataset}")
    plt.show()

###################################################################
#
# NEW PLOTS
#
###################################################################



# Extract the underlying decision tree
if hasattr(hscart_model, 'estimator_'):
    decision_tree = hscart_model.estimator_
else:
    raise AttributeError("HSCART model does not contain an attribute 'estimator_' for tree visualization.")

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=features)
plt.title("HSCART: Gait", fontsize=35)
plt.savefig('HSCART_gait')
plt.show()



# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Filter results for CART and HSCART
cart_results = results_df[results_df['Model'] == 'CART']
hscart_results = results_df[results_df['Model'] == 'HSCART']

# Extract tree sizes and depths
cart_tree_sizes = cart_results['Tree Size']
cart_tree_depths = cart_results['Tree Depth']
hscart_tree_sizes = hscart_results['Tree Size']
hscart_tree_depths = hscart_results['Tree Depth']

# Combine data for box plots
tree_sizes = pd.DataFrame({
    'Model': ['CART'] * len(cart_tree_sizes) + ['HSCART'] * len(hscart_tree_sizes),
    'Tree Size': list(cart_tree_sizes) + list(hscart_tree_sizes)
})

tree_depths = pd.DataFrame({
    'Model': ['CART'] * len(cart_tree_depths) + ['HSCART'] * len(hscart_tree_depths),
    'Tree Depth': list(cart_tree_depths) + list(hscart_tree_depths)
})

# Create box plots
plt.figure(figsize=(12, 6))

# Box plot for Tree Size
plt.subplot(1, 2, 1)
sns.boxplot(x='Model', y='Tree Size', data=tree_sizes, palette='Set2')
plt.title('Tree Size Comparison')
plt.xlabel('Model')
plt.ylabel('Tree Size (Number of Nodes)')

# Box plot for Tree Depth
plt.subplot(1, 2, 2)
sns.boxplot(x='Model', y='Tree Depth', data=tree_depths, palette='Set2')
plt.title('Tree Depth Comparison')
plt.xlabel('Model')
plt.ylabel('Tree Depth')

plt.tight_layout()
plt.show()


###################################################################
#
# OXFORD PETS CODE
#
###################################################################


# Function to convert tf.data.Dataset to a pandas DataFrame
def dataset_to_dataframe(tf_dataset):
    images = []
    labels = []

    # Iterate through the dataset
    for image, label in tf_dataset:
        # Convert image to numpy array and append to list
        images.append(image.numpy())
        labels.append(label.numpy())

    # Create a DataFrame
    df = pd.DataFrame({
        'image': images,
        'label': labels
    })

    return df


# Function to resize and convert images to uint8
def resize_images_to_uint8(image_list, target_size):
    resized_images = []

    for image in image_list:
        # Resize the image to the target size
        resized_image = tf.image.resize(image, target_size).numpy().astype(np.uint8)
        resized_images.append(resized_image)

    return np.array(resized_images)


# Desired image size (500x500 for this example)
TARGET_SIZE = (500, 500)
METHOD = 'PCA-DT (OXFORD PETS)'
MODEL = DecisionTreeClassifier()
PATH = '/Users/luisvazquez/Master\'s Thesis/HierarchicalShrinkage/Results.csv'


with open(PATH, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([METHOD])
    writer.writerow([])
    writer.writerow(['Run',  'Accuracy', 'Time (s)'])
    for run in range(1, 6):
        # Load the Oxford Pets dataset
        dataset, info = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
        print("Loaded dataset")

        # Access the train and test sets
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        # Convert train and test datasets to DataFrames
        train_df = dataset_to_dataframe(train_dataset)
        test_df = dataset_to_dataframe(test_dataset)
        print("Converted to dataframes")

        # # Resize all images and convert them to uint8
        x_train_pets = resize_images_to_uint8(train_df['image'], TARGET_SIZE)
        x_test_pets = resize_images_to_uint8(test_df['image'], TARGET_SIZE)
        y_train_pets = train_df['label']
        y_test_pets = test_df['label']
        print("Resized and converted")

        # Flattening images
        x_train_pets_flat = x_train_pets.reshape(x_train_pets.shape[0], -1)
        x_test_pets_flat = x_test_pets.reshape(x_test_pets.shape[0], -1)
        print("Flatten images")

        # Normalize images
        x_train_pets_flat = x_train_pets_flat / 255.0
        x_test_pets_flat = x_test_pets_flat / 255.0
        print("Normalized images")

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_pets_flat, y_train_pets.to_numpy())
        end_time = time.time()

        # Getting model accuracy
        predictions = MODEL.predict(x_test_pets_flat)
        accuracy = accuracy_score(y_pets_cifar, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])
        print("Finished run: ", run)





