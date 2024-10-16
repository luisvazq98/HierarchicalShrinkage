###############################################
#
# LIBRARIES
#
###############################################
import csv
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, fashion_mnist
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from imodels import HSTreeClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time
import numpy as np


###############################################
#
# URL'S FOR DATASETS
#
###############################################
TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
CREDIT_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

OXFORD_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OXFORD_DIR = tf.keras.utils.get_file('oxford_pets', origin=OXFORD_URL, extract=True)
OXFORD_DIR = pathlib.Path(OXFORD_DIR).parent / "images"

MODEL = DecisionTreeClassifier()
#MODEL = HSTreeClassifierCV()
METHOD = 'DT (OXFORD PETS)'


###############################################
# 
# FUNCTIONS
#
###############################################
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


###############################################
#
# CIFAR-10
#
###############################################
with open('Results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([METHOD])
    writer.writerow(['Run',  'Accuracy', 'Time (s)'])
    for run in range(1, 6):
        # Loading dataset
        x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_data_image(cifar10)

        # Flatten images
        x_train_cifar_flat = x_train_cifar.reshape(x_train_cifar.shape[0], -1)
        x_test_cifar_flat = x_test_cifar.reshape(x_test_cifar.shape[0], -1)

        # Normalize images
        x_train_cifar_flat = x_train_cifar_flat / 255.0
        x_test_cifar_flat = x_test_cifar_flat / 255.0

        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_cifar_flat)
        x_train_cifar_flat = pca.transform(x_train_cifar_flat)
        x_test_cifar_flat = pca.transform(x_test_cifar_flat)

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_cifar_flat, y_train_cifar.ravel())
        end_time = time.time()

        predictions = MODEL.predict(x_test_cifar_flat)
        accuracy = accuracy_score(y_test_cifar, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])




###############################################
#
# FASHION-MINST
#
###############################################
with open('Results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([METHOD])
    writer.writerow(['Run', 'Accuracy', 'Time (s)'])
    for run in range(1, 6):
        # Loading dataset
        x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion = load_data_image(fashion_mnist)

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


###############################################
#
# ADULT INCOME
#
###############################################
with open('Results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([''])
    writer.writerow([METHOD])
    writer.writerow(['Run', 'Accuracy', 'Time (s)'])
    for run in range(1, 6):
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
        accuracy = accuracy_score(y_test_adult, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])

###############################################
#
# TITANIC
#
###############################################
with open('Results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow([METHOD])
    writer.writerow(['Run', 'Accuracy', 'Time (s)'])
    for run in range(1, 6):
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
        accuracy = accuracy_score(y_test_titanic, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])


###############################################
#
# CREDIT CARD
#
###############################################
with open('Results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([])
    writer.writerow([METHOD])
    writer.writerow(['Run', 'Accuracy', 'Time (s)'])
    for run in range(1, 6):
        # Loading dataset
        data_credit = load_data_tabular(CREDIT_URL)
        data_credit.drop(columns=['ID', 'SEX'], inplace=True)
        x_credit = data_credit.drop('default payment next month', axis=1)
        y_credit = data_credit['default payment next month']

        # Splitting dataset
        x_train_credit, x_test_credit, y_train_credit, y_test_credit = split_data(x_credit, y_credit)

        # PCA
        pca = PCA(0.99)
        pca.fit(x_train_credit)
        x_train_credit = pca.transform(x_train_credit)
        x_test_credit = pca.transform(x_test_credit)

        # Converting pandas to numpy array
        y_train_credit = np.array(y_train_credit)
        y_test_credit = np.array(y_test_credit)

        # Training model
        start_time = time.time()
        MODEL.fit(x_train_credit, y_train_credit)
        end_time = time.time()

        predictions = MODEL.predict(x_test_credit)
        accuracy = accuracy_score(y_test_credit, predictions)
        writer.writerow([run, accuracy, (end_time - start_time)])


###############################################
#
# OXFORD PETS
#
###############################################
# Loading dataset
batch_size = 32
img_height = 128
img_width = 128
dataset = image_dataset_from_directory(
    OXFORD_DIR,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'  # Labels are categorical integers (0-36 for the breeds)
)

# List all image files
image_files = [f for f in os.listdir(OXFORD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load images
images = []
for file in image_files:
    img_path = os.path.join(OXFORD_DIR, file)
    img = load_img(img_path, target_size=(128, 128))  # Resize images to 128x128 pixels
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    images.append(img_array)

# Convert to a numpy array
images = np.array(images)

# print(f"Loaded {len(images)} images.")
# print(images.shape)

# PCA
# pca = PCA(0.99)
# pca.fit(x_train_credit)
# x_train_credit = pca.transform(x_train_credit)
# x_test_credit = pca.transform(x_test_credit)


###############################################
#
# PLOTS
#
###############################################
# Simple boxplot
data_adult = {
    'Dataset': ['CIFAR-10', 'Fashion-MNIST', 'Oxford Pets', 'Adult Income', 'Titanic', 'Credit Card'],
    'DT': [3.23, 0.659, 4.56, 0.011, 0.0001, 0.0071],
    'PCA-DT': [1.27, 1.57, 3.3, 0.012, 0.0001, 0.0071],
    'HS-DT': [13.05, 2.65, 24.09, 0.023, 0.0016, 0.06],
    'PCA-HS-DT': [8.37, 7.50, 22.13, 0.0805, 0.0018, 0.043]
}

# Create DataFrame
df = pd.DataFrame(data_adult)

# Creating figure
plt.figure(figsize=(6,6))
df.boxplot()

# Labeling x and y axis
plt.ylabel("Time (minutes)")
plt.xlabel("Method")

# Saving and showing boxplot
#plt.savefig("boxplot")
plt.show()

# Fancy boxplot
data_adult = {
    'Dataset': ['CIFAR-10', 'Fashion-MNIST', 'Oxford Pets', 'Adult Income', 'Titanic', 'Credit Card'],
    'DT': [3.23, 0.659, 4.56, 0.011, 0.0001, 0.0071],
    'PCA-DT': [1.27, 1.57, 3.3, 0.012, 0.0001, 0.0071],
    'HS-DT': [13.05, 2.65, 24.09, 0.023, 0.0016, 0.06],
    'PCA-HS-DT': [8.37, 7.50, 22.13, 0.0805, 0.0018, 0.043]
}

# Create DataFrame
df = pd.DataFrame(data_adult)

# Set the style of seaborn
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(10, 6))
boxplot = sns.boxplot(data=df.iloc[:, 1:], palette="Set2", width=0.5)

# Labeling x and y axis
plt.ylabel("Time (minutes)", fontsize=20)
plt.xlabel("Method", fontsize=20)

# Setting size of x and y ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adding gridlines
plt.grid(axis='y', linestyle='--', alpha=0.9)

# Saving and showing boxplot
#plt.savefig("boxplot")
plt.show()
