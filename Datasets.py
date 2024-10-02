###############################################
#
# LIBRARIES AND GLOBAL VARIABLES
#
###############################################

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, fashion_mnist
from sklearn.datasets import load_iris
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
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

TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
CREDIT_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']



# Download the dataset
dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
data_dir = tf.keras.utils.get_file('oxford_pets', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).parent / "images"

# Load dataset from the directory
batch_size = 32
img_height = 128
img_width = 128

dataset = image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'  # Labels are categorical integers (0-36 for the breeds)
)




# List all image files
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load images
images = []
for file in image_files:
    img_path = os.path.join(data_dir, file)
    img = load_img(img_path, target_size=(128, 128))  # Resize images to 128x128 pixels
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    images.append(img_array)

# Convert to a numpy array
images = np.array(images)

print(f"Loaded {len(images)} images.")
print(images.shape)




###############################################
# 
# FUNCTIONS FOR LOADING DATA
#
###############################################

###### TABULAR DATASETS ######
def load_data_tabular(url):
    if url.endswith('.csv'):
        df = pd.read_csv(url)
    else:
        df = pd.read_excel(url, header=1)

    return df


###### IMAGE DATASETS ######
def load_data_image(dataset):
    (x_train_images, y_train_images), (x_test_images, y_test_images) = dataset.load_data()

    return x_train_images, y_train_images, x_test_images, y_test_images

#dataset, info = tfds.load('oxford_iiit_pet', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)

###############################################
#
# FUNCTIONS FOR PREPROCESSING DATA
#
###############################################

###### SPLITTING DATASETS (80 TRAINING, 20 TESTING) ######
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test



###############################################
#
# DEVELOPMENT AND TRAINING OF MODELS
#
###############################################

###### LOADING DATA AND PROCESSING DATA ######
# CIFAR-10
x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_data_image(cifar10)

# Flatten images
x_train_cifar_flat = x_train_cifar.reshape(x_train_cifar.shape[0], -1)
x_test_cifar_flat = x_test_cifar.reshape(x_test_cifar.shape[0], -1)

# Normalize images
x_train_cifar_flat = x_train_cifar_flat / 255.0
x_test_cifar_flat = x_test_cifar_flat / 255.0



# FASHION-MINST
x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion = load_data_image(fashion_mnist)

# Flatten images
x_train_fashion_flat = x_train_fashion.reshape(x_train_fashion.shape[0], -1)
x_test_fashion_flat = x_test_fashion.reshape(x_test_fashion.shape[0], -1)

# Normalize images
x_train_fashion_flat = x_train_fashion_flat / 255.0
x_test_fashion_flat = x_test_fashion_flat / 255.0



# ADULT INCOME
data = pd.read_csv(url, names=columns, sep=',\s', engine='python')

# Remove rows with missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and target variable
x = data.drop('income', axis=1)
y = data['income']

# Convert categorical variables to numerical using Label Encoding
categorical_columns = x.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])


# TITANIC
titanic = load_data_tabular(TITANIC_URL)
titanic.drop(columns=['PassengerId', 'Cabin', 'Ticket', 'Name'], inplace=True)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
categorical_columns = titanic.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    titanic[col] = le.fit_transform(titanic[col])

x_titanic = titanic.drop('Survived', axis=1)
y_titanic = titanic['Survived']



# CREDIT CARD
credit_card = load_data_tabular(CREDIT_URL)
credit_card.drop(columns=['ID', 'SEX'], inplace=True)
x_credit = credit_card.drop('default payment next month', axis=1)
y_credit = credit_card['default payment next month']


###### SPLITTING DATA ######
# ADULT INCOME
x_train, x_test, y_train, y_test = split_data(x, y)

# Standard Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# TITANIC
x_train_titanic, x_test_titanic, y_train_titanic, y_test_titanic = split_data(x_titanic, y_titanic)


# CREDIT
x_train_credit, x_test_credit, y_train_credit, y_test_credit = split_data(x_credit, y_credit)




###### PCA ######




###### DT ######
clf = DecisionTreeClassifier()

# CIFAR-10
start_time = time.time()
clf.fit(x_train_cifar_flat, y_train_cifar.ravel())
end_time = time.time()

# FASHION-MINST
start_time = time.time()
clf.fit(x_train_fashion_flat, y_train_fashion.ravel())
end_time = time.time()

# ADULT INCOME
start_time = time.time()
clf.fit(x_train, y_train)
end_time = time.time()

# TITANIC
start_time = time.time()
clf.fit(x_train_titanic, y_train_titanic)
end_time = time.time()

# CREDIT
start_time = time.time()
clf.fit(x_train_credit, y_train_credit)
end_time = time.time()



# TITANIC

# OXFORD PETS

# CREDIT CARD


# Calculate the elapsed time
training_time = end_time - start_time
print(f"Time taken to train the model: {training_time:.2f} seconds")

y_pred = clf.predict(x_test_credit)
accuracy = accuracy_score(y_test_credit, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")



###### PCA-DT ######

###### HS-DT ######

###### PCA-HS-DT ######
