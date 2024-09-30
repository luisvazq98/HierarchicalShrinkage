###############################################
#
# LIBRARIES AND GLOBAL VARIABLES
#
###############################################

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, fashion_mnist
import tensorflow_datasets as tfds
from sklearn.datasets import load_iris
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score

TITANIC_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
CREDIT_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'



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

dataset, info = tfds.load('oxford_iiit_pet', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)

###############################################
#
# FUNCTIONS FOR PREPROCESSING DATA
#
###############################################

###### SPLITTING DATASETS (80 TRAINING, 20 TESTING) ######
train_dataset = dataset[0]
test_dataset = dataset[1]

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

###### PCA ######





###############################################
#
# DEVELOPMENT AND TRAINING OF MODELS
#
###############################################

###### LOADING DATA AND PROCESSING DATA ######
x_train_images, y_train_images, x_test_images, y_test_images = load_data_image(cifar10)

# Flatten images
x_train_images_flat = x_train_images.reshape(x_train_images.shape[0], -1)
x_test_images_flat = x_test_images.reshape(x_test_images.shape[0], -1)

# Normalize images
x_train_images_flat = x_train_images_flat / 255.0
x_test_images_flat = x_test_images_flat / 255.0


###### SPLITTING DATA ######




###### DT ######
clf = DecisionTreeClassifier()
clf.fit(x_train_images_flat, y_train_images.ravel())

y_pred = clf.predict(x_test_images_flat)
accuracy = accuracy_score(y_test_images, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("hello")

###### PCA-DT ######

###### HS-DT ######

###### PCA-HS-DT ######
