import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from imodels import HSTreeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.datasets import cifar10


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data - flatten images and normalize data
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0


# PCA
pca = PCA(0.99)
pca.fit(x_train_flat)
x_train_flat = pca.transform(x_train_flat)
x_test_flat = pca.transform(x_test_flat)


# Decision Tree Model
DT = DecisionTreeClassifier()
DT.fit(x_train_flat, y_train)

# HS Model
HS = HSTreeClassifierCV()
HS.fit(x_train_flat, y_train)


# Predictions
DT_predictions = DT.predict(x_test_flat)
HS_predictions = HS.predict(x_test_flat)


# Calculate accuracy
DT_accuracy = accuracy_score(y_test, DT_predictions)
HS_accuracy = accuracy_score(y_test, HS_predictions)
print("Decision Tree Accuracy:", DT_accuracy)
print("Hierarchical Shrinkage Accuracy:", HS_accuracy)










# Plotting cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
#plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.savefig('PCA')
plt.show()
