import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from imodels import HSTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

#load dataset from cifar with keras library and load the dataset and store in keras directory
pic_class = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = pic_class.load_data()

#normalize pixels between 0 and 1
x_train = x_train/255.0

#flatten images
x_train_flat = x_train.reshape(-1,3072)
feat_cols = ['pixel' + str(i) for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns = feat_cols)
df_cifar['Label'] = y_train
print('Size of Data Frame: {}'.format(df_cifar.shape))


#reshape dataset and determine number of variance
x_test = x_test/255.0
x_test = x_test.reshape(-1, 32, 32, 3)
x_test_flat = x_test.reshape(-1, 3072)







pca = PCA(0.99)
pca.fit(x_train_flat)
PCA(copy = True, iterated_power = 'auto', n_components = 0.99, random_state = None, svd_solver = 'auto', tol = 0.0, whiten = False)


train_img_pca = pca.transform(x_train_flat)
test_img_pca = pca.transform(x_test_flat)

y_train_new = np_utils.to_categorical(y_train)
y_test_new = np_utils.to_categorical(y_test)

# Splitting the dataset into training and testing sets
x_train_new, x_val_new, y_train_new, y_val_new = train_test_split(train_img_pca, y_train_new, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = HSTreeClassifier()
clf.fit(x_train_new, y_train_new)

# Make predictions on the validation set
y_pred = clf.predict(x_val_new)

# Evaluate the model
accuracy = accuracy_score(y_val_new, y_pred)
print("Validation Accuracy:", accuracy)

# Test the model
test_pred = clf.predict(test_img_pca)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test_new, test_pred)
print("Test Accuracy:", test_accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_new, test_pred))
