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
from keras.datasets import cifar10


#load dataset from cifar with keras library and load the dataset and store in keras directory
pic_class = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = pic_class.load_data()

#print the shape of training, testing, and label data
# print('Training Data Shape: ', x_train.shape)
# print('Testing Data Shape: ', x_test.shape)
#
# print('Label Training Data Shape: ', y_train.shape)
# print('Label Testing Data Shape: ', y_test.shape)


#normalize pixels between 0 and 1
x_train = x_train/255.0



#flatten images
x_train_flat = x_train.reshape(-1,3072)
feat_cols = ['pixel' + str(i) for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns = feat_cols)
df_cifar['Label'] = y_train

#create PCA method
# pca_cifar = PCA(n_components = 2)
# principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:, :-1])
#
# #convert principal components
# principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar,
#                                   columns = ['Principal Component 1', 'Principal Component 2'])
# principal_cifar_Df['Label'] = y_train
# principal_cifar_Df.head()


pca = PCA(0.99)
pca.fit(x_train_flat)
PCA(copy = True, iterated_power = 'auto', n_components = 0.99, random_state = None, svd_solver = 'auto', tol = 0.0, whiten = False)


# Plotting cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
#plt.title('PCA Explained Variance Ratio')
plt.grid(True)
plt.savefig('PCA')
plt.show()






# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Preprocess data - flatten images and normalize data
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

# Decision Tree Model
model = DecisionTreeClassifier()
history = model.fit(x_train_flat, y_train)

# HS Model
model = HSTreeClassifierCV()
history = model.fit(x_train_flat, y_train)


# Predictions
predictions = model.predict(x_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
