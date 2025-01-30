import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from sklearn.metrics import accuracy_score
import time
from sklearn.tree import DecisionTreeClassifier


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




