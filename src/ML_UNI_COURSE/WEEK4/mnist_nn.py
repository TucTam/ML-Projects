import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import argparse

# accuracy performance evaluation function
def class_acc(pred,gt):
    TP = np.sum([pred == gt])
    acc = TP / len(pred)
    return acc

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Choose MNIST dataset to train the model.")
parser.add_argument(
    'dataset', 
    choices=['fashion', 'original'], 
    help="Choose 'fashion' for Fashion MNIST, 'original' for number MNIST"
)

args = parser.parse_args()
mnist_data = None
# Fetching the data and splitting into training and testing sets
if args.dataset == "original":
    mnist_data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_data.load_data()
else:
    mnist_data = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist_data.load_data()    


# Reshaping the training and testing sets
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1, 28*28)

# Creating the model and initializing it
KNN_model = KNeighborsClassifier(n_neighbors=1)

# Training the model
KNN_model.fit(x_train,y_train)

# Testing the model
predictions = KNN_model.predict(x_test)

# Evaluating the performance
accuracy = class_acc(predictions,y_test)
print(f"Classification accuracy is {accuracy}")

