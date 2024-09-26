import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from scipy.stats import multivariate_normal

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
train_noise = np.random.normal(loc=0.0,scale=0.1,size=x_train.shape)
x_train = x_train.reshape(-1,28*28) + train_noise.reshape(-1,28*28)

test_noise = np.random.normal(loc=0.0,scale=0.1,size=x_test.shape)
x_test = x_test.reshape(-1, 28*28) + test_noise.reshape(-1,28*28)

# Getting the means and variances 
def mean_cov_class(train = x_train, label = y_train):
    mean_vec = []
    cov_vec = []
    for i in range(0,10):
        condition = label == i
        mean, cov = multivariate_normal.fit(x_train[condition,:])
        mean_vec.append(mean)
        cov_vec.append(cov)
    return mean_vec, cov_vec

mean_class, cov_class = mean_cov_class(x_train, y_train)
mean_class, cov_class = np.asarray(mean_class), np.asarray(cov_class)

# Function that classifies
def MAP(test, mean, var):
    likelihoods = np.zeros((10, len(test)))
    for k in range(0, 10):    
        likelihoods[k,:] = (multivariate_normal.logpdf(test, mean[k], var[k]))
    
    return np.argmax(likelihoods, axis=0)

##Predictions
predictions = MAP(x_test, mean_class, cov_class)

# Evaluating the performance
accuracy = class_acc(predictions,y_test)
print(f"Classification accuracy is {accuracy}")

