import matplotlib.pyplot as plt
import tensorflow as tf
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
train_noise = np.random.normal(loc=0.0,scale=0.1,size=x_train.shape)
x_train = x_train.reshape(-1,28*28) + train_noise.reshape(-1,28*28)

test_noise = np.random.normal(loc=0.0,scale=0.1,size=x_test.shape)
x_test = x_test.reshape(-1, 28*28) + test_noise.reshape(-1,28*28)

# Getting the means and variances
def mean_var_class(train = x_train, label = y_train):
    mean_vec = list()
    var_vec = list()
    for i in range(0,10):
        condition = label == i
        mean = np.mean(train[condition,:], axis=0)
        mean_vec.append(mean)

        var = np.var(train[condition,:], axis=0)
        var_vec.append(var)
    return mean_vec, var_vec
    
mean_class, var_class = mean_var_class(x_train, y_train)
mean_class, var_class = np.asarray(mean_class), np.asarray(var_class)

def loglikelihood(sample_feats, mean, var, k):
    pi = np.pi
    constant = np.log(2*pi)
    term1 = np.log(var[k,:])
    term2 = np.square(sample_feats-mean[k,:])/var[k,:]
    sum = -1/2*np.sum(constant + term1 + term2)    
    return sum

def logMLE(train, mean, var, i):
    likelihoods = []
    for d in range(0, 10):    
        row = train[i,:]
        likelihoods.append(loglikelihood(row, mean, var, d))
    
    return np.argmax(likelihoods)

# Testing the models
predictions = []
for sample in range(0, len(x_test)):
    preds = logMLE(x_test,mean_class,var_class, sample)
    predictions.append(preds)
    
# Evaluating the performance
accuracy = class_acc(predictions,y_test)
print(f"Classification accuracy is {accuracy}")

