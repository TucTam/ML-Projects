import matplotlib.pyplot as plt
import tensorflow as tf
from random import random

# Original
mnist = tf.keras.datasets.mnist
# New
#mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the size of training and test data
print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

for i in range(x_test.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(x_test[i], cmap='gray_r')
        plt.title(f"Image {i} label num {y_test[i]} predicted {0}")
        plt.pause(1)
