import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train.dat", encoding="utf8")

malecondlive = np.logical_and(data[:, 1] == 1, data[:,2] == 0)
survivedmale = data[malecondlive, 4]

maleconddie = np.logical_and(data[:, 1] == 0, data[:,2] == 0)
deadmale = data[maleconddie, 4]

femalecondlive = np.logical_and(data[:, 1] == 1, data[:,2] == 1)
survivedfemale = data[femalecondlive, 4]

femaleconddie = np.logical_and(data[:, 1] == 0, data[:,2] == 1)
deadfemale = data[femaleconddie, 4]

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.hist(survivedmale,bins=30, color='blue', alpha=0.5, label='Male survivors')
plt.hist(deadmale, bins=30, color='red', alpha=0.5, label='Male dead')
plt.xlabel("Fares of male survivors and dead")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1,2,2)
plt.hist(survivedfemale,bins=30, color='green', alpha=0.5, label='Female Survivors')
plt.hist(deadfemale, bins=30, color='yellow', alpha=0.5, label='Female dead')
plt.xlabel("Fares of females survivors and dead")
plt.ylabel("Frequency")
plt.legend()
plt.show()