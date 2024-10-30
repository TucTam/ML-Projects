import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train.dat", encoding="utf8")

survived = data[data[:, 1] == 1, 4]
dead = data[data[:, 1] == 0,4]

plt.figure(figsize=(10, 6))
plt.hist(survived,bins=30, color='blue', alpha=0.5, label='Survived')
plt.hist(dead,bins=30, color='red', alpha=0.5, label='Dead')
plt.xlabel("Fares by the survivors")
plt.ylabel("Frequency")
plt.legend()
plt.show()