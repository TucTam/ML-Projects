import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train.dat", encoding="utf8")

n_passengers = len(data)
n_survived = len(data[data[:, 1] == 1])
n_dead = len(data[data[:, 1] == 0])
print(f"Out of {n_passengers} passengers {n_survived} ({n_survived/n_passengers*100:.2f}%) survived and {n_dead} ({n_dead/n_passengers*100:.2f}%) died.")