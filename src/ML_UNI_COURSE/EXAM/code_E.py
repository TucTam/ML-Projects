import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train.dat", encoding="utf8")
n_males = len(data[data[:,2]==0])
malecond = np.logical_and(data[:, 1] == 1, data[:,2] == 0)
survivedmale = len(data[malecond])
male_rate = survivedmale/n_males

n_females = len(data[data[:,2] == 1])
femalecond = np.logical_and(data[:, 1] == 1, data[:,2] == 1)
survivedfemale = len(data[femalecond])
female_rate = survivedfemale/n_females

print(f"Out of {n_males} male passengers {survivedmale} survived and {n_males-survivedmale} died (surveillance rate {male_rate*100:.2f}%)")
print(f"Out of {n_females} male passengers {survivedfemale} survived and {n_females-survivedfemale} died (surveillance rate {female_rate*100:.2f}%)")