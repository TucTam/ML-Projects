import numpy as np
from random import randint
values = np.zeros((10,10))

for i in range(len(values)):
    index = randint(0,9)
    values[i,index] = 1
    
print(values)
print(np.where(values==1)[1])