import random
import numpy as np

n = 100
array = np.linspace(0, n-1, num=n)
print(array)
random.shuffle(array)
print(array)