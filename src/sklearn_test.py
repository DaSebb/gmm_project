from sklearn import mixture
import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.random.rand(20)
plt.plot(x, '*-', color='red', markersize=10)

plt.show()