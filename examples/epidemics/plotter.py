import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("locs.txt")
plt.scatter(dat[:,0], dat[:,1],s=0.1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()
