import numpy as np
import pdb
#dat = np.loadtxt('locs.txt')
num = 1000
x = np.arange(0, 1, 1./num)
y = np.arange(0, 1, 1./num)
X = []
Y = []
for i in range(num):
	for j in range(num):
		X.append(x[i])
		Y.append(y[j])
ones = np.ones((num*num,1))
DAT = np.column_stack((ones, X, Y, ones, ones, ones))
np.savetxt('uniform_data.dat', DAT)
pdb.set_trace()
