import pdb
import numpy as np

DAT = np.loadtxt("Tracts_table_cleaned.txt", skiprows=1)

ID = np.array(DAT[:,0])
origin_lon = DAT[:,1]
origin_lat = DAT[:,2]

area = np.array(DAT[:,3])
pop = np.array(DAT[:,4])

y = np.array(DAT[:,5])
x = np.array(DAT[:,6])

density = pop/np.mean(pop)
OUT = np.column_stack((ID, x, y, density, pop, area))
np.savetxt("US_census.dat", OUT)
pdb.set_trace()
