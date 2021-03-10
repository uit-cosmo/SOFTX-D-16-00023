import numpy as  np 
import netCDF4 
import datetime
from matplotlib import pyplot as plt
from netCDF4 import Dataset,num2date,date2num
import matplotlib.animation as animation



ny = 65
nx = 128



geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)
geo_original = '/home/nils/Documents/Models/FortranCode/EBM/input/geography.nc'
geo_modifided = '/home/nils/Documents/Models/FortranCode/EBM/input/geo_year_1.nc'
geo_original_dat = netCDF4.Dataset(geo_original)
geo_modifided_dat = netCDF4.Dataset(geo_modifided)

fig, axs = plt.subplots(3)

geo_orig = geo_original_dat["landmask"]
geo_mod = geo_modifided_dat["landmask"]

axs[0].imshow(geo_orig[:,:])

axs[1].imshow(geo_mod[:,:])
axs[2].imshow(geography[:,:])
plt.show()
