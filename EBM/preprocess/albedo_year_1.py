import numpy as  np 
import netCDF4 
from matplotlib import pyplot as plt

dataset = '../FortranCode/EBM/output/timesteps-output2.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

ny = 65
nx = 128
timesteps = netCDF4.Dataset(dataset)
lon = timesteps["longitude"]
lat = timesteps["latitude"]
temp = timesteps["temperature"]
legendre = lambda x : 0.5*(3*np.sin((90.0-(x-1)*2.8125)*3.1415926/180.0)*np.sin((90.0-(x-1)*2.8125)*3.1415926/180.0)-1)

def define_albedo_1st_year(): 
    albedo = np.zeros((48, ny,nx))
    temp_mask = np.zeros((ny,nx))
    for t in range(48): 
        temp_mask[temp[t, :, :][0, :, :] <= -1] = 1
        temp_mask[temp[t, :, :][0, :, :] > -1] = 0
        

        # land without ice 
        albedo[t, (geography == 1) & (temp_mask == 0)] = 0.3 + 0.09 * legendre(np.where((geography == 1) & (temp_mask == 0))[0])
        # sea ice
        albedo[t, (geography == 2) & (temp_mask == 1)] = 0.6
        # land with ice 
        albedo[t, (geography == 1) & (temp_mask == 1)] = 0.7
        # ocean without ice 
        albedo[t, (geography == 2) & (temp_mask == 0)] = 0.29 + 0.09 * legendre(np.where((geography == 2) & (temp_mask == 0))[0])
    #np.savetxt('EBM/preprocess/albedo_year_1.dat', albedo, fmt= "%i",delimiter = '')
    return albedo
albedo = define_albedo_1st_year()
print(albedo)

plt.imshow(albedo[0,:,:])
plt.colorbar()
plt.show()
