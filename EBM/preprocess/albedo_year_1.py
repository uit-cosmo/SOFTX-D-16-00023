"""
This function creates the input albedo AND geography for the Fortran code. 
It loads the temperature from one Fortran output and calculates the albedo in the same form as in the 
Fortran model then it gets saved as netCDF file. The same for the geography (ice distribution).

!!!IMPORTANT: The files get saved in /FortranCode, you have to move them to EBM/input- after running this script!!!
"""

import numpy as  np 
import netCDF4 
import datetime
from matplotlib import pyplot as plt
from netCDF4 import Dataset,num2date,date2num
import matplotlib.animation as animation
dataset = '../FortranCode/EBM/output/timesteps-output_albedo.nc'
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
        temp_mask[(temp[98*48 + t, :, :][0, :, :] <= -1) & (geography == 2)] = 1#1 # sea ice
        temp_mask[(temp[98*48 + t, :, :][0, :, :] > -1) & (geography == 2)] = 0 # sea without ice
        #temp_mask[(temp[0*48 + t, :, :][0, :, :] <= -5) & (geography == 1)] = 1 # land ice
        #temp_mask[(temp[0*48 + t, :, :][0, :, :] > -5) & (geography == 1)] = 0 # land without ice

        # land without ice 
        #albedo[t, (geography == 1) & (temp_mask == 0)] = 0.3 + 0.09 * legendre(np.where((geography == 1) & (temp_mask == 0))[0])
        # sea ice
        albedo[t, (geography == 2) & (temp_mask == 1)] = 0.6
        # land with ice 
        #albedo[t, (geography == 1) & (temp_mask == 1)] = 0.7
        # ocean without ice 
        albedo[t, (geography == 2) & (temp_mask == 0)] = 0.29 + 0.09 * legendre(np.where((geography == 2) & (temp_mask == 0))[0])
    return albedo

def define_geography_1st_year(): 
    geo = np.zeros((ny,nx))
    
    geo[(temp[98*48, :, :][0, :, :] <= -1) & (geography == 2)] = 2 # sea ice
    geo[(temp[98*48, :, :][0, :, :] > -1) & (geography == 2)] = 5 # sea without ice
    #geo[(temp[48, :, :][0, :, :] <= -5) & (geography == 1)] = 3 # land ice
    #geo[(temp[48, :, :][0, :, :] > -5) & (geography == 1)] = 1 # land without ice

    return geo
    

def calculate_albedo(t, temperature): 
    #unused function in this script
    albedo = np.zeros((ny, nx))
    temp_mask = np.zeros((ny,nx))
    temp_mask[temp[t, :, :][0, :, :] <= -1] = 1
    temp_mask[temp[t, :, :][0, :, :] > -1] = 0
        

    # land without ice 
    #albedo[(geography == 1) & (temp_mask == 0)] = 0.3 + 0.09 * legendre(np.where((geography == 1) & (temp_mask == 0))[0])
    # sea ice
    albedo[(geography == 2) & (temp_mask == 1)] = 0.6
    # land with ice 
    #albedo[(geography == 1) & (temp_mask == 1)] = 0.7
    # ocean without ice 
    albedo[(geography == 2) & (temp_mask == 0)] = 0.29 + 0.09 * legendre(np.where((geography == 2) & (temp_mask == 0))[0])
    return albedo
albedo = define_albedo_1st_year()
geo = define_geography_1st_year()

def save_netCDF_albedo():
    geography
    nsteps = 48;
    unout = 'UNLIMITED'

    datesout = [datetime.datetime(1+iyear,1,1) for iyear in range(nsteps)]; # create datevalues

    ncout = Dataset('albedo_year_1.nc','w','NETCDF4'); 
    ncout.createDimension('level',1)
    ncout.createDimension('latitude',ny)
    ncout.createDimension('longitude',nx)
    ncout.createDimension('time', None);
    
    latvar = ncout.createVariable('latitude','float32',('latitude'));
    latvar.setncattr('units','degrees_north')
    latvar[:] = np.array(lat)

    lonvar = ncout.createVariable('longitude','float32',('longitude'));
    lonvar.setncattr('units','degrees_east')
    lonvar[:] = np.array(lon)
    
    myvar = ncout.createVariable('ALBEDO','float32',('time', 'level', 'latitude', 'longitude'));myvar.setncattr('units','dimensionless');
    myvar[:] = albedo;
    
    ncout.close();
    
def save_netCDF_geo():
    
    unout = 'UNLIMITED'

    
    ncout = Dataset('geo_year_1.nc','w','NETCDF4'); 
    ncout.createDimension('latitude',ny)
    ncout.createDimension('longitude',nx)
    
    latvar = ncout.createVariable('latitude','float32',('latitude'));
    latvar.setncattr('units','degrees_north')
    latvar[:] = np.array(lat)

    lonvar = ncout.createVariable('longitude','float32',('longitude'));
    lonvar.setncattr('units','degrees_east')
    lonvar[:] = np.array(lon)
    
    myvar = ncout.createVariable('landmask','float32',('latitude', 'longitude'));myvar.setncattr('units','landmask: 1. land;  2. sea ice; 3. land ice; 5 ocean. ');
    myvar[:] = geo;
    
    ncout.close();  
save_netCDF_geo()

save_netCDF_albedo()
