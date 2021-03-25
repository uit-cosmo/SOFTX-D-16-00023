############################################################
# This script creates a data for the initial temperature   #
# and saves it as .dat to load it in Fortran90.            #
############################################################

import numpy as  np 
import netCDF4 

dataset = '../FortranCode/EBM/output/timesteps-output_albedo.nc' # here you should think about which temperature you chose
temperature_nc = netCDF4.Dataset(dataset)
temp = temperature_nc["temperature"]

def save_data(data):
    """
    Save the ocean-land mask array. 
    """ 
    np.savetxt('EBM/preprocess/init_temp.dat', data, fmt= "%f",delimiter = ',')
    
save_data(temp[98*48,0,:,:]) #take one of the last temperature values (so "converged")