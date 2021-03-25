#################################################
# This script creates a land-ocean mask array   #
# and saves it as .dat to load it in Fortran90. #
#################################################

import numpy as np 

data = np.genfromtxt('EBM/preprocess/The_World.dat', dtype=int, delimiter=1)

def convert_data_land_ocean(data): 
    """
    Converts the data of the array defining the world into binary array with 2 = ocean (waterplanet). 
    In the original data:  1 = land;  2 = sea ice; 3 = land ice; 5 = ocean.
    """
    data[:, :] = 2 # waterplanet = everything is water (2)
    
    return data

def save_data(data):
    """
    Save the ocean-land mask array. 
    """ 
    np.savetxt('EBM/preprocess/geography_0.dat', data, fmt= "%i",delimiter = '')
    
data = convert_data_land_ocean(data)
save_data(data)
