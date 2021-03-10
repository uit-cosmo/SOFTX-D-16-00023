import numpy as np 
import netCDF4 
from matplotlib import pyplot as plt
dataset_master = '../FortranCode/EBM/output/timesteps-output_original.nc'
dataset_no_albedo = '../FortranCode/EBM/output/timesteps-output_no_albedo.nc'
dataset_albedo = '../FortranCode/EBM/output/timesteps-output_albedo.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)


area = np.load('EBM/output/area_numpy.npz')
area_pole = area['area_pole']
area_wo_poles = area['area_wo_poles']
area_wo_poles = area_wo_poles.reshape(-1,128)
earth_area = 510100000

#plt.imshow(area_wo_poles.reshape(-1,128))
#plt.show()
def load_temperature(dataset_directory):
    T = netCDF4.Dataset(dataset_directory)
    return T["temperature"]

temp_albedo = load_temperature(dataset_albedo)
temp_no_albedo = load_temperature(dataset_no_albedo)
temp_master = load_temperature(dataset_master)
print(temp_master)
#print(temp_master[:,0,1:-1,:].shape)
#print((temp_master[0,0,1:-1,:] * area_wo_poles).sum()/area_wo_poles.sum())


def calculate_weighted_temp(temp): 
    temp_weighted_poles = (temp[:, 0, 0, 0] + temp[:, 0, -1, 0]) * area_pole # poles
    temp_weighted_wo_poles = np.sum(temp[:, 0, 1:-1, :] * area_wo_poles , axis=(1,2)) # non-poles
    return temp_weighted_poles + temp_weighted_wo_poles

def calculate_SIA(temp_array):
    SIA_array = np.zeros_like(temp_array[:, 0, 1:-1, :]) 
    pole_condition = np.zeros_like(temp_array[:, 0, 0, 0])
    pole_condition[temp_array[:, 0, 0, 0] <= -1] = 1
    print(SIA_array.shape)
    SIA_mask = np.zeros_like(SIA_array)
    temp_array = temp_array[:, :, :, :]

    SIA_mask[(temp_array[:, 0, 1:-1, :] <= -1) & (geography[1:-1, :] == 2)] = 1 #sea ice

    print((area_wo_poles*earth_area).sum())
    SIA_array = SIA_mask * area_wo_poles * earth_area
    plt.imshow(SIA_array[20000, :, :])
    plt.imshow(temp_array[20000, 0,:, :])

    plt.colorbar()
    plt.show()
    SIA = SIA_array[:, 0:32, :].sum(axis = (1,2)) +  pole_condition * area_pole * earth_area
    print(SIA)
    return SIA

SIA = calculate_SIA(temp_no_albedo)
#plt.plot(calculate_weighted_temp(temp_albedo), SIA)
def save_SIA_against_T(temp_array): 
    T_against_SIA = np.vstack((calculate_weighted_temp(temp_array), calculate_SIA(temp_array)))
    np.savetxt(f'../FortranCode/EBM/output/SIA/fortran_T_SIA_albd1_a{535}.txt', T_against_SIA, fmt= "%e", delimiter=',')
        
#save_SIA_against_T(temp_albedo)

#plt.plot((calculate_weighted_temp(temp_albedo))[37:-48])
#plt.plot((calculate_weighted_temp(temp_no_albedo))[37:-48])
#plt.plot((calculate_weighted_temp(temp_master))[37:-48])

plt.show()


