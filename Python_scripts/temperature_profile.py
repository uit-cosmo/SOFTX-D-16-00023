"""
This script animates the longitudinal temperature profile for 
increasing CO2 levels for model w/o albedo feedback, w/ feedback and w/ feedback + time-dependent heatcapacities.
"""

import numpy as np 
import netCDF4 
from matplotlib import pyplot as plt
import matplotlib.animation as animation
dataset_master = '../FortranCode/EBM/output/timesteps-output_original.nc'
dataset_no_albedo = '../FortranCode/EBM/output/timesteps-output_no_albedo.nc'
dataset_no_albedo_no_noise = '../FortranCode/EBM/output/timesteps-output_no_albedo_no_noise.nc'
dataset_albedo = '../FortranCode/EBM/output/timesteps-output_albedo.nc'
dataset_albedo_no_noise = '../FortranCode/EBM/output/timesteps-output_albedo_no_noise.nc'
dataset_albedo_hc = '../FortranCode/EBM/output/timesteps-output_albedo_heatc.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

fps = 100
nSeconds = 20
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
temp_heatc = load_temperature(dataset_albedo_hc)
temp_no_albedo_no_noise = load_temperature(dataset_no_albedo_no_noise)
temp_albedo_no_noise = load_temperature(dataset_albedo_no_noise)


print(temp_albedo.shape)
temp_albedo_lat  =np.mean(temp_albedo[:,0,:,:], axis = 2)
temp_no_albedo_lat  =np.mean(temp_no_albedo[:,0,:,:], axis = 2)
temp_albedo_htc_lat  =np.mean(temp_heatc[:,0,:,:], axis = 2)


fig, axs = plt.subplots(3, figsize=(7,7))

axs[2].set_title("Albedo time-dependent + heatcapacities time-dependent", fontsize=8, weight='bold')
axs[0].set_title("Albedo time-dependent", fontsize=8, weight='bold')
axs[1].set_title("Albedo constant", fontsize=8, weight='bold')

im0, = axs[0].plot(temp_albedo_lat[0,:], lw = 2)
im1, = axs[1].plot(temp_no_albedo_lat[0,:], lw = 2)
im2, = axs[2].plot(temp_albedo_htc_lat[0,:], lw = 2)
for i in range(3): 
    axs[i].set_ylim([-50,50])
    axs[i].set_ylabel(r'Temperature [$^\circ$C]')
    axs[i].set_xlabel(r'Latitude index')
    axs[i].grid(ls = '--')
    
start_year = 300
def animate(i):
    #a0 = im0.set_ydata()
    a0 = temp_albedo_lat[start_year*48 + i,:]
    #a1 = im1.set_ydata()
    a1 = temp_no_albedo_lat[start_year*48 + i,:]
    #a2 = im2.set_ydata()
    a2 = temp_albedo_htc_lat[start_year*48 + i,:]
    #a = calculate_albedo(i, temp)# exponential decay of the values
    im0.set_ydata(a0)
    im1.set_ydata(a1)
    im2.set_ydata(a2)
    return [im0], [im1], [im2]


anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

fig.tight_layout()
plt.show()