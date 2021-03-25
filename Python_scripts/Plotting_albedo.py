"""
This script plots the temperature actually not the albedo. But you can plot the initial albedo with the function at the bottom.
Just run it and you see the animation. It's basically for testing.
"""


import numpy as  np 
import netCDF4 
import datetime
from matplotlib import pyplot as plt
from netCDF4 import Dataset,num2date,date2num
import matplotlib.animation as animation
import sys

dataset_master = '../FortranCode/EBM/output/timesteps-output_original.nc'
dataset_no_albedo = '../FortranCode/EBM/output/timesteps-output_no_albedo.nc'
dataset_albedo = '../FortranCode/EBM/output/timesteps-output_albedo.nc'
dataset_albedo_hc = '../FortranCode/EBM/output/timesteps-output_albedo_heatc.nc'
#dataset_master = '../FortranCode/EBM/output/timesteps-output2.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'

ny = 65
nx = 128
T_albedo_hc = netCDF4.Dataset(dataset_albedo_hc)
T_albedo = netCDF4.Dataset(dataset_albedo)
T_no_albedo = netCDF4.Dataset(dataset_no_albedo)
#T_master = netCDF4.Dataset(dataset_master)

temp_albedo = T_albedo["temperature"]
temp_albedo_hc = T_albedo_hc["temperature"]
temp_no_albedo = T_no_albedo["temperature"]

fps = 2000
nSeconds = 20

fig, axs = plt.subplots(3, figsize=(7,7))

im0 = axs[0].imshow(temp_albedo_hc[0,0,:,:], cmap ="RdYlBu_r", vmin = -40, vmax = 40)
axs[0].imshow(geography, vmin = 1, vmax = 2, alpha = 0.1, cmap = 'gist_gray')

im1 = axs[1].imshow(temp_albedo[0,0,:,:], cmap ="RdYlBu_r", vmin = -40, vmax = 40)
axs[1].imshow(geography, vmin = 1.1, vmax = 2, alpha = 0.1, cmap = 'gist_gray')

im2 = axs[2].imshow(temp_no_albedo[0,0,:,:], cmap ="RdYlBu_r", vmin = -40, vmax = 40)
axs[2].imshow(geography, vmin = 1.1, vmax = 2, alpha = 0.1, cmap = 'gist_gray')

for i in range(3): 
    #axs[i]
    None
time_text = axs[0].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[0].transAxes)
Temp_text = axs[0].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[0].transAxes)
time_text_1 = axs[1].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[1].transAxes)
Temp_text_1 = axs[1].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[1].transAxes)
time_text_2 = axs[2].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[2].transAxes)
Temp_text_2 = axs[2].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[2].transAxes)

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label(r'Temperature $^\circ$C')
axs[0].set_title("Albedo time-dependent + heatcapacities time-dependent", fontsize=8, weight='bold')
axs[1].set_title("Albedo time-dependent", fontsize=8, weight='bold')
axs[2].set_title("Albedo constant", fontsize=8, weight='bold')


# animation function.  This is called sequentially
def animate(i):
    a0 = im0.get_array()
    a0 = temp_albedo_hc[i,0,:,:]  
    a1 = im1.get_array()
    a1 = temp_albedo[i,0,:,:]  
    a2 = im2.get_array()
    a2 = temp_no_albedo[i,0,:,:]  
    #a = calculate_albedo(i, temp)# exponential decay of the values
    im0.set_array(a0)
    im1.set_array(a1)
    im2.set_array(a2)
    
    time_text.set_text('time = %.1d' % i)
    Temp_text.set_text(f'temp ={np.mean(temp_albedo_hc[i,0,:,:]):.2f} ')
    
    time_text_1.set_text('time = %.1d' % i)
    Temp_text_1.set_text(f'temp ={np.mean(temp_albedo[i,0,:,:]):.2f} ')
    
    time_text_2.set_text('time = %.1d' % i)
    Temp_text_2.set_text(f'temp ={np.mean(temp_no_albedo[i,0,:,:]):.2f} ')
    return [im0], [im1], [im2], [time_text,], [time_text_1,], [Temp_text_1,], [time_text_2,], [Temp_text_2,]


anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )        
plt.tight_layout()


plt.show()


def plot_initial_albedo(): 
    """
    This function plots the original input albedo (for constant albedo simulation) and my modified albedo file.
    """
    albedo_input = '/home/nils/Documents/Models/FortranCode/EBM/input/albedo.nc'
    albedo_modified = '/home/nils/Documents/Models/FortranCode/EBM/input/albedo_year_1.nc'
    albedo_input_dat = netCDF4.Dataset(albedo_input)
    albedo_modified_dat = netCDF4.Dataset(albedo_modified)


    alb_in = albedo_input_dat["ALBEDO"]
    alb_mod = albedo_modified_dat["ALBEDO"]

    plt.imshow(alb_in[0,0,:,:])
    plt.show()
    plt.imshow(alb_mod[0,0,:,:])
    plt.show()
