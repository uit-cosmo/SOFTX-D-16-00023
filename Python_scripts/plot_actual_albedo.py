"""
This script plots the albedo.
Just run it and you see the animation. It's basically for testing.
"""

import numpy as  np 
import netCDF4 
import datetime
from matplotlib import pyplot as plt
from netCDF4 import Dataset,num2date,date2num
import matplotlib.animation as animation
import sys


dataset_albedo = '../FortranCode/EBM/output/albedo-output_no_albedo.nc'
#dataset_albedo = '../FortranCode/EBM/output/albedo-output_albedo.nc'

geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.weight'] = 'bold'

ny = 65
nx = 128
albedo_td = netCDF4.Dataset(dataset_albedo)
#albedo = netCDF4.Dataset(dataset_no_albedo)
#T_master = netCDF4.Dataset(dataset_master)

albedo_td_values = albedo_td["co-albedo"]
#albedo_values = albedo["co-albedo"]

fps = 2000
nSeconds = 20

fig, axs = plt.subplots(1, figsize=(7,7))

im0 = axs.imshow(albedo_td_values[0,0,:,:], cmap ="RdYlBu_r", vmin = 0, vmax = 1)
axs.imshow(geography, vmin = 1.1, vmax = 2, alpha = 0.1, cmap = 'gist_gray')

#im2 = axs[1].imshow(albedo_values[0,0,:,:], cmap ="RdYlBu_r", vmin = -40, vmax = 40)
#axs[1].imshow(geography, vmin = 1.1, vmax = 2, alpha = 0.1, cmap = 'gist_gray')

for i in range(3): 
    #axs[i]
    None
time_text = axs.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top')

#time_text_2 = axs[1].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[2].transAxes)


cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label(r'Temperature $^\circ$C')

axs.set_title("Albedo time-dependent", fontsize=8, weight='bold')
#axs[1].set_title("Albedo constant", fontsize=8, weight='bold')


# animation function.  This is called sequentially
def animate(i):
    a0 = im0.get_array()
    a0 = albedo_td_values[i,0,:,:]  
    #a1 = im1.get_array()
    #a1 = albedo_values[i,0,:,:]  

    im0.set_array(a0)
    #im1.set_array(a1)

    
    time_text.set_text('time = %.1d' % i)
    
    #time_text_1.set_text('time = %.1d' % i)
    

    return [im0], [time_text,]


anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )        
plt.tight_layout()


plt.show()


