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
#dataset_master = '../FortranCode/EBM/output/timesteps-output2.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

ny = 65
nx = 128
T_albedo = netCDF4.Dataset(dataset_albedo)
T_no_albedo = netCDF4.Dataset(dataset_no_albedo)
T_master = netCDF4.Dataset(dataset_master)

#lon = timesteps["longitude"]
#lat = timesteps["latitude"]
temp_albedo = T_albedo["temperature"]
temp_no_albedo = T_no_albedo["temperature"]
temp_master = T_master["temperature"]

print(np.mean(temp_master[:,0,:,:], axis =(1,2))[100], np.mean(temp_no_albedo[:,0,:,:], axis =(1,2))[100])
#sys.exit()
fps = 48
nSeconds = 20
#fig, ax = plt.subplots()
#im = ax.imshow(calculate_albedo(0, temp))
fig, axs = plt.subplots(3)

im0 = axs[0].imshow(temp_albedo[0,0,:,:], cmap ="RdYlBu_r", vmin = -30, vmax = 30)
im1 = axs[1].imshow(temp_no_albedo[0,0,:,:], cmap ="RdYlBu_r", vmin = -30, vmax = 30)
im2 = axs[2].imshow(temp_master[0,0,:,:], cmap ="RdYlBu_r", vmin = -30, vmax = 30)
#plt.show()
time_text = axs[0].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[0].transAxes)
Temp_text = axs[0].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[0].transAxes)
time_text_1 = axs[1].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[1].transAxes)
Temp_text_1 = axs[1].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[1].transAxes)
time_text_2 = axs[2].text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=axs[2].transAxes)
Temp_text_2 = axs[2].text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=axs[2].transAxes)

fig.colorbar(im0)
# initialization function: plot the background of each frame
"""
def init():
    im.set_data(temp[0,0,:,:])
    #im.set_data(calculate_albedo(i, temp))
    time_text.set_text('0')

    return [im], [time_text,]
"""
# animation function.  This is called sequentially
def animate(i):
    a0 = im0.get_array()
    a0 = temp_albedo[i,0,:,:]  
    a1 = im1.get_array()
    a1 = temp_no_albedo[i,0,:,:]  
    a2 = im2.get_array()
    a2 = temp_master[i,0,:,:]  
    #a = calculate_albedo(i, temp)# exponential decay of the values
    im0.set_array(a0)
    im1.set_array(a1)
    im2.set_array(a2)
    
    time_text.set_text('time = %.1d' % i)
    Temp_text.set_text(f'temp ={np.mean(temp_albedo[i,0,:,:])} ')
    
    time_text_1.set_text('time = %.1d' % i)
    Temp_text_1.set_text(f'temp ={np.mean(temp_no_albedo[i,0,:,:])} ')
    
    time_text_2.set_text('time = %.1d' % i)
    Temp_text_2.set_text(f'temp ={np.mean(temp_master[i,0,:,:])} ')
    return [im0], [im1], [im2], [time_text,], [time_text_1,], [Temp_text_1,], [time_text_2,], [Temp_text_2,]


anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

plt.show()
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
"""