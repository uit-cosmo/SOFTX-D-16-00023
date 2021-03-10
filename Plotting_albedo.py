import numpy as  np 
import netCDF4 
import datetime
from matplotlib import pyplot as plt
from netCDF4 import Dataset,num2date,date2num
import matplotlib.animation as animation

dataset1 = '../FortranCode/EBM/output/timesteps-output_original.nc'
dataset2 = '../FortranCode/EBM/output/timesteps-output_no_albedo.nc'
dataset4 = '../FortranCode/EBM/output/timesteps-output_albedo.nc'
geography = np.genfromtxt('EBM/preprocess/geography_0.dat', dtype=int, delimiter=1)

ny = 65
nx = 128
timesteps = netCDF4.Dataset(dataset1)
timesteps2 = netCDF4.Dataset(dataset2)
lon = timesteps["longitude"]
lat = timesteps["latitude"]
temp = timesteps["temperature"]
temp2 = timesteps2["temperature"]

fps = 48
nSeconds = 20
temp3 = temp2
fig, ax = plt.subplots()
#im = ax.imshow(calculate_albedo(0, temp))
im = ax.imshow(temp3[0,0,:,:], cmap ="RdYlBu_r", vmin = -30, vmax = 30)
#plt.show()
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
Temp_text = ax.text(0.05, 0.8,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
fig.colorbar(im)
# initialization function: plot the background of each frame
def init():
    im.set_data(temp[0,0,:,:])
    #im.set_data(calculate_albedo(i, temp))
    time_text.set_text('0')

    return [im], [time_text,]

# animation function.  This is called sequentially
def animate(i):
    a = im.get_array()
    a = temp3[i,0,:,:]  
    #a = calculate_albedo(i, temp)# exponential decay of the values
    im.set_array(a)
    time_text.set_text('time = %.1d' % i)
    Temp_text.set_text(f'temp ={np.mean(temp3[i,0,:,:])} ')
    return [im], [time_text,]


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