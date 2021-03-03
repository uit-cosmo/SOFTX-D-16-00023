import numpy as np 
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import netCDF4 
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg'] # you need the ffmpeg encoder to save the animation
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


dataset = '../FortranCode/EBM/output/timesteps-output2.nc'

timesteps = netCDF4.Dataset(dataset)
lon = timesteps["longitude"]
lat = timesteps["latitude"]
temp = timesteps["temperature"]
lons, lats = np.meshgrid(lon, lat)
fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(1, 1, 1, projection=ccrs.EckertIII())



#ax.set_global()
#origin = 'lower'
line_c2 = ax.contour(lons, lats,  temp[1,0,:,:], levels=[-40,-30,-20,-10,0,10,20,30],
                        colors=['black'],
                        transform=ccrs.PlateCarree())
line_c = ax.contourf(lons, lats,  temp[1,0,:,:], transform=ccrs.PlateCarree())
clabel = ax.clabel(
    line_c,  # Typically best results when labelling line contours.
    colors=['None'],
    manual=False,  # Automatic placement vs manual placement.
    inline=True,  # Cut the line where the label will be placed.
    fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    )
def init_run(): 
    ax.coastlines(zorder=3)
    ax.stock_img()
    ax.gridlines()
    crs = ccrs.PlateCarree()
    
    #extent = (-177.1875, 180, -90, 90)
    
class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

def main(i):
    global line_c2
    global clabel
        
    for c in line_c2.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    for label in line_c2.labelTexts:
        label.remove() # removes only the contour labels, leaves the rest intact
        

    temp_0 = temp[i,0,:,:]
    line_c = ax.contourf(lons, lats,  temp_0, transform=ccrs.PlateCarree())
    line_c2 = ax.contour(lons, lats,  temp_0, levels=[-40,-30,-20,-10,0,10,20,30],
                        colors=['black'],
                        transform=ccrs.PlateCarree())

    clabel = ax.clabel(
    line_c2,  # Typically best results when labelling line contours.
    colors=['black'],
    manual=False,  # Automatic placement vs manual placement.
    inline=True,  # Cut the line where the label will be placed.
    fmt=' {:.0f} '.format,  # Labels as integers, with some extra space.
    )


    print(i)

    return line_c, line_c2, clabel

    

timesteps = 100 # steps between animation frames

ani = animation.FuncAnimation(fig, main, init_func=init_run, frames  = np.arange(0,temp.shape[0], timesteps), 
                             interval=200, blit=False, save_count=100, repeat  = True)

ani.save("../FortranCode/test.mp4", writer = writer)
