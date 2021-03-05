from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from area import area
from tqdm import tqdm # not required, just for progress bar plotting
from pyproj import Proj
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

def next_lon(longitudes, idx):
    all = len(longitudes)
    wrap_around = 0 if idx+1 < len(longitudes) else 360
    return longitudes[(idx+1)%all] + wrap_around
    # return longitudes[min(len(longitudes), idx+1)]
    # return longitudes[idx+1] if idx + 1 < len(longitudes) else None

def next_lat(latitudes, idx):
    all = len(latitudes)
    return latitudes[min(idx+1, all-1)]
    # return latitudes[idx+1] if idx + 1 < len(latitudes) else None   # not wrap latitude, as good as the above

def get_lower_right_quad_vertices(latitudes, longitudes, idx_lat, idx_lon):
    # todo: see comment in Zhuang Fortran EBM.f90 about how grid is structured, vs what netcdf dump shows (discrepancy)
    # todo: as of now south pole not included anywhere (total area of lat -90 is zero), because there is nothing more south to take.
    #       and the north pole decides all adjacent quadrants :/. Maybe interpolate the values at grid potint to middle of grids and compute further from those...
    return latitudes[idx_lat], next_lat(latitudes, idx_lat), longitudes[idx_lon], next_lon(longitudes, idx_lon)

def area_1quad_sphere(lat1, lat2, lon1, lon2):
    # if lat2 == None or lon2 == None:
    #     return 0
    # else:
    obj = {'type': 'Polygon', 'coordinates': [[[lon1, lat1], [lon1, lat2], [lon2, lat2], [lon2, lat1], [lon1, lat1]]]}
    return area(obj)  # in m2

def area_quads_sphere(latitudes, longitudes, indices_lat, indices_lon):
    if len(indices_lat) != len(indices_lon):
        raise IndexError("Number of latitude and longitude coordinates differs.")
    areas = np.empty(len(indices_lat))
    for i in range(0, len(indices_lat)):
        areas[i] = area_1quad_sphere(*get_lower_right_quad_vertices(latitudes, longitudes, indices_lat[i], indices_lon[i]))
    # areas = np.fromiter((area_1quad_sphere(*get_lower_right_quad_vertices(latitudes, longitudes, indices_lat[i], indices_lon[i])) for i in range(0, len(indices_lon))), dtype='float')
    return np.sum(areas)  # in m2


def total_earth_area():
    obj = {'type': 'Polygon',
           'coordinates': [[[-180, -90], [-180, 90], [180, 90], [180, -90], [-180, -90]]]}  # full earth area
    return area(obj)  # in km2

    # from functools import partial
    # from shapely.geometry.polygon import Polygon
    # geom = Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)])
    # geom_aea = transform(
    #     partial(
    #         pyproj.transform,
    #         pyproj.Proj(init='epsg:4326'),
    #         pyproj.Proj(
    #             proj='aea',
    #             lat1=geom.bounds[1],
    #             lat2=geom.bounds[3])),
    #     geom)
    # return geom_aea.area

    # co = {"type": "Polygon", "coordinates": [
    #     [(-180, 90),
    #      (-180, -90),
    #      (180, -90),
    #      (180, 90)]]}
    # lon, lat = zip(*co['coordinates'][0])
    # pa = Proj("+proj=aea +lat_1=-180 +lat_2=180 +lat_0=39.0 +lon_0=-106.55")
    # x, y = pa(lon, lat)
    # cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
    # return shape(cop).area  # 268952044107.43506


def align_year_vernal_eqx(data, w_in_y=48, w_in_y_before_vernal_eqx=11, remove_trailing0_y=True):
    '''Original Zhuang model: 48 weeks. The astronomical calculations of the Earth's orbit begin at the vernal equinox. The first time step is at the vernal equinox.
    Month     Time Steps
    Jan    38, 39, 40, 41
    Feb    42, 43, 44, 45
    Mar    46, 47, 48   1
    Apr     2,  3,  4,  5
    May     6,  7,  8,  9
    Jun    10, 11, 12, 13
    Jul    14, 15, 16, 17
    Aug    18, 19, 20, 21
    Sep    22, 23, 24, 25
    Oct    26, 27, 28, 29
    Nov    30, 31, 32, 33
    Dec    34, 35, 36, 37'''

    # remove first 37 and last 11 - so we loose 1 yr
    aligned_data = data[w_in_y-w_in_y_before_vernal_eqx:-w_in_y_before_vernal_eqx,:,:,:]
    if remove_trailing0_y:
        while np.all(aligned_data[-1,:,:,:] == 0.0): # aligned_data[-1] == 0.0:
            aligned_data = aligned_data[:np.shape(aligned_data)[0]-w_in_y,:,:,:]
        # aligned = aligned[:len(aligned)-w_in_y]
    return aligned_data


def test_weighted_temp(temp):
    '''testing weighted temp  - it will not be exactly the same, cause in equally-weighted mean south pole counted |lon vals| times, and in weighted one zero times'''
    # todo: remove
    areas_grid = np.ones((len(lats), len(lons)),
                         dtype=float)  # equal weithts to compare old equally weighted, and now weighted
    tg_w_old = np.mean(temps[:, 0, :-1, :], axis=(1, 2))  # all except south pole, equally weighted
    tg_w_new = np.empty(np.shape(temps)[0], dtype='float')
    for k in range(0, np.shape(temps)[0]):
        tg_w_new[k] = np.sum(temps[k, 0, :, :] * areas_grid[:, :])  # weigh temperatures by area
    sum_weights = np.sum(areas_grid[:-1, :])  # sum of weights, exclude south pole cause its quad area is zero
    tg_w_new = tg_w_new / sum_weights
    diff = tg_w_new - tg_w_old
    print(
        f'Test weighted temperature: min abs diff = {np.min(abs(diff))}, avg abs diff = {np.average(abs(diff))}, max abs diff = {np.max(abs(diff))}.')


def test_total_area(lats, lons, temps, total_area_earth_m2):
    # check whather total earth area is same computed different ways
    T_all = np.where(temps != np.nan)  # all
    T_all = np.asarray(T_all)
    idc_whole_earth_1st_epoch = np.where(T_all[0] == 0)  # first epoch
    area_earth = area_quads_sphere(latitudes=lats, longitudes=lons, indices_lat=(T_all[2])[idc_whole_earth_1st_epoch],
                                   indices_lon=(T_all[3])[idc_whole_earth_1st_epoch])
    print('Total earth area various ways:')
    print(f'Polygon +-180, +-90:                          {total_earth_area()/1e6} [km2]')
    print(f'Sum of grid quads from lat, lon coordinates1: {area_earth / 1e6} [km2]')
    print(f'Sum of grid quads from lat, lon coordinates2: {total_area_earth_m2/1e6} [km2]')


def temp_model_lat(lat_array, equator_pole_temp_diff=45, pole_temp=-20):
    '''Based on https://journals.ametsoc.org/view/journals/clim/26/18/jcli-d-12-00636.1.xml'''
    # todo: might need earth tilt with seasons etc - so maybe this rorrection shoudl be done to yearly temp
    # todo: maybe degrees to radians needed
    # todo: maybe model from insolation, + eccentricy itp
    return np.cos(lat_array)*equator_pole_temp_diff + pole_temp


if __name__ == '__main__':
    Tc = -1.5   # sea water freezing point
    w = 48      # how to decompose first axis (e.g. 4752=48*99)
    y = 99
    processed_data_dir = None # './T_area/2/' # './1_supressed_albedo/' # None

    if processed_data_dir is None:

        file2read = netcdf.NetCDFFile('../EBM/output/' + 'timesteps-output2.nc', 'r')
        # file2read = netcdf.NetCDFFile('/home/mk/UiT/code/ebm/FortranCode/EBM/output/' + 'timesteps-output2.nc', 'r')
        temps = (file2read.variables['temperature'])[:] * 1
        lats = (file2read.variables['latitude'])[:] * 1
        lons = (file2read.variables['longitude'])[:] * 1
        file2read.close()

        # todo: only northern hemisphere
        # todo: correct entire surface area
        temps = temps[:,:,:32,:]
        lats = lats[:32]

        # -------- TEMPERATURE -----------------------------------------------------------------------------------------
        # -------- 1. Align to spring equinox
        # alight Temp to start of the year - this will loose some data!
        # Tg_weekly = align_year_vernal_eqx(data=Tg_weekly, w_in_y=w, w_in_y_before_vernal_eqx=11,            # todo!!: on temperature
        #                                   remove_trailing0_y=True)  # year starts in March, at equinox
        temps_aligned = align_year_vernal_eqx(data=temps, w_in_y=w, w_in_y_before_vernal_eqx=11,            # todo!!: on temperature
                                          remove_trailing0_y=True)  # year starts in March, at equinox

        # Tg_weekly_old = np.mean(temps, axis=(2, 3))  # w*y entries
        y = np.shape(temps_aligned)[0] / w #y = len(Tg_weekly) / w  # update nr of years
        if not y.is_integer():
            raise ValueError(
                f"Number of full years ({y:1.5f}) seems to be not an integer number. Something went wrong.")
            # todo: why is Temp for the last 48 weeks (1 year) ==0 ?
            # todo: is it 48*7.6 day = 365 days or 48*7d = 336 and we are missing entire month in a year?
        else:
            y = int(y)


        # -------- ICE AREA --------------------------------------------------------------------------------------------
        # -------- 2. Find frozen gridpoints, compute total frozen area, compute annual mean
        # area below threshold temperature Tc weekly
        T_below_thr_indices = np.where(temps_aligned <= Tc) # todo: To + Tg < Tc
        T_below_thr_indices = np.asarray(T_below_thr_indices)
                                                    # 4 coordinates:
                                                        # temporal: T_below_thr_indices[0]  (max:w*y-1),
                                                        # NA:       T_below_thr_indices[1],
                                                        # latitude: T_below_thr_indices[2]  (max:65-1),
                                                        # longitude:T_below_thr_indices[3]  (max: 128-1)

        frozen_area_weekly_m2 = np.empty(w * y)
        # for each time epoch get frozen area
        for t in tqdm(range(0, w*y), position=0, leave=True):       # with progress bar
        # for t in range(0, w*y):                                   # without progress bar
            # list of indices where Tg < Tc for current epoch
            idc_frozen_in_this_epoch = np.where(T_below_thr_indices[0] == t)
            # frozen area
            frozen_area_weekly_m2[t] = area_quads_sphere(latitudes=lats, longitudes=lons, indices_lat=(T_below_thr_indices[2])[idc_frozen_in_this_epoch], indices_lon=(T_below_thr_indices[3])[idc_frozen_in_this_epoch])    # todo: double-check

        # frozen area yearly
        frozen_area_annual_m2 = np.mean(np.reshape(frozen_area_weekly_m2, (y, w)), axis=1) # todo: double-check
                                                                                    # todo: above equator
                                                                                    # todo: detached icebergs?

        # # old approach, probably wrong
        # global_mean_annual = np.fromiter((np.mean(temps[iy * w:iy * w + w, :, :, :]) for iy in range(0, y)), dtype='float')
        # num_frozen_weekly = np.count_nonzero(temp <= Tc, axis=(2, 3)) # area of ice (possibly detached into separate pieces)
        # num_frozen_annual = np.fromiter((np.mean(np.count_nonzero(temps[iy*w:iy*w+w] <= Tc, axis=(2, 3))) for iy in range(0,y)), dtype='float')

        # -------- TEMPERATURE -----------------------------------------------------------------------------------------
        # -------- 2. Compute global Tg: 2.1. subtract To(theta), 2.2./2.3. compute global weekly/annual average weighted by grid area
        # 2.1.
        # # todo: TEST!
        # # subtract latitude-dependent temperature component, assumina that T(t, theta) = Tg(t) + To(theta). We check whether subtracting estimation of To from simulated temperatures T gives us better final results that MR predicted
        # temps_equator = temps[:, 0, 32, :]  # all epochs, all lats
        # temp_mean_equator = np.mean(temps_equator)  # avg over all lats and epochs
        # temps_north_pole = temps[:, 0, 0, :]  # all epochs, all lats
        # temp_mean_north_pole = np.mean(temps_north_pole)  # avg over all lats and epochs
        # To_lat = temp_model_lat(lat_array=lats, equator_pole_temp_diff=temp_mean_equator - temp_mean_north_pole,
        #                         pole_temp=temp_mean_north_pole)  # todo: try this
        To_lat = temp_model_lat(lat_array=lats)  # todo: try default

        # # correct simulated temperatures - for each latitude subtract To for this latitude
        # todo meeting: keep including To
        temps_no_To = temps_aligned # todo:they are with To
        # #
        # temps_no_To = np.empty(shape=np.shape(temps_aligned))
        # # todo: subtract in axis, optimize below, double-check, remove trailing 0 and equinox align already for temp,
        # #     todo stopped: need to adapt Tc to new Tg without To, cause ice % area suddenly changed
        # #
        # # for epoch in range(0, np.shape(temps)[0]):
        # #     for dummy in range(0, np.shape(temps)[1]):
        # #         for lat in range(0, np.shape(temps)[2]):
        # #             for lon in range(0, np.shape(temps)[3]):
        # #                 temps_no_To[epoch, dummy, lat, lon] = temps[epoch, dummy, lat, lon] - To_lat[lat]
        # for l in range(0, len(lats)):
        #     temps_no_To[:,:,l,:] = temps_aligned[:, :, l, :] - To_lat[l]    # todo stopped: model gives extreme temperature
        #                                                             # todo: for loop to iter
        #                                                             # todo stopped: maybe do everything, compute Tg, and then subtract To ? (no, no sense: checke where frozen areas are, then subtract temp-T0, and computeTg?)
        # # temps = temps_no_To

        # 2.2.
        # weekly (1/48 yr) global temperature Tg (mean over lat and lon)
            # Tg_weekly = np.mean(temps, axis=(2,3))[:len(Tg_weekly)-w]                     # w*y entries, not weighing Temp by area, not realigning year to March equinox
            # y -= 1
        # get grid areas for weighting temperature
        areas_grid = np.empty((len(lats), len(lons)), dtype=float)
        for la in range(0, len(lats)):
            for lo in range(0, len(lons)):
                areas_grid[la, lo] = area_1quad_sphere(lat1=lats[la], lat2=next_lat(latitudes=lats, idx=la),
                                                       lon1=lons[lo], lon2=next_lon(longitudes=lons, idx=lo))
        total_area_earth_m2 = np.sum(areas_grid)  # in m2
        # test_total_area(lats, lons, temps, total_area_earth_m2)

        # compute mean global temperature weekly, weighted by area of each quad. Note: this emphasizes equatorial areas increasing temperature by ~9 degrees wrt equally-weighted mean
        Tg_weekly = np.empty(np.shape(temps_no_To)[0], dtype='float')
        for k in range(0, np.shape(temps_no_To)[0]):
            Tg_weekly[k] = np.sum(temps_no_To[k, 0, :, :] * areas_grid[:, :])  # weigh temperatures by area (south pole has gquad area ==0 !)
        Tg_weekly = Tg_weekly / total_area_earth_m2 # avg by sum of quad areas
        # test_weighted_temp(temps)

        # 2.3.
        # annual global temperature Tg (mean over w weeks)
        Tg_annual = np.mean(np.reshape(Tg_weekly, (y, w)), axis=1)  # y entries
        # Tg_annual_old = np.mean(np.reshape(Tg_weekly_old[:,0], (99, w)), axis=1)  # y entries

                                                # todo: check what happens to triangles at poles

                                                # todo: how to distinguish Tg from Tg+To(theta) ?
                        # get temperature profile with latitude for first epoch
                        # check if it differs for different longitudes, and if so, how much, maybe avg over latitudes - NO fcn of latitude is the point ! and if it is reasonable (sinusoid or something)
                        # subtract it from each consecutive epoch



        np.savetxt('Tg_weekly.txt', Tg_weekly)
        np.savetxt('Tg_annual.txt', Tg_annual)
        np.savetxt('area_ice_weekly.txt', frozen_area_weekly_m2)
        np.savetxt('area_ice_annual.txt', frozen_area_annual_m2)

    else:
        total_area_earth_m2 = total_earth_area()
        Tg_weekly = np.loadtxt(processed_data_dir + 'Tg_weekly.txt')
        Tg_annual = np.loadtxt(processed_data_dir + 'Tg_annual.txt')
        frozen_area_weekly_m2 = np.loadtxt(processed_data_dir + 'area_ice_weekly.txt')
        frozen_area_annual_m2 = np.loadtxt(processed_data_dir + 'area_ice_annual.txt')


    # if len(np.unique(temp)) == np.size(temp):   # all global temperatures are unique
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 7))
    ax1[0].plot(Tg_weekly, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[0].set(title='Temperature global (weekly)')
    ax1[0].set_xlabel(r'$1/48^{th}$ of yr')
    ax1[0].set_ylabel(r'$T_g$')
    # ax1[1].plot(Tg_weekly, frozen_area_weekly_m2 /1e6, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[1].plot(Tg_weekly, frozen_area_weekly_m2 / total_area_earth_m2 * 100, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[1].set(title='Ice area (weekly)')
    ax1[1].set_xlabel(r'$T_g$')
    # ax1[1].set_ylabel(r'Area(Tg) [$km^2$]')
    ax1[1].set_ylabel(r'Area(Tg) [%Earth surface]')
    plt.savefig('weekly.png') if processed_data_dir is None else None

    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 7))
    ax2[0].plot(Tg_annual, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[0].set(title='Temperature global (annual)')
    ax2[0].set_xlabel(r'yr')
    ax2[0].set_ylabel(r'$T_g$')
    # ax2[1].plot(Tg_annual, frozen_area_annual_m2 /1e6, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[1].plot(Tg_annual, frozen_area_annual_m2 / total_area_earth_m2 * 100, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[1].set(title='Ice area (annual)')
    ax2[1].set_xlabel(r'$T_g$')
    # ax2[1].set_ylabel(r'Area(Tg) [$km^2$]')
    ax2[1].set_ylabel(r'Area(Tg) [%Earth surface]')
    plt.savefig('annual.png') if processed_data_dir is None else None

    plt.show(block=True)
