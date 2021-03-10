from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from area import area
from tqdm import tqdm # not required, just for progress bar plotting


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


def compute_sphere_grid_areas(latitude_coord, longitude_coord):
    # todo: could optimize furhter by computing one longitude strip and using for all
    num_lats = len(latitude_coord)
    num_lons = len(longitude_coord)
    areas = np.empty((num_lats, num_lons))
    for lat in range(0,num_lats):
        for lon in range(0, num_lons):
            areas[lat, lon] = area_1quad_sphere(*get_lower_right_quad_vertices(latitude_coord, longitude_coord, lat, lon))
    return areas


def total_earth_area():
    obj = {'type': 'Polygon',
           'coordinates': [[[-180, -90], [-180, 90], [180, 90], [180, -90], [-180, -90]]]}  # full earth area
    return area(obj)  # in km2


def align_year_vernal_eqx(data, epochs_in_year=48, e_in_y_before_vernal_eqx=11, remove_trailing0_y=True):
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
    aligned_data = data[epochs_in_year - e_in_y_before_vernal_eqx:-e_in_y_before_vernal_eqx, :, :, :]
    if remove_trailing0_y:
        while np.all(aligned_data[-1,:,:,:] == 0.0): # aligned_data[-1] == 0.0:
            aligned_data = aligned_data[:np.shape(aligned_data)[0] - epochs_in_year, :, :, :]
        # aligned = aligned[:len(aligned)-epochs_in_period]
    return aligned_data


def compute_weighted_total(values, weights):
    if values == None:
        values = np.ones(np.shape(weights))
    elif weights == None:
        weights = np.ones(np.shape(values))
    return np.sum(values*weights)


def avg_over_num_epochs(input_vector, num_periods, epochs_in_period, reshape=None):
        '''returns vector of averages in each period'''
        input_vector_2D = input_vector.reshape((num_periods, epochs_in_period))
        input_vector_period_avg = np.mean(input_vector_2D, axis=1)
        if reshape is not None:
            return input_vector_period_avg.reshape(reshape)
        else:
            return input_vector_period_avg


def test_weighted_temp():
    '''testing weighted temp  - it will not be exactly the same, cause in equally-weighted mean south pole counted |lon vals| times, and in weighted one zero times'''
    areas_grid = np.ones((len(lats), len(lons)),dtype=float)  # equal weithts to compare old equally weighted, and now weighted
    tg_w_old = np.mean(temps[:, 0, :-1, :], axis=(1, 2))  # all except south pole, equally weighted
    tg_w_new = np.empty(np.shape(temps)[0], dtype='float')
    for k in range(0, np.shape(temps)[0]):
        tg_w_new[k] = np.sum(temps[k, 0, :, :] * areas_grid[:, :])  # weigh temperatures by area
    sum_weights = np.sum(areas_grid[:-1, :])  # sum of weights, exclude south pole cause its quad area is zero
    tg_w_new = tg_w_new / sum_weights
    diff = tg_w_new - tg_w_old
    print(f'Test weighted temperature: min abs diff = {np.min(abs(diff))}, avg abs diff = {np.average(abs(diff))}, max abs diff = {np.max(abs(diff))}.')


def test_total_area(lats, lons, temps, total_area_earth_m2):
    # check whather total earth area is same computed different ways
    T_all = np.where(temps != np.nan)  # all
    T_all = np.asarray(T_all)
    idc_whole_earth_1st_epoch = np.where(T_all[0] == 0)  # first epoch
    area_earth = area_quads_sphere(latitudes=lats, longitudes=lons, indices_lat=(T_all[2])[idc_whole_earth_1st_epoch],
                                   indices_lon=(T_all[3])[idc_whole_earth_1st_epoch])
    area_grid = np.sum(compute_sphere_grid_areas(latitude_coord=lats, longitude_coord=lons))
    print('Total earth area various ways:')
    print(f'Polygon +-180, +-90:                          {total_earth_area()/1e6} [km2]')
    print(f'Sum of grid quads from lat, lon coordinates1: {area_earth / 1e6} [km2]')
    print(f'Sum of grid quads from lat, lon coordinates2: {total_area_earth_m2/1e6} [km2]')
    print(f'Sum of grid quads from lat, lon coordinates3: {area_grid/1e6} [km2]')


# def temp_model_lat(lat_array, equator_pole_temp_diff=45, pole_temp=-20):
#     '''Based on https://journals.ametsoc.org/view/journals/clim/26/18/jcli-d-12-00636.1.xml'''
#     return np.cos(lat_array)*equator_pole_temp_diff + pole_temp


if __name__ == '__main__':
    Tc = -1.5   # sea water freezing point
    epochs_in_year = 48      # how to decompose first axis (e.g. 4752=48*99)
    epochs_before_equinox = 11
    num_years = 500
    processed_data_dir = None # './T_area/2/' # './1_supressed_albedo/' # None
    raw_data_file_path = '../EBM/output/500yrs' + 'timesteps-output2.nc'

    if processed_data_dir is None and raw_data_file_path is None:
        raise ValueError('Need input.')
    elif processed_data_dir is None:
        # read unprocessed temperatures
        file2read = netcdf.NetCDFFile(raw_data_file_path, 'r')
        temps = (file2read.variables['temperature'])[:] * 1
        lats = (file2read.variables['latitude'])[:] * 1
        lons = (file2read.variables['longitude'])[:] * 1
        file2read.close()

        if np.shape(temps)[0]/num_years != epochs_in_year:
            raise ValueError('Specified number of epochs in year and number of years is not consistent with the total number of epochs in the temperature record.')

        # get grid areas for weighting temperature and computing ice area
        grid_areas = compute_sphere_grid_areas(latitude_coord=lats, longitude_coord=lons)
        total_area_earth_m2 = np.sum(grid_areas)  # in m2
        # test_total_area(lats, lons, temps, total_area_earth_m2)
        
        # remove southern hemisphere to exclude impact from Antarctica
        half_len = int(np.ceil(len(lats) / 2))
        lats = lats[:half_len]
        temps = temps[:, :, :half_len, :]
        grid_areas = grid_areas[:half_len,:]


        # -------- 1. Align temperatures to spring equinox - this will loose some data! Update #years num_periods
        temps = align_year_vernal_eqx(data=temps, epochs_in_year=epochs_in_year, e_in_y_before_vernal_eqx=epochs_before_equinox,
                                      remove_trailing0_y=True)  # year starts in March, at equinox
        num_years = np.shape(temps)[0] / epochs_in_year
        if not num_years.is_integer():
            raise ValueError(f"Number of full years ({num_years:1.5f}) seems to be not an integer number. Something went wrong.")
        else:
            num_years = int(num_years)
                                                                                                                        # todo: why is Temp for the last 48 weeks (1 year) ==0 ?
                                                                                                                        # todo: is it 48*7.6 day = 365 days or 48*7d = 336 and we are missing entire month in a year?

        # -------- 2. Find frozen gridpoints in each weekly epoch
        # temps_below_thr_indices = np.asarray(np.where(temps <= Tc))
        #                                             # 4 coordinates:
        #                                                 # temporal: temps_below_thr_indices[0]  (max:epochs_in_period*num_periods-1),
        #                                                 # NA:       temps_below_thr_indices[1],
        #                                                 # latitude: temps_below_thr_indices[2]  (max:65-1),
        #                                                 # longitude:temps_below_thr_indices[3]  (max: 128-1)
        # todo: detached icebergs?

        # -------- 3. Compute global temperature and total frozen area: weekly (for each separate time epoch)
        # todo !!: not north point to decide temperature cause Norhte pole has too much impact then
        # todo: see fortran code how they do it app.f90 line 15 and on
        total_frozen_area_per_epoch = np.empty(epochs_in_year * num_years)
        global_temperature_per_epoch = np.empty(epochs_in_year * num_years)
        for t in tqdm(range(0, epochs_in_year * num_years), position=0, leave=True):       # without progress bar: for t in range(0, epochs_in_period*num_periods):
            epoch_temps = temps[t, 0, :, :]
            indices_below_threshold = np.where(epoch_temps <= Tc)
            total_frozen_area_per_epoch[t] = compute_weighted_total(values=None, weights=grid_areas[indices_below_threshold])
            global_temperature_per_epoch[t] = compute_weighted_total(values=epoch_temps[indices_below_threshold], weights=grid_areas[indices_below_threshold])
        global_temperature_per_epoch /= total_area_earth_m2

        total_frozen_area_per_epoch__2D_YbyE = total_frozen_area_per_epoch.reshape(num_years, epochs_in_year)
        global_temperature_per_epoch__2D_YbyE = global_temperature_per_epoch.reshape(num_years, epochs_in_year)
        # test_weighted_temp()

        # -------- 4. Compute yearly, seasonal and monthly averages
        # yearly average
        total_frozen_area_yearly_avg__1D_Y =avg_over_num_epochs(input_vector=total_frozen_area_per_epoch__2D_YbyE, num_periods=num_years, epochs_in_period=epochs_in_year)         # colapses 2nd dimension into 1, can reshape if needed
        global_temperature_yearly_avg__1D_Y =avg_over_num_epochs(input_vector=global_temperature_per_epoch__2D_YbyE, num_periods=num_years, epochs_in_period=epochs_in_year)

        # seasonal average
        seasons_in_year = 4
        if epochs_in_year % seasons_in_year == 0:
            total_frozen_area_seasonal_avg__2D_YbyS = avg_over_num_epochs(input_vector=total_frozen_area_per_epoch__2D_YbyE, num_periods=num_years * seasons_in_year, epochs_in_period=epochs_in_year / seasons_in_year, reshape=(num_years, seasons_in_year))
            global_temperature_seasonal_avg__2D_YbyS = avg_over_num_epochs(input_vector=global_temperature_per_epoch__2D_YbyE, num_periods=num_years * seasons_in_year, epochs_in_period=epochs_in_year / seasons_in_year, reshape=(num_years, seasons_in_year))
        else:
            print(f'Number of epochs in year ({str(epochs_in_year)}) not divisible by 4 seasons. Seasonal average not computed.')

        # monthly average
        months_in_year = 12
        if epochs_in_year % months_in_year == 0:
            total_frozen_area_monthly_avg__2D_YbyM = avg_over_num_epochs(input_vector=total_frozen_area_per_epoch__2D_YbyE, num_periods=num_years * months_in_year, epochs_in_period=epochs_in_year / months_in_year, reshape=(num_years, months_in_year))
            global_temperature_monthly_avg__2D_YbyM = avg_over_num_epochs(input_vector=global_temperature_per_epoch__2D_YbyE, num_periods=num_years * months_in_year, epochs_in_period=epochs_in_year / months_in_year, reshape=(num_years, months_in_year))
        else:
            print(f'Number of epochs in year ({str(epochs_in_year)}) not divisible by 12 months. Monthly average not computed.')

                        # todo?
                        # get temperature profile with latitude for first epoch
                        # check if it differs for different longitudes, and if so, how much, maybe avg over latitudes - NO fcn of latitude is the point ! and if it is reasonable (sinusoid or something)
                        # subtract it from each consecutive epoch


        np.savetxt('temp_global_per_epoch.txt', global_temperature_per_epoch__2D_YbyE)
        np.savetxt('temp_global_monthly_avg.txt', global_temperature_monthly_avg__2D_YbyM) if epochs_in_year % months_in_year == 0 else None
        np.savetxt('temp_global_seasonal_avg.txt', global_temperature_seasonal_avg__2D_YbyS) if epochs_in_year % seasons_in_year == 0 else None
        np.savetxt('temp_global_yearly_avg.txt', global_temperature_yearly_avg__1D_Y)

        np.savetxt('total_frozen_area_per_epoch__2D_YbyE.txt', total_frozen_area_per_epoch__2D_YbyE)
        np.savetxt('total_frozen_area_monthly_avg__2D_YbyM.txt', total_frozen_area_monthly_avg__2D_YbyM) if epochs_in_year % months_in_year == 0 else None
        np.savetxt('total_frozen_area_seasonal_avg__2D_YbyS.txt', total_frozen_area_seasonal_avg__2D_YbyS) if epochs_in_year % seasons_in_year == 0 else None
        np.savetxt('total_frozen_area_yearly_avg__1D_Y.txt', total_frozen_area_yearly_avg__1D_Y)

    else:
        total_area_earth_m2 = total_earth_area()
        global_temperature_per_epoch__2D_YbyE = np.loadtxt(processed_data_dir + 'temp_global_per_epoch.txt')
        global_temperature_yearly_avg__1D_Y = np.loadtxt(processed_data_dir + 'temp_global_yearly_avg.txt')
        total_frozen_area_per_epoch__2D_YbyE = np.loadtxt(processed_data_dir + 'total_frozen_area_per_epoch__2D_YbyE.txt')
        total_frozen_area_yearly_avg__1D_Y = np.loadtxt(processed_data_dir + 'total_frozen_area_yearly_avg__1D_Y.txt')

        try:
            global_temperature_monthly_avg__2D_YbyM = np.loadtxt(processed_data_dir + 'temp_global_monthly_avg.txt')
            total_frozen_area_monthly_avg__2D_YbyM = np.loadtxt(processed_data_dir + 'total_frozen_area_monthly_avg__2D_YbyM.txt')
        except:
            global_temperature_monthly_avg__2D_YbyM = None
            total_frozen_area_monthly_avg__2D_YbyM = None
            print('Monthly averages not available.')

        try:
            global_temperature_seasonal_avg__2D_YbyS = np.loadtxt(processed_data_dir + 'temp_global_seasonal_avg.txt')
            total_frozen_area_seasonal_avg__2D_YbyS = np.loadtxt(processed_data_dir + 'total_frozen_area_seasonal_avg__2D_YbyS.txt')
        except:
            global_temperature_seasonal_avg__2D_YbyS = None
            total_frozen_area_seasonal_avg__2D_YbyS = None
            print('Seasonal averages not available.')


    def plot_subplot(ax, x_vec, y_vec, title, xlbl, ylbl, linestyle='solid'):
        ax.plot(x_vec, y_vec, '.', markersize=4, linestyle=linestyle, linewidth=0.7)
        ax.set(title=title)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)

    def plot_area_vs_temp_subpl(temp_2D__YbyTotalSubplots, area_2D__YbyTotalSubplots, subplot_rows, subplot_cols, figsize, titles, xlabels, ylabels, linestyle, fig_savepath):
        '''Plot in subplots.
        in 2D arrays:
            #cols == total number of subplots == subplot_rows*subplot_cols,
                  == len of titles, xlabels, ylabels
            #rows == #points to plot in each plot.
        '''
        fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize)

        for r in range(0, subplot_rows):
            for c in range(0, subplot_cols):
                record = r*subplot_cols+c
                plot_subplot(ax=ax[r,c], x_vec=temp_2D__YbyTotalSubplots[:,record], y_vec=area_2D__YbyTotalSubplots[:,record], title=titles[record], xlbl=xlabels[record], ylbl=ylabels[record], linestyle=linestyle)

        plt.savefig(fig_savepath) if fig_savepath is None else None

    def plot_temp_and_area_vs_temp(temp_1D, area_1D, figsize, titles_2, xlabels_2, ylabels_2, linestyles_2, fig_savepath):
        fig, ax = plt.subplots(1, 2, figsize)

        ax1[0].plot(temp_1D, '.', markersize=4, linestyle=linestyles_2[0], linewidth=0.7)
        ax1[0].set(title=titles_2[0])
        ax1[0].set_xlabel(xlabels_2[0])
        ax1[0].set_ylabel(ylabels_2[0])
        ax1[1].plot(temp_1D, area_1D, '.', markersize=4, linestyle=linestyles_2[1], linewidth=0.7)
        ax1[1].set(title=titles_2[1])
        ax1[1].set_xlabel(xlabels_2[1])
        ax1[1].set_ylabel(ylabels_2[1])
        plt.savefig(fig_savepath) if fig_savepath is None else None


    # todo: scatter for other than yearly

    # plot_area_vs_temp(temp_2D=, area_2D=, subplot_rows=, subplot_cols=, figsize=, titles=, xlabel=, ylabel=, fig_savepath=)
    titles = np.array([])
    plot_area_vs_temp_subpl(x_2D=global_temperature_seasonal_avg__2D_YbyS, y_2D=total_frozen_area_seasonal_avg__2D_YbyS,
                      subplot_rows=1, subplot_cols=2, figsize=(18,6), titles=, xlabel=, ylabel=, fig_savepath=)



    # by epoch
    plot_temp_and_area_vs_temp(temp_1D=np.matrix.flatten(global_temperature_per_epoch__2D_YbyE), area_1D=np.matrix.flatten(total_frozen_area_per_epoch__2D_YbyE),
                               figsize=(18,7), titles_2=np.array(['Global temperature (by epoch)', 'Ice area (by epoch)']),
                               xlabels_2, ylabels_2, linestyles_2=np.array(['solid',' ']), fig_savepath)
    # avg by monthly window - not plotted
    # avg by seasonal window - not plotted
    # avg by yearly window
    plot_temp_and_area_vs_temp(temp_1D=np.matrix.flatten(global_temperature_yearly_avg__1D_Y), area_1D=np.matrix.flatten(total_frozen_area_yearly_avg__1D_Y),
                               figsize=(18,7), titles_2np.array(['Global temperature (yearly avg window)', 'Ice area (yearly avg window)']),
                               xlabels_2, ylabels_2, linestyles_2=np.array(['solid','solid']), fig_savepath)

    # todo: plot what available, sure where 1st epoch starts (january or spring?)
    # if len(np.unique(temp)) == np.size(temp):   # all global temperatures are unique
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 7))
    ax1[0].plot(Tg_weekly, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[0].set(title='Temperature global (weekly)')
    ax1[0].set_xlabel(r'$1/48^{th}$ of yr')
    ax1[0].set_ylabel(r'$T_g$')
    # ax1[1].plot(Tg_weekly, frozen_area_weekly_m2 /1e6, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[1].plot(Tg_weekly, frozen_area_weekly_m2 /(1e6*1e6), '.', markersize=4, linestyle='solid', linewidth=0.7)
    # ax1[1].plot(Tg_weekly, frozen_area_weekly_m2 / total_area_earth_m2 * 100, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax1[1].set(title='Ice area (weekly)')
    ax1[1].set_xlabel(r'$T_g$')
    # ax1[1].set_ylabel(r'Area(Tg) [$km^2$]')
    ax1[1].set_ylabel(r'Area(Tg) [$10^6\;km^2$]')
    # ax1[1].set_ylabel(r'Area(Tg) [%Earth surface]')
    plt.savefig('weekly.png') if processed_data_dir is None else None

    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 7))
    ax2[0].plot(Tg_annual, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[0].set(title='Temperature global (annual)')
    ax2[0].set_xlabel(r'yr')
    ax2[0].set_ylabel(r'$T_g$')
    # ax2[1].plot(Tg_annual, frozen_area_annual_m2 /1e6, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[1].plot(Tg_annual, frozen_area_annual_m2 /(1e6*1e6), '.', markersize=4, linestyle='solid', linewidth=0.7)
    # ax2[1].plot(Tg_annual, frozen_area_annual_m2 / total_area_earth_m2 * 100, '.', markersize=4, linestyle='solid', linewidth=0.7)
    ax2[1].set(title='Ice area (annual)')
    ax2[1].set_xlabel(r'$T_g$')
    # ax2[1].set_ylabel(r'Area(Tg) [$km^2$]')
    ax2[1].set_ylabel(r'Area(Tg) [$10^6\;km^2$]')
    # ax2[1].set_ylabel(r'Area(Tg) [%Earth surface]')
    plt.savefig('annual.png') if processed_data_dir is None else None

    plt.show(block=True)
