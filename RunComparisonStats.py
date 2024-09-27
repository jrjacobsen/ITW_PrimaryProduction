"""

@author: jjacob2
"""
import os
os.chdir('/home/jjacob2/python/island_wave/subroutines/')
import sys
sys.path.append('/home/jjacob2/python/island_wave/subroutines/')

import numpy as np
from netCDF4 import Dataset as nc4
import matplotlib.pyplot as plt
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subsetting
from scipy import stats

#Directories & Files
###############################################################################
root = '/home/jjacob2/runs/island_wave/'
dataroot = '/home/jjacob2/python/island_wave/data/'

#run file
runName = root+'wind_forced/output/my25_npzd04/diurnal_itw/roms_his_slice.nc'
runNameCL = root+'wind_forced/output/my25_npzd04/itw/roms_his_slice.nc'

#figure directory
fig_dir = '/home/jjacob2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'NP_PP_DiurnLight_23m_40m_240110'

#interp files
#oligotrophic
dep = 'interp23m_'
zlev = -23

file_itw = [dataroot+dep+'npzd04_lightconst_botlinT_rotaWind_itw.nc',
            dataroot+dep+'npzd04_diurnlight_botlinT_rotaWind_itw.nc',
            dataroot+dep+'diurnlight_botlinT_ulandsea_itw.nc',
            dataroot+dep+'diurnlight_botlinT_amp15_rotaWind_itw.nc',
            dataroot+dep+'diurnlight_botlinT_amp05_rotaWind_itw.nc',
            dataroot+dep+'diurnlight_botlinT_amp02_rotaWind_itw.nc',
            dataroot+dep+'diurnlight_botlinT_amp01_rotaWind_itw.nc',
            dataroot+dep+'diurnlight_botlinT_counter_rotaWind_itw.nc']

file_ref = [dataroot+dep+'npzd04_lightconst_botlinT_rotaWind_noisland.nc',
            dataroot+dep+'npzd04_diurnlight_botlinT_rotaWind_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_ulandsea_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_amp15_rotaWind_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_amp05_rotaWind_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_amp02_rotaWind_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_amp01_rotaWind_noisland.nc',
            dataroot+dep+'diurnlight_botlinT_counter_rotaWind_noisland.nc']

#grid file
gridfile = root+'grid_amp_lrg/grids/island_subgrid.nc'

#load files 
#Grid
grid = nc4(gridfile, 'r')

#run file
rnfile = nc4(runName, 'r')
rnfileCL = nc4(runNameCL, 'r')

#Slicing & Grid 
###############################################################################
idx = {'lat' : slice(None, None),
       'lon' : slice(None,None),
       'time': slice(192,-12)}

#time
time = rnfile.variables['ocean_time'][idx['time']]

#land mask
mask = np.array(grid.variables['mask_rho'][idx['lat'],idx['lon']],dtype = bool)


#grid
xrho = np.ma.array(rnfile.variables['x_rho'][idx['lat'],idx['lon']]/1000 - 120,
                               mask = ~mask)
yrho = np.ma.array(rnfile.variables['y_rho'][idx['lat'],idx['lon']]/1000 - 100,
                               mask = ~mask)

#distance from island center
dist = np.ma.sqrt(xrho**2 + yrho**2)
def sub10(dist, var) :
    x = var[dist <= 10]
    return x[x.mask == False]

idist = sub10(dist,dist)

#Load files
###############################################################################
#ITW
ncfile = {}
ncfile['Const'] = nc4(file_itw[0], 'r')
ncfile['Rota'] = nc4(file_itw[1], 'r')
ncfile['uLan'] = nc4(file_itw[2], 'r')
ncfile['amp15'] = nc4(file_itw[3], 'r')
ncfile['amp05'] = nc4(file_itw[4], 'r')
ncfile['amp02'] = nc4(file_itw[5], 'r')
ncfile['amp01'] = nc4(file_itw[6], 'r')
ncfile['counter'] = nc4(file_itw[7], 'r')

t_itw = {}
t_itw['Const'] = ncfile['Const'].variables['temp'][idx['time'],
                                                idx['lat'],idx['lon']]
t_itw['Rota'] = ncfile['Rota'].variables['temp'][idx['time'],
                                                idx['lat'],idx['lon']]
t_itw['uLan'] = ncfile['uLan'].variables['temp'][idx['time'],
                                                idx['lat'],idx['lon']]
t_itw['amp15'] = ncfile['amp15'].variables['temp'][idx['time'],
                                                  idx['lat'],idx['lon']]
t_itw['amp05'] = ncfile['amp05'].variables['temp'][idx['time'],
                                                  idx['lat'],idx['lon']]
t_itw['amp02'] = ncfile['amp02'].variables['temp'][idx['time'],
                                                  idx['lat'],idx['lon']]
t_itw['amp01'] = ncfile['amp01'].variables['temp'][idx['time'],
                                                  idx['lat'],idx['lon']]
t_itw['counter'] = ncfile['counter'].variables['temp'][idx['time'],
                                                      idx['lat'],idx['lon']]

#subset to distance
const = np.empty((time.shape[0], idist.shape[0]))
diurn = np.empty(const.shape)
uLand = np.empty(const.shape)
amp05 = np.empty(const.shape)
amp02 = np.empty(const.shape)
amp01 = np.empty(const.shape)
amp15 = np.empty(const.shape)
countr= np.empty(const.shape)
for i in range(time.shape[0]) :
    const[i,:] = sub10(dist, t_itw['Const'][i,:,:])
    diurn[i,:] = sub10(dist, t_itw['Rota'][i,:,:])
    uLand[i,:] = sub10(dist, t_itw['uLan'][i,:,:])
    amp15[i,:] = sub10(dist, t_itw['amp15'][i,:,:])
    amp05[i,:] = sub10(dist, t_itw['amp05'][i,:,:])
    amp02[i,:] = sub10(dist, t_itw['amp02'][i,:,:])
    amp01[i,:] = sub10(dist, t_itw['amp01'][i,:,:])
    countr[i,:] = sub10(dist, t_itw['counter'][i,:,:])

#place in dictionary
temp = {}
temp['Amp1.5'] = amp15.flatten()
temp['Amp1.0'] = diurn.flatten()
temp['Amp0.5'] = amp05.flatten()
temp['Amp0.2'] = amp02.flatten()
temp['Amp0.1'] = amp01.flatten()

#Plotting
###############################################################################
#histograms
fig, ax = plt.subplots()
ax.boxplot(temp.values())
ax.set_xticklabels(temp.keys())
ax.set_ylabel('Temperature [$^{\circ}$C]')
ax.set_title(str(-zlev)+' m depth', loc = 'left')
