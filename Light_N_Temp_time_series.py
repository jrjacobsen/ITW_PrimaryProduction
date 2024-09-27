#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:41:02 2023

@author: jjacob2
"""

import os
os.chdir('/home/jjacobs2/python/island_wave/subroutines/')
import numpy as np
from netCDF4 import Dataset as nc4
import obs_depth_vs145 as dep
import matplotlib.pyplot as plt
import cmocean
from scipy.signal import welch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from subsetting import island_ts as locator

#Directories & Files
###############################################################################
#Root Directory
root = '/home/jjacobs2/runs/island_wave/wind_forced/output/wind_forced/'
gridfile = root+'grids/island_subgrid.nc'

#load file
ncfile = nc4(root+'output/my25_npzd04/diurnal_itw/roms_his_slice.nc')
gdfile = nc4(gridfile)

#figure directory
fig_dir = '/home/jjacobs2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'Physical_Characteristics_240828'



#variables and dimensions
###############################################################################
#spatial slicing
'''
'sw' or  1  => southwest corner
'se' or  2  => southeast corner
'ne' or  3  => northeast corner
'nw' or  4  => northwest corner
'''
idx, title = locator(1) 
#western coast
idx['lat'] = 41
idx['lon'] = 30
idx['time'] = slice(0,-10)

#dimensions
#depth
romsvars = {'Vstretching' : ncfile.variables['Vstretching'][0], \
            'Vtransform' : ncfile.variables['Vtransform'][0], \
            'theta_s' : ncfile.variables['theta_s'][0], \
            'theta_b' : ncfile.variables['theta_b'][0], \
            'N' : ncfile.variables['temp'][0,:,0,0].shape[0], \
            'h' : gdfile.variables['h'][:], \
            'hc': ncfile.variables['hc'][0]}
                
depth = dep._set_depth(ncfile, romsvars, 
                       'rho', romsvars['h'])[:,idx['lat'],idx['lon']] #meters
dz = np.diff(depth,
             axis = 0)
time = ncfile.variables['ocean_time'][idx['time']]/(24*3600) #days

#variables
temp = ncfile.variables['temp'][idx['time'],:,idx['lat'],idx['lon']]
swrad = ncfile.variables['swrad'][idx['time'],idx['lat'],idx['lon']]
rho = ncfile.variables['rho'][idx['time'],:,idx['lat'],idx['lon']]

#parameters
rho0 = ncfile.variables['R0'][0]
g = 9.80665

#Calcluated Values
###############################################################################
#light fraction
lightbar = np.mean(swrad[:48], axis = 0)
light = swrad/lightbar

#gradient of density
drho_dz = np.diff(rho, axis = 1)/dz

#buoyancy frequency
Nsq = -g/rho0*drho_dz


#plotting
###############################################################################

fig, (ax0, ax) = plt.subplots(2,1, figsize =(8,7),sharex = True,
                       gridspec_kw={'height_ratios': [1,3]})
ax0.plot(time, light, c = 'b')
ax0.plot([time[0], time[-1]], [1,1],
         c = 'b', linestyle = 'dashdot')
ax0.set_xlim([4,7])
ax0.set_ylabel('Surface Irradiance \n I(t)/I$_0$', color = 'b')
ax0.grid(alpha = 0.5)
ax0.tick_params(top=True, labeltop=False, bottom=False, labelbottom=False)
ax0.tick_params(axis='y', colors='b')
ax0b = ax0.twinx()
ax0b.plot(time, np.max(Nsq, axis = 1), 
          c = 'k', linewidth = 2, linestyle = '--')
ax0b.set_yticks([0.01, 0.007, 0.004,0.001])
ax0b.set_ylabel('$N^2$ [s$^{-2}$]')

cf=ax.contourf(time,depth, temp.T, 
               cmap = cmocean.cm.thermal,
                vmin = 13.8, vmax = 28,
                levels = np.arange(14,30,2),
                extend = 'both')
ax.contour(time,depth, temp.T,
           colors = 'white',
           levels = [14.2, 21],
           linestyles = ['dashed', 'solid'])
ax.grid(alpha = 0.5)
ax.set_ylim([-70,0])
ax.set_ylabel('Depth [m]')
ax.set_xlabel('Time [Days]')


#save figure
#sav_str = fig_dir+fig_name+'.png'
#plt.savefig(sav_str, bbox_inches='tight', dpi = 300)


