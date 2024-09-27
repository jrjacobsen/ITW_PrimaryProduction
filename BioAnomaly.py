#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:29:49 2023

@author: jjacob2
"""
import os
os.chdir('/home/jjacobs2/python/island_wave/subroutines/')
import sys
sys.path.append('/home/jjacobs2/python/island_wave/subroutines/')

import numpy as np
from netCDF4 import Dataset as nc4
import matplotlib.pyplot as plt
import cmocean 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subsetting
from scipy import stats

#Directories & Files
###############################################################################
root = '/home/jjacobs2/runs/island_wave/wind_forced/output/wind_forced/'


#interp file
#oligotrophic
file = 'npzd04_lightconst_botlinT_rotaWind'

##run file
rnfile = nc4(root+'output/my25_npzd04/itw/roms_his_slice.nc') 
#rnfile = nc4(root+'output/my25_npzd04/diurnal_itw/roms_his_slice.nc', 'r')

dataroot = '/home/jjacobs2/python/island_wave/data/' 
fname23 = 'interp23m_'+file+'_itw.nc'
refname23 = 'interp23m_'+file+'_noisland.nc'

fname40 = 'interp40m_'+file+'_itw.nc'
refname40 = 'interp40m_'+file+'_noisland.nc'

#grid file
gridfile = root+'grids/island_subgrid.nc'

#figure directory
fig_dir = '/home/jjacob2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'NP_PP_DiurnLight_23m_40m_240110'


#load files 
grid = nc4(gridfile, 'r')
refile = nc4(root+'output/my25_npzd04/noisland/roms_his_slice.nc') 
runID = ' at 23 m Depth'


#plot label
lab = {}
lab['n'] = 'Percent Difference [%]'
lab['p'] = 'Percent Difference [%]'
lab['z'] = 'Percent Difference [%]'
lab['title'] = 'Percent Differences from \n Ref. run'+runID


#Slicing & Grid 
###############################################################################
idx = {'lat' : slice(None, None),
       'lon' : slice(None,None),
       'time': slice(192,-12)}

#time
time = rnfile.variables['ocean_time'][idx['time']]/(3600*24)

#land mask
mask = np.array(grid.variables['mask_rho'][idx['lat'],idx['lon']],dtype = bool)


#grid
xrho = np.ma.array(rnfile.variables['x_rho'][idx['lat'],idx['lon']]/1000 - 120, 
                               mask = ~mask)
yrho = np.ma.array(rnfile.variables['y_rho'][idx['lat'],idx['lon']]/1000 - 100, 
                               mask = ~mask)

#bio params
bio = subsetting.bio_params(rnfile)

#Load variables
###############################################################################
s_run = rnfile.variables['dye_01'][idx['time'], 76,51,40]
n_run = rnfile.variables['NO3'][idx['time'], 76, 51, 40]

#bio variables
s_itw = rnfile.variables['dye_01'][idx['time'],76,idx['lat'],idx['lon']]
s_ref = refile.variables['dye_01'][idx['time'],76,idx['lat'],idx['lon']]

n_itw = rnfile.variables['NO3'][idx['time'],76,idx['lat'],idx['lon']]
n_ref = refile.variables['NO3'][idx['time'],76,idx['lat'],idx['lon']]

p_itw = rnfile.variables['phytoplankton'][idx['time'],76,idx['lat'],idx['lon']]
p_ref = refile.variables['phytoplankton'][idx['time'],76,idx['lat'],idx['lon']]

z_itw = rnfile.variables['zooplankton'][idx['time'],76,idx['lat'],idx['lon']]
z_ref = refile.variables['zooplankton'][idx['time'],76,idx['lat'],idx['lon']]

d_itw = rnfile.variables['detritus'][idx['time'],76,idx['lat'],idx['lon']]
d_ref = refile.variables['detritus'][idx['time'],76,idx['lat'],idx['lon']]


#Calculated values
###############################################################################
#average percent difference
s_pdiff = np.mean(100*(s_itw - s_ref)/s_ref, axis = 0)
n_pdiff = np.mean(100*(n_itw - n_ref)/n_ref, axis = 0)
p_pdiff = np.mean(100*(p_itw - p_ref)/p_ref, axis = 0)
z_pdiff = np.mean(100*(z_itw - z_ref)/z_ref, axis = 0)
d_pdiff = np.mean(100*(d_itw - d_ref)/d_ref, axis = 0)


#Plot Parameters
###############################################################################
#colorbar settings
#23 m
pparam= {}
pparam['s'] = (-20, 20, 5)
pparam['n'] = (-20, 20, 5)
pparam['p'] = (-1.5, 1.5, 0.25)
pparam['z'] = (-5.5, 5.5, 0.5)
pparam['d'] = (-3.5, 3.5, 0.5)

cmap = cmocean.cm.balance

#Plotting
###############################################################################
fig, ((ax0, ax0b), (ax1, ax1b)) = plt.subplots(2,2, figsize = (10,9))
#     sharex = True, sharey = True,
#tracer
cf0 =ax0.contourf(xrho, yrho, s_pdiff, 
            vmin = pparam['s'][0], vmax = pparam['s'][1],
            levels = np.arange(pparam['s'][0],pparam['s'][1]+pparam['s'][2],
                               pparam['s'][2]), 
            cmap=cmap)
ax0.grid(alpha = 0.5)
ax0.patch.set_facecolor('silver')
ax0.set_ylabel('Y [km]')
ax0.set_aspect('equal', adjustable='box')
ax0.set_title(lab['title'])
ax0.set_xticklabels([])
divider = make_axes_locatable(ax0)
cax = divider.append_axes('bottom', size='5%', pad=-0.1)
fig.colorbar(cf0, cax=cax, orientation='horizontal',
             label = '%')

ax0b.plot(time, s_itw[:,51,40], label = 'itw')
ax0b.plot(time, s_ref[:,51,40], label = 'ref')
#ax0b.plot(time, s_run, label = 'run 23.5m')
ax0b.set_title('Tracer')
ax0b.legend()


#Nutrient
cf1 =ax1.contourf(xrho, yrho, n_pdiff, 
            vmin = pparam['n'][0], vmax = pparam['n'][1],
            levels = np.arange(pparam['n'][0],pparam['n'][1]+pparam['n'][2],
                               pparam['n'][2]), 
            cmap=cmap)
ax1.grid(alpha = 0.5)
ax1.patch.set_facecolor('silver')
ax1.set_ylabel('Y [km]')
ax1.set_aspect('equal', adjustable='box')
ax1.set_xticklabels([])
divider = make_axes_locatable(ax1)
cax = divider.append_axes('bottom', size='5%', pad=-0.1)
fig.colorbar(cf1, cax=cax, orientation='horizontal',
             label = '%')

ax1b.plot(time, n_itw[:,51,40], label = 'itw')
ax1b.plot(time, n_ref[:,51,40], label = 'ref')
#ax1b.plot(time, n_run, label = 'run 23.5m')
ax1b.set_title('Nutrient')
ax1b.legend()



fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5,1, 
     sharex = True, sharey = True,figsize = (10,9))
#tracer
cf0 =ax0.contourf(xrho, yrho, s_pdiff, 
            vmin = pparam['s'][0], vmax = pparam['s'][1],
            levels = np.arange(pparam['s'][0],pparam['s'][1]+pparam['s'][2],
                               pparam['s'][2]), 
            cmap=cmap)
ax0.grid(alpha = 0.5)
ax0.patch.set_facecolor('silver')
ax0.set_ylabel('Y [km]')
ax0.set_aspect('equal', adjustable='box')
ax0.set_title(lab['title'])
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf0, cax=cax, orientation='vertical',
             label = 'Average Percent \n Anomaly s')

#Nutrient
cf1 =ax1.contourf(xrho, yrho, n_pdiff, 
            vmin = pparam['n'][0], vmax = pparam['n'][1],
            levels = np.arange(pparam['n'][0],pparam['n'][1]+pparam['n'][2],
                               pparam['n'][2]), 
            cmap=cmap)
ax1.grid(alpha = 0.5)
ax1.patch.set_facecolor('silver')
ax1.set_ylabel('Y [km]')
ax1.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf1, cax=cax, orientation='vertical',
             label = 'Average Percent \n Anomaly N')

#Phytoplankton
cf2 =ax2.contourf(xrho, yrho, p_pdiff, 
            vmin = pparam['p'][0], vmax = pparam['p'][1],
            levels = np.arange(pparam['p'][0],pparam['p'][1]+pparam['p'][2],
                               pparam['p'][2]), 
            cmap=cmap)
ax2.grid(alpha = 0.5)
ax2.patch.set_facecolor('silver')
ax2.set_ylabel('Y [km]')
ax2.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf2, cax=cax, orientation='vertical',
             label = 'Average Percent \n Anomaly P')

#Zooplankton
cf3 =ax3.contourf(xrho, yrho, z_pdiff, 
            vmin = pparam['z'][0], vmax = pparam['z'][1],
            levels = np.arange(pparam['z'][0],pparam['z'][1]+pparam['z'][2],
                               pparam['z'][2]), 
            cmap=cmap)
ax3.grid(alpha = 0.5)
ax3.patch.set_facecolor('silver')
ax3.set_ylabel('Y [km]')
ax3.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf3, cax=cax, orientation='vertical',
             label = 'Average Percent \n Anomaly Z')

#Detritus
cf4 =ax4.contourf(xrho, yrho, d_pdiff, 
            vmin = pparam['d'][0], vmax = pparam['d'][1],
            levels = np.arange(pparam['d'][0],pparam['d'][1]+pparam['d'][2],
                               pparam['d'][2]), 
            cmap=cmap)
ax4.grid(alpha = 0.5)
ax4.patch.set_facecolor('silver')
ax4.set_ylabel('Y [km]')
ax4.set_xlabel('X [km]')
ax4.set_aspect('equal', adjustable='box')
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf4, cax=cax, orientation='vertical',
             label = 'Average Percent \n Anomaly D')
