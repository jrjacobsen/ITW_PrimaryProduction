#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:25:17 2023

@author: jjacob2
"""
import os
os.chdir('/home/jjacobs2/python/island_wave/subroutines/')
import sys
sys.path.append('/home/jjacobs2/python/island_wave/subroutines/')
import numpy as np
from netCDF4 import Dataset as nc4
import obs_depth_vs145 as dep
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
from subsetting import island_ts as locator

#Directories & Files
###############################################################################
##interp file
file_r = 'my25_npzd04/itw'
#file_u = 'usineWind'
file_l = 'my25_npzd04/ulandsea_itw'
file_c = 'my25_npzd04/counter_diurnal_noisland'
#file = 'freewave_rotaWind'


#grid file
root = '/proj/jjacobs2_enstrophy/runs/island_wave/'
gridfile = root+'wind_forced/grids/island_subgrid.nc'

#figure directory
fig_dir = '/home/jjacobs2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'ITW_NormalizedWindForcing_240423'

#load files
ncfile_r = nc4(root+'wind_forced/output/'+file_r+'/roms_his_slice.nc', 'r') 
#ncfile_u = nc4(root+'wind_forced/output/'+file_u+'_itw/roms_his_slice.nc', 'r') 
ncfile_l = nc4(root+'wind_forced/output/'+file_l+'/roms_his_slice.nc', 'r') 
ncfile_c = nc4(root+'wind_forced/output/'+file_c+'/roms_his_slice.nc', 'r') 

grid = nc4(gridfile, 'r')

#Dimensions, Slicging, and variables
###############################################################################
#spatial slicing
'''
'sw' or  1  => southwest corner
'se' or  2  => southeast corner
'ne' or  3  => northeast corner
'nw' or  4  => northwest corner
'''
idx, title = locator(1)

#time slicing
idx['time'] = slice(192,-12)

#dimensions
time = ncfile_r.variables['ocean_time'][idx['time']]/(3600*24) #days
t = time[::2]

def load(ncfile, idx) :
    #wind stress
    ustr = ncfile.variables['sustr'][idx['time'], idx['lat'], idx['lon']]
    vstr = ncfile.variables['svstr'][idx['time'], idx['lat'], idx['lon']]
    u = 0.5*ustr[::2]/np.sqrt(ustr[::2]**2+vstr[::2]**2)
    v = 0.5*vstr[::2]/np.sqrt(ustr[::2]**2+vstr[::2]**2)
    
    return u, v

u_r, v_r = load(ncfile_r, idx)
#v_u, u_u = load(ncfile_u, idx)
u_l, v_l = load(ncfile_l, idx)
u_c, v_c = load(ncfile_c, idx)

#Plotting
###############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (8,7), 
      sharey = True, sharex = True)
#Rotational
for i in range(t[:].shape[0]) :
    #ax1.plot([t[i], t[i]+u_r[i]*1.5], [0,v_r[[i]]*1.5], 'r')
    ax1.plot([t[i], t[i]+u_r[i]], [0,v_r[[i]]], 'k')
    #ax1.plot([t[i], t[i]+u_r[i]*0.5], [0,v_r[[i]]*0.5], 'b')
ax1.set_ylabel('Normalized  \n Wind Stress')
ax1.grid()

#Land sea
for i in range(t[:].shape[0]) :
    ax2.plot([t[i], t[i]+v_l[i]], [0,u_l[[i]]], 'k')
ax2.set_ylabel('Normalized  \n Wind Stress')
ax2.grid()

#Counter rotational
for i in range(t[:].shape[0]) :
    ax3.plot([t[i], t[i]+u_c[i]], [0,v_c[[i]]], 'k')


ax3.set_ylabel('Normalized  \n Wind Stress')
ax3.set_xlabel('Time [Days]')
ax3.set_xticks(np.arange(4,8,1))
ax3.set_ylim([-0.75,0.75])
#ax3.set_yticks(np.arange(-1, 1.5,0.5))
ax3.grid()

#save figure
###############################################################################
#sav_str = fig_dir+fig_name+'.png'
#plt.savefig(sav_str, bbox_inches='tight', dpi = 300)



