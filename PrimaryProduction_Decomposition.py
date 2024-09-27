#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:06:27 2023

@author: jjacob2
"""
import os
os.chdir('/home/jjacobs2/python/island_wave/subroutines/')
import sys
sys.path.append('/home/jjacobs2/python/island_wave/subroutines/')
import numpy as np
import scipy as sp
from netCDF4 import Dataset as nc4
import obs_depth_vs145 as dep
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
from subsetting import island_ts as locator
import subsetting
import diag_itw as diag

#File Directories
###########################################################################
#file = 'lightconst_noForceNudIsland'
file = 'my25_npzd04'

root = '/proj/jjacobs2_enstrophy/runs/island_wave/wind_forced/'
gridfile = root+'grids/island_subgrid.nc'

#load files
ncfile = nc4(root+'output/'+file+'/diurnal_itw/roms_his_slice.nc', 'r') 

grid = nc4(gridfile, 'r')

#title = 'Eastside Diurnal Light NPZD v4'
title = ' '

#figure directory
fig_dir = '/home/jjacobs2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'PrimProdDecomp_DiurnLight_240420'


#Grid and slicing
###########################################################################
#slicing
idx = {}
idx['time'] = slice(192,-12)
idx['lat'] = 41
idx['lon_e'] = slice(51,80) #east
idx['lon_w'] = slice(0,50) #west
idx['depth'] = slice(None, None)

#grid
time = ncfile.variables['ocean_time'][idx['time']]/3600
xrho = grid.variables['x_rho'][:]/1000 - 120

#depth
romsvars = {'Vstretching' : ncfile.variables['Vstretching'][0], \
            'Vtransform' : ncfile.variables['Vtransform'][0], \
            'theta_s' : ncfile.variables['theta_s'][0], \
            'theta_b' : ncfile.variables['theta_b'][0], \
            'N' : ncfile.variables['temp'][0,:,0,0].shape[0], \
            'h' : grid.variables['h'][:], \
            'hc': ncfile.variables['hc'][0]}

depth_e = dep._set_depth(ncfile, romsvars, 'rho',
                       romsvars['h'])[:,idx['lat'],idx['lon_e']]
depth_w = dep._set_depth(ncfile, romsvars, 'rho',
                       romsvars['h'])[:,idx['lat'],idx['lon_w']]

#distance from island coast
drho_e = np.sqrt(xrho[idx['lat'], idx['lon_e']]**2) - 5.25
drho_w = np.sqrt(xrho[idx['lat'], idx['lon_w']]**2) - 5.25

xx_e = np.repeat(drho_e[np.newaxis,:], 
              repeats = depth_e.shape[0], axis = 0)
xx_w = np.repeat(drho_w[np.newaxis,:], 
              repeats = depth_w.shape[0], axis = 0)

#bio params
bio = subsetting.bio_params(ncfile)


#import variables
###########################################################################
#NP sw - East
n_e = ncfile.variables['NO3'][idx['time'], idx['depth'], 
                          idx['lat'], idx['lon_e']]
p_e = ncfile.variables['phytoplankton'][idx['time'], idx['depth'], 
                          idx['lat'], idx['lon_e']]
swrad_e = ncfile.variables['swrad'][idx['time'],idx['lat'],idx['lon_e']]

#NP sw - West
n_w = ncfile.variables['NO3'][idx['time'], idx['depth'], 
                          idx['lat'], idx['lon_w']]
p_w = ncfile.variables['phytoplankton'][idx['time'], idx['depth'], 
                          idx['lat'], idx['lon_w']]
swrad_w = ncfile.variables['swrad'][idx['time'],idx['lat'],idx['lon_w']]



#calculations
###########################################################################
#light fraction
#EAST
lightbar_e = np.mean(swrad_e[:48,:], axis = 0)
light_e = swrad_e/lightbar_e
swbar_e = np.mean(swrad_e, axis = 0)
ibar2d_e = np.repeat(swbar_e[np.newaxis,:],
                   repeats = n_e.shape[1], axis = 0)
ibar3d_e = np.repeat(ibar2d_e[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
iprime_e = np.repeat((swrad_e - swbar_e)[:,np.newaxis,:],
                   repeats = depth_e.shape[0], axis = 1)
#constant values
alpha_e = bio['Vm']*np.exp(bio['kext']*(depth_e))/lightbar_e

#average values
#Nutrient
nbar_e = np.mean(n_e, axis = 0)
nstd_e = np.std(n_e-nbar_e, axis = 0)
nmet = nstd_e/(bio['ks']+nbar_e)
nbar3d_e = np.repeat(nbar_e[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
nprime_e = n_e - nbar_e

#Phyt
pbar_e = np.mean(p_e, axis = 0)
pbar3d_e = np.repeat(pbar_e[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
pprime_e = p_e - pbar_e

#uptake factor
upfac_e = alpha_e/(bio['ks']+nbar_e)

#average of product of averages
proavg_e = upfac_e*pbar_e*nbar_e*swbar_e

#anomaly average product
Pni = pbar3d_e*nprime_e*iprime_e
Npi = nbar3d_e*pprime_e*iprime_e
Inp = ibar3d_e*pprime_e*nprime_e
npi = nprime_e*pprime_e*iprime_e
proprimebar_e = upfac_e*np.mean(Pni+Npi+Inp+npi, axis = 0)

#WEST
lightbar_w = np.mean(swrad_w[:48,:], axis = 0)
light_w = swrad_w/lightbar_w
swbar_w = np.mean(swrad_w, axis = 0)
ibar2d_w = np.repeat(swbar_w[np.newaxis,:],
                   repeats = n_w.shape[1], axis = 0)
ibar3d_w = np.repeat(ibar2d_w[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
iprime_w = np.repeat((swrad_w - swbar_w)[:,np.newaxis,:],
                   repeats = depth_w.shape[0], axis = 1)
#constant values
alpha_w = bio['Vm']*np.exp(bio['kext']*(depth_w))/lightbar_w

#average values
#Nutrient
nbar_w = np.mean(n_w, axis = 0)
nstd_w = np.std(n_w-nbar_w, axis = 0)
nmet = nstd_w/(bio['ks']+nbar_w)
nbar3d_w = np.repeat(nbar_w[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
nprime_w = n_w - nbar_w

#Phyt
pbar_w = np.mean(p_w, axis = 0)
pbar3d_w = np.repeat(pbar_w[np.newaxis,:,:],
                   repeats = time.shape[0], axis = 0)
pprime_w = p_w - pbar_w

#uptake factor
upfac_w = alpha_w/(bio['ks']+nbar_w)

#average of product of averages
proavg_w = upfac_w*pbar_w*nbar_w*swbar_w

#anomaly average product
Pni = pbar3d_w*nprime_w*iprime_w
Npi = nbar3d_w*pprime_w*iprime_w
Inp = ibar3d_w*pprime_w*nprime_w
npi = nprime_w*pprime_w*iprime_w
proprimebar_w = upfac_w*np.mean(Pni+Npi+Inp+npi, axis = 0)

#plotting
###########################################################################
clim = (-0.002, 0.002, 0.00025) #PP FLUX

cmap = cmocean.cm.balance

fig, (ax1, ax2) = plt.subplots(2,1, figsize = (7,7),
      sharey = True)
cf1 = ax1.contourf(xx_e, depth_e, proprimebar_e,
                  cmap = cmap,
                  vmin = clim[0], vmax = clim[1],
                  levels = np.arange(clim[0], clim[1]+clim[2], clim[2]))
ax1.set_title(title, loc = 'left')
ax1.set_ylabel('depth [m]')
ax1.text(13, -8, 'East')
ax1.set_xlim(0,14)
ax1.set_ylim(-100,0)
ax1.grid()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf1, cax=cax, orientation='vertical', 
             label = r'$\langle PP^{\prime} \rangle$'+ '\n [mmol N m$^{-3}$ d$^{-1}$]')


cf2 = ax2.contourf(np.abs(xx_w), depth_w, proprimebar_w,
                  cmap = cmap,
                  vmin = clim[0], vmax = clim[1],
                  levels = np.arange(clim[0], clim[1]+clim[2], clim[2]))
ax2.set_ylabel('depth [m]')
ax2.set_xlabel('Dist. from Island [km]')
ax2.set_xlim(0,14)
ax2.text(13, -8, 'West')
ax2.grid()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cf2, cax=cax, orientation='vertical', 
             label = r'$\langle PP^{\prime} \rangle$'+ '\n [mmol N m$^{-3}$ d$^{-1}$]')


#save figure
###############################################################################
#sav_str = fig_dir+fig_name+'.png'
#plt.savefig(sav_str, bbox_inches='tight', dpi = 300)
