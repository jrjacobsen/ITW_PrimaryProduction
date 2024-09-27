import os
os.chdir('/home/jjacobs2/python/island_wave/subroutines/')
import sys
sys.path.append('/home/jjacobs2/python/island_wave/subroutines/')
import numpy as np
from netCDF4 import Dataset as nc4
from scipy.integrate import cumtrapz
import obs_depth_vs145 as dep
import matplotlib.pyplot as plt


#Directories & Files
###############################################################################
#files
file = 'output/my25_npzd04/itw/roms_dia_slice.nc'
root = '/home/jjacobs2/runs/island_wave/wind_forced/output/wind_forced/'
gridfile = 'grids/island_subgrid.nc'

#load files
hsfile = nc4(root+'output/my25_npzd04/itw/roms_his_subgrid.nc')
ncfile = nc4(root+file,'r')
grid = nc4(root+gridfile, 'r')

#figure directory
fig_dir = '/home/jjacobs2/python/island_wave/figures/figs_Jan2024/'
fig_name = 'Budget_IntRate_240926'

#select variable
var = 'dye_01'


#unit factor
dayunit = 3600*24

#Import variables
###############################################################################
#Slicing
idx = {}
idx['time'] = slice(0,-5)
idx['lat'] = slice(None,None)
idx['lon'] = slice(None,None)
idx['depth'] = slice(None,None)

#grid
xrho = grid.variables['x_rho'][idx['lat'],idx['lon']] - 120*1000
yrho = grid.variables['y_rho'][idx['lat'],idx['lon']] - 100*1000
mask = grid.variables['mask_rho'][idx['lat'],idx['lon']]
time = ncfile.variables['ocean_time'][idx['time']]/dayunit
dt_day = np.mean(np.diff(time))

#depth
romsvars = {'Vstretching' : hsfile.variables['Vstretching'][0], \
            'Vtransform' : hsfile.variables['Vtransform'][0], \
            'theta_s' : hsfile.variables['theta_s'][0], \
            'theta_b' : hsfile.variables['theta_b'][0], \
            'N' : hsfile.variables['temp'][0,:,0,0].shape[0], \
            'h' : grid.variables['h'][:], \
            'hc': hsfile.variables['hc'][0]}

depth = dep._set_depth(hsfile, romsvars, 'rho',
                       romsvars['h'])[idx['depth'],1,1]



#diagnostic variables
hadv = ncfile.variables[var+'_xadv'][idx['time'],idx['depth'],
                                     idx['lat'],idx['lon']] \
       +ncfile.variables[var+'_yadv'][idx['time'],idx['depth'],
                                      idx['lat'],idx['lon']]
vadv = ncfile.variables[var+'_vadv'][idx['time'],idx['depth'],
                                     idx['lat'],idx['lon']]
rate = ncfile.variables[var+'_rate'][idx['time'],idx['depth'],
                                     idx['lat'],idx['lon']]
vdiff = ncfile.variables[var+'_vdiff'][idx['time'],idx['depth'],
                                       idx['lat'],idx['lon']]


#Calculations
###############################################################################
#convert units
hadv_day = hadv*dayunit
vadv_day = vadv*dayunit
rate_day = rate*dayunit
vdif_day = vdiff*dayunit

#cumulative sum
hadv_sum = np.cumsum(hadv_day*dt_day, axis = 0)
vadv_sum = np.cumsum(vadv_day*dt_day, axis = 0)
rate_sum = np.cumsum(rate_day*dt_day, axis = 0)
vdif_sum = np.cumsum(vdif_day*dt_day, axis = 0)

#Convert Grid to polar
rad = np.sqrt(xrho**2 + yrho**2)
theta = np.arctan2(yrho, xrho)

#radius on one spoke
radround = np.round(rad/1000, decimals = 0)
rbar = np.unique(radround)
nrad = rbar.shape[0]

#azimuthal averaging
hadv_az = np.empty((hadv_sum.shape[0], depth.shape[0], rbar.shape[0]))
vadv_az = np.empty(hadv_az.shape)
rate_az = np.empty(hadv_az.shape)
vdif_az = np.empty(hadv_az.shape)
for t in range(hadv_sum.shape[0]) :
    for z in range(depth.shape[0]) :
        hadv_ = hadv_sum[t,z,:,:]
        vadv_ = vadv_sum[t,z,:,:]
        rate_ = rate_sum[t,z,:,:]
        vdif_ = vdif_sum[t,z,:,:]
        
        #averaging by constant radius
        for r in range(rbar.shape[0]) :
            hadv_az[t,z,r] = np.ma.mean(hadv_[rbar[r] == radround])
            vadv_az[t,z,r] = np.ma.mean(vadv_[rbar[r] == radround])
            rate_az[t,z,r] = np.ma.mean(rate_[rbar[r] == radround])
            vdif_az[t,z,r] = np.ma.mean(vdif_[rbar[r] == radround])
            

#Time average values over analysis period 
timebin = slice(192,None)
hadv_bar = np.mean(hadv_az[timebin,:,:], axis = 0)
vadv_bar = np.mean(vadv_az[timebin,:,:], axis = 0)
rate_bar = np.mean(rate_az[timebin,:,:], axis = 0)
diff_bar = np.mean(vdif_az[timebin,:,:], axis = 0)


#radial averaging
radbin = slice(5, 10)
rad_afd = np.mean(hadv_bar[:,radbin]+vadv_bar[:,radbin], axis = 1)
rad_rate = np.mean(rate_bar[:,radbin], axis = 1)
rad_dif = np.mean(diff_bar[:,radbin], axis = 1)

#residual
residual = rad_rate - (rad_afd +rad_dif)

#integrate rate
rate_vint = cumtrapz(rad_rate, x = depth)

pad = np.concatenate((rate_vint[0:1], rate_vint, rate_vint[-2:-1]))
vint = 0.5*(pad[1:]+pad[:-1])

#plotting
###############################################################################

fig, (ax1) = plt.subplots(1,1, figsize = (4,8), sharey = True)
ax1.plot(rad_afd, depth, 
         color = 'tab:blue', linewidth = 2,
         label = r'$s_{AFD}$')
ax1.plot(rad_rate, depth, 
         color = 'k', linewidth = 3,
         label = '$s_{rate}$')
ax1.plot(rad_dif, depth, 
         color = 'red', linestyle = '--',linewidth = 2.5,
         label = r'$s_{diff}$')
ax1.plot(residual, depth, 
         'k:', linewidth = 2, 
         label = 'Residual')
ax1.grid()
ax1.set_xlim([-0.02,0.02])
ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('[mmol s m$^{-3}$ day$^{-1}$]')
ax1.legend(bbox_to_anchor = (-0.25,1))


# sav_str = fig_dir+fig_name+'_NoLegRedDash.png'
# plt.savefig(sav_str, bbox_inches='tight', dpi = 300)

fig, ax2 = plt.subplots(1,1, figsize = (4,8))
ax2.plot(vint, depth, 
         color = 'k', linewidth = 2,
         linestyle = '--')
ax2.grid()
ax2.set_title(r'$\int s_{rate} dz^{\prime}$ ')


# sav_str = fig_dir+fig_name+'_RedDash.png'
# plt.savefig(sav_str, bbox_inches='tight', dpi = 300)
