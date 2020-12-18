# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:44:53 2017

@author: pearseb
"""

#%% imports

from __future__ import unicode_literals

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import netCDF4 as nc
import cmocean.cm as cmo
from scipy.optimize import curve_fit
from matplotlib.animation import ArtistAnimation
import seaborn as sb
sb.set(style='ticks')


#%% get Rafter's dataset

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\data_for_publication')
d15N = np.genfromtxt('RafterTuerena_watercolumn_d15N_no3.txt',skip_header=1, usecols=(0,1,2,3,4))

# change the longitudes to positives
d15N[:,1][d15N[:,1]<0] += 360.0

# check dimensions
print(np.shape(d15N))

# check for nans in important columns
print(np.any(np.isnan(d15N[:,0])))
print(np.any(np.isnan(d15N[:,1])))
print(np.any(np.isnan(d15N[:,2])))
print(np.any(np.isnan(d15N[:,3])))
print(np.any(np.isnan(d15N[:,4])))

# check for how many values are -999
print(np.any(d15N[:,0]==-999))
print(np.any(d15N[:,1]==-999))
print(np.any(d15N[:,2]==-999))
print(np.any(d15N[:,3]==-999))
print(np.any(d15N[:,4]==-999))
print(len(d15N[d15N[:,4]==-999]))

# convert -999 values to NaNs
d15N[:,4][d15N[:,4]==-999] = np.nan
# count nans to make sure that all were converted
print(len(d15N[:,4][np.isnan(d15N[:,4])]))

print(len(d15N[:,0]), "data points", \
      "with", len(d15N[:,4][np.isnan(d15N[:,4])]), "NaNs in the Nitrate column")

print("unweighted mean delta15NO3 = ", np.mean(d15N[:,3]), " plus/minus ", np.std(d15N[:,3]) ) 
print("min delta15NO3 = ", np.min(d15N[:,3]) )
print("max delta15NO3 = ", np.max(d15N[:,3]) )


#%% load model grid

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\data_for_publication')
data = nc.Dataset('ETOPO_spinup_d15Nno3.nc', 'r')
print(data)
no3 = data.variables['NO3'][...]
lon = data.variables['ETOPO60X'][...]
lat = data.variables['ETOPO60Y'][...]
dep = data.variables['deptht'][...]

# fix longitudes
lon[lon>360.0] -= 360.0

dep_bnds = data.variables['deptht_bnds'][...]
lon_bnds = np.zeros((len(lon),2))
lat_bnds = np.zeros((len(lat),2))
lon_bnds[:,0] = lon[:]-0.5; lon_bnds[:,1] = lon[:]+0.5
lat_bnds[:,0] = lat[:]-0.5; lat_bnds[:,1] = lat[:]+0.5


#%% average data at each grid point of the model


d15n_obs_grid = np.zeros(np.shape(no3))
count = np.zeros(np.shape(no3))

# for each grid cell
for i in np.arange(len(lon)):
    print("longitude index %i"%i)
    for j in np.arange(len(lat)):
        for k in np.arange(len(dep)):
            
            cnt = 0.0
            val = 0.0
            
            # for every data point location
            for row in np.arange(len(d15N[:,0])):
                
                if d15N[row,1] >= lon_bnds[i,0] and d15N[row,1] < lon_bnds[i,1]:
                    if d15N[row,0] >= lat_bnds[j,0] and d15N[row,0] < lat_bnds[j,1]:
                        if d15N[row,2] >= dep_bnds[k,0] and d15N[row,2] < dep_bnds[k,1]:
                            cnt += 1.0
                            val += d15N[row,3]
                                  
            count[k,j,i] = cnt
            if cnt > 0:
                d15n_obs_grid[k,j,i] = val/cnt
            else:
                d15n_obs_grid[k,j,i] = np.nan


 #%%

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\data_for_publication')
np.savez('RafterTuerena_watercolumn_d15N_no3_gridded.npz', d15n_obs_grid)

