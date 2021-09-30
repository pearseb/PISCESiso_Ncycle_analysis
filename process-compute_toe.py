# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:29:17 2020


Purpose
-------
    Calculate the Time of Emergence (ToE) of trends in bgc variables

@author: pearseb
"""

#%% imports

from __future__ import unicode_literals

import os
import numpy as np
import netCDF4 as nc
import seaborn as sb
sb.set(style='ticks')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean
import cmocean.cm as cmo

import mpl_toolkits.basemap as bm
from tqdm import tqdm


#%% get the data on each depth level


os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

data = nc.Dataset('ETOPO_picontrol_1y_no3_ez_utz_ltz.nc','r')
no3_pic_ez = data.variables['NO3A_EZ_ZAVE'][50:300,...]
no3_pic_utz = data.variables['NO3A_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_1y_nst_ez_utz_ltz.nc','r')
nst_pic_ez = data.variables['NSTA_EZ_ZAVE'][50:300,...]
nst_pic_utz = data.variables['NSTA_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_1y_d15n_no3_ez_utz_ltz.nc','r')
d15n_no3_pic_ez = data.variables['D15N_NO3_EZ_ZAVE'][50:300,...]
d15n_no3_pic_utz = data.variables['D15N_NO3_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_1y_d15n_pom_ez_utz_ltz.nc','r')
d15n_pom_pic_ez = data.variables['D15N_POM_EZ_ZAVE'][50:300,...]
d15n_pom_pic_utz = data.variables['D15N_POM_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_ndep_1y_no3_ez_utz_ltz.nc','r')
no3_picndep_ez = data.variables['NO3A_EZ_ZAVE'][50:300,...]
no3_picndep_utz = data.variables['NO3A_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_ndep_1y_nst_ez_utz_ltz.nc','r')
nst_picndep_ez = data.variables['NSTA_EZ_ZAVE'][50:300,...]
nst_picndep_utz = data.variables['NSTA_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_ndep_1y_d15n_no3_ez_utz_ltz.nc','r')
d15n_no3_picndep_ez = data.variables['D15N_NO3_EZ_ZAVE'][50:300,...]
d15n_no3_picndep_utz = data.variables['D15N_NO3_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_picontrol_ndep_1y_d15n_pom_ez_utz_ltz.nc','r')
d15n_pom_picndep_ez = data.variables['D15N_POM_EZ_ZAVE'][50:300,...]
d15n_pom_picndep_utz = data.variables['D15N_POM_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_future_1y_no3_ez_utz_ltz.nc','r')
no3_fut_ez = data.variables['NO3A_EZ_ZAVE'][...]
no3_fut_utz = data.variables['NO3A_UTZ_ZAVE'][...]

time = data.variables['TIME_COUNTER11'][...]/86400/365+1899
lon = data.variables['ETOPO60X'][...]
lat = data.variables['ETOPO60Y'][...]
lon -= 360

data = nc.Dataset('ETOPO_future_1y_nst_ez_utz_ltz.nc','r')
nst_fut_ez = data.variables['NSTA_EZ_ZAVE'][...]
nst_fut_utz = data.variables['NSTA_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_1y_d15n_no3_ez_utz_ltz.nc','r')
d15n_no3_fut_ez = data.variables['D15N_NO3_EZ_ZAVE'][...]
d15n_no3_fut_utz = data.variables['D15N_NO3_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_1y_d15n_pom_ez_utz_ltz.nc','r')
d15n_pom_fut_ez = data.variables['D15N_POM_EZ_ZAVE'][...]
d15n_pom_fut_utz = data.variables['D15N_POM_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_ndep_1y_no3_ez_utz_ltz.nc','r')
no3_futndep_ez = data.variables['NO3A_EZ_ZAVE'][...]
no3_futndep_utz = data.variables['NO3A_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_ndep_1y_nst_ez_utz_ltz.nc','r')
nst_futndep_ez = data.variables['NSTA_EZ_ZAVE'][...]
nst_futndep_utz = data.variables['NSTA_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_ndep_1y_d15n_no3_ez_utz_ltz.nc','r')
d15n_no3_futndep_ez = data.variables['D15N_NO3_EZ_ZAVE'][...]
d15n_no3_futndep_utz = data.variables['D15N_NO3_UTZ_ZAVE'][...]

data = nc.Dataset('ETOPO_future_ndep_1y_d15n_pom_ez_utz_ltz.nc','r')
d15n_pom_futndep_ez = data.variables['D15N_POM_EZ_ZAVE'][...]
d15n_pom_futndep_utz = data.variables['D15N_POM_UTZ_ZAVE'][...]

# Temperature
data = nc.Dataset('ETOPO_picontrol_1y_temp_ez_utz_ltz.nc','r')
temp_pic_ez = data.variables['TEMP_EZ_ZAVE'][50:300,...]
temp_pic_utz = data.variables['TEMP_UTZ_ZAVE'][50:300,...]

data = nc.Dataset('ETOPO_future_1y_temp_ez_utz_ltz.nc','r')
temp_fut_ez = data.variables['TEMP_EZ_ZAVE'][...]
temp_fut_utz = data.variables['TEMP_UTZ_ZAVE'][...]

# NPP
mask = np.ma.getmask(temp_pic_ez)
data = nc.Dataset('ETOPO_picontrol_1y_npp.nc','r')
npp_pic = np.ma.masked_where(mask, data.variables['NPP_ZINT'][50:300,...])
data = nc.Dataset('ETOPO_picontrol_ndep_1y_npp.nc','r')
npp_picndep = np.ma.masked_where(mask, data.variables['NPP_ZINT'][50:300,...])
data = nc.Dataset('ETOPO_future_1y_npp.nc','r')
npp_fut = np.ma.masked_where(mask, data.variables['NPP_ZINT'][1::,...])
data = nc.Dataset('ETOPO_future_ndep_1y_npp.nc','r')
npp_futndep = np.ma.masked_where(mask, data.variables['NPP_ZINT'][1::,...])


# N2 fixation
data = nc.Dataset('ETOPO_picontrol_1y_nfix.nc','r')
fix_pic = np.ma.masked_where(mask, data.variables['FIX_ZINT'][50:300,...])
data = nc.Dataset('ETOPO_picontrol_ndep_1y_nfix.nc','r')
fix_picndep = np.ma.masked_where(mask, data.variables['FIX_ZINT'][50:300,...])
data = nc.Dataset('ETOPO_future_1y_nfix.nc','r')
fix_fut = np.ma.masked_where(mask, data.variables['FIX_ZINT'][1::,...])
data = nc.Dataset('ETOPO_future_ndep_1y_nfix.nc','r')
fix_futndep = np.ma.masked_where(mask, data.variables['FIX_ZINT'][1::,...])


data.close()


#%% detrend the data in each grid cell

# find slope of picontrol at each grid cell by iterating over the grid
from scipy.optimize import curve_fit
def func(x,a,b):
    return a*x + b

fix_pic_slope = np.zeros((180,360))
npp_pic_slope = np.zeros((180,360))

temp_pic_ez_slope = np.zeros((180,360))
temp_pic_utz_slope = np.zeros((180,360))

no3_pic_ez_slope = np.zeros((180,360))
no3_pic_utz_slope = np.zeros((180,360))

nst_pic_ez_slope = np.zeros((180,360))
nst_pic_utz_slope = np.zeros((180,360))

d15n_no3_pic_ez_slope = np.zeros((180,360))
d15n_no3_pic_utz_slope = np.zeros((180,360))

d15n_pom_pic_ez_slope = np.zeros((180,360))
d15n_pom_pic_utz_slope = np.zeros((180,360))


for i in tqdm(np.arange(len(no3_pic_ez[0,0,:])), desc="longitudes", position=1):
    for j in np.arange(len(no3_pic_ez[0,:,0])):
        if np.ma.is_mask(fix_pic[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, fix_pic[:,j,i])
            fix_pic_slope[j,i] = popt[0]

        if np.ma.is_mask(npp_pic[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, npp_pic[:,j,i])
            npp_pic_slope[j,i] = popt[0]

        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, temp_pic_ez[:,j,i])
            temp_pic_ez_slope[j,i] = popt[0]
        if np.ma.is_mask(temp_pic_utz[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, temp_pic_utz[:,j,i])
            temp_pic_utz_slope[j,i] = popt[0]
        
        if np.ma.is_mask(no3_pic_ez[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, no3_pic_ez[:,j,i])
            no3_pic_ez_slope[j,i] = popt[0]
        if np.ma.is_mask(no3_pic_utz[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, no3_pic_utz[:,j,i])
            no3_pic_utz_slope[j,i] = popt[0]
        
        if np.ma.is_mask(nst_pic_ez[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, nst_pic_ez[:,j,i])
            nst_pic_ez_slope[j,i] = popt[0]
        if np.ma.is_mask(nst_pic_utz[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, nst_pic_utz[:,j,i])
            nst_pic_utz_slope[j,i] = popt[0]
        
        if np.ma.is_mask(d15n_no3_pic_ez[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, d15n_no3_pic_ez[:,j,i])
            d15n_no3_pic_ez_slope[j,i] = popt[0]
        if np.ma.is_mask(d15n_no3_pic_utz[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, d15n_no3_pic_utz[:,j,i])
            d15n_no3_pic_utz_slope[j,i] = popt[0]
        
        if np.ma.is_mask(d15n_pom_pic_ez[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, d15n_pom_pic_ez[:,j,i])
            d15n_pom_pic_ez_slope[j,i] = popt[0]
        if np.ma.is_mask(d15n_pom_pic_utz[0,j,i]) == False:
            popt, pcov = curve_fit(func, time, d15n_pom_pic_utz[:,j,i])
            d15n_pom_pic_utz_slope[j,i] = popt[0]

        
        
# stack 2D slope array 250 times
mask = np.ma.getmask(fix_pic); fix_pic_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*fix_pic_slope[np.newaxis,:,:])
mask = np.ma.getmask(npp_pic); npp_pic_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*npp_pic_slope[np.newaxis,:,:])

mask = np.ma.getmask(temp_pic_ez); temp_pic_ez_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*temp_pic_ez_slope[np.newaxis,:,:])
mask = np.ma.getmask(temp_pic_utz); temp_pic_utz_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*temp_pic_utz_slope[np.newaxis,:,:])

mask = np.ma.getmask(no3_pic_ez); no3_pic_ez_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*no3_pic_ez_slope[np.newaxis,:,:])
mask = np.ma.getmask(no3_pic_utz); no3_pic_utz_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*no3_pic_utz_slope[np.newaxis,:,:])

mask = np.ma.getmask(nst_pic_ez); nst_pic_ez_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*nst_pic_ez_slope[np.newaxis,:,:])
mask = np.ma.getmask(nst_pic_utz); nst_pic_utz_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*nst_pic_utz_slope[np.newaxis,:,:])

mask = np.ma.getmask(d15n_no3_pic_ez); d15n_no3_pic_ez_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*d15n_no3_pic_ez_slope[np.newaxis,:,:])
mask = np.ma.getmask(d15n_no3_pic_utz); d15n_no3_pic_utz_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*d15n_no3_pic_utz_slope[np.newaxis,:,:])

mask = np.ma.getmask(d15n_pom_pic_ez); d15n_pom_pic_ez_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*d15n_pom_pic_ez_slope[np.newaxis,:,:])
mask = np.ma.getmask(d15n_pom_pic_utz); d15n_pom_pic_utz_slope_x = np.ma.masked_where(mask, np.ones((250,180,360))*d15n_pom_pic_utz_slope[np.newaxis,:,:])



# detrend data
fix_pic_detrended = fix_pic - (fix_pic_slope_x*time[:,np.newaxis,np.newaxis])
fix_picndep_detrended = fix_picndep - (fix_pic_slope_x*time[:,np.newaxis,np.newaxis])
fix_fut_detrended = fix_fut - (fix_pic_slope_x*time[:,np.newaxis,np.newaxis])
fix_futndep_detrended = fix_futndep - (fix_pic_slope_x*time[:,np.newaxis,np.newaxis])

npp_pic_detrended = npp_pic - (npp_pic_slope_x*time[:,np.newaxis,np.newaxis])
npp_picndep_detrended = npp_picndep - (npp_pic_slope_x*time[:,np.newaxis,np.newaxis])
npp_fut_detrended = npp_fut - (npp_pic_slope_x*time[:,np.newaxis,np.newaxis])
npp_futndep_detrended = npp_futndep - (npp_pic_slope_x*time[:,np.newaxis,np.newaxis])


temp_pic_ez_detrended = temp_pic_ez - (temp_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
temp_pic_utz_detrended = temp_pic_utz - (temp_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

no3_pic_ez_detrended = no3_pic_ez - (no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
no3_pic_utz_detrended = no3_pic_utz - (no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

nst_pic_ez_detrended = nst_pic_ez - (nst_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
nst_pic_utz_detrended = nst_pic_utz - (nst_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_no3_pic_ez_detrended = d15n_no3_pic_ez - (d15n_no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_no3_pic_utz_detrended = d15n_no3_pic_utz - (d15n_no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_pom_pic_ez_detrended = d15n_pom_pic_ez - (d15n_pom_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_pom_pic_utz_detrended = d15n_pom_pic_utz - (d15n_pom_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])


no3_picndep_ez_detrended = no3_picndep_ez - (no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
no3_picndep_utz_detrended = no3_picndep_utz - (no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

nst_picndep_ez_detrended = nst_picndep_ez - (nst_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
nst_picndep_utz_detrended = nst_picndep_utz - (nst_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_no3_picndep_ez_detrended = d15n_no3_picndep_ez - (d15n_no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_no3_picndep_utz_detrended = d15n_no3_picndep_utz - (d15n_no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_pom_picndep_ez_detrended = d15n_pom_picndep_ez - (d15n_pom_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_pom_picndep_utz_detrended = d15n_pom_picndep_utz - (d15n_pom_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])


temp_fut_ez_detrended = temp_fut_ez - (temp_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
temp_fut_utz_detrended = temp_fut_utz - (temp_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

no3_fut_ez_detrended = no3_fut_ez - (no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
no3_fut_utz_detrended = no3_fut_utz - (no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

nst_fut_ez_detrended = nst_fut_ez - (nst_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
nst_fut_utz_detrended = nst_fut_utz - (nst_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_no3_fut_ez_detrended = d15n_no3_fut_ez - (d15n_no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_no3_fut_utz_detrended = d15n_no3_fut_utz - (d15n_no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_pom_fut_ez_detrended = d15n_pom_fut_ez - (d15n_pom_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_pom_fut_utz_detrended = d15n_pom_fut_utz - (d15n_pom_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])


no3_futndep_ez_detrended = no3_futndep_ez - (no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
no3_futndep_utz_detrended = no3_futndep_utz - (no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

nst_futndep_ez_detrended = nst_futndep_ez - (nst_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
nst_futndep_utz_detrended = nst_futndep_utz - (nst_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_no3_futndep_ez_detrended = d15n_no3_futndep_ez - (d15n_no3_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_no3_futndep_utz_detrended = d15n_no3_futndep_utz - (d15n_no3_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])

d15n_pom_futndep_ez_detrended = d15n_pom_futndep_ez - (d15n_pom_pic_ez_slope_x*time[:,np.newaxis,np.newaxis])
d15n_pom_futndep_utz_detrended = d15n_pom_futndep_utz - (d15n_pom_pic_utz_slope_x*time[:,np.newaxis,np.newaxis])



#%% remove average value of picontrol run from trend so the values vary about zero, and take the difference between the two trends

# subtract detrended picontrol values from themselves and from climate change experiments
fix_pic_normalised = fix_pic_detrended - np.ma.mean(fix_pic_detrended, axis=0)
fix_picndep_normalised = fix_picndep_detrended - np.ma.mean(fix_pic_detrended, axis=0) 
fix_fut_normalised = fix_fut_detrended - np.ma.mean(fix_pic_detrended, axis=0) 
fix_futndep_normalised = fix_futndep_detrended - np.ma.mean(fix_pic_detrended, axis=0)

npp_pic_normalised = npp_pic_detrended - np.ma.mean(npp_pic_detrended, axis=0)
npp_picndep_normalised = npp_picndep_detrended - np.ma.mean(npp_pic_detrended, axis=0) 
npp_fut_normalised = npp_fut_detrended - np.ma.mean(npp_pic_detrended, axis=0) 
npp_futndep_normalised = npp_futndep_detrended - np.ma.mean(npp_pic_detrended, axis=0)


temp_pic_ez_normalised = temp_pic_ez_detrended - np.ma.mean(temp_pic_ez_detrended, axis=0)
temp_pic_utz_normalised = temp_pic_utz_detrended - np.ma.mean(temp_pic_utz_detrended, axis=0) 

no3_pic_ez_normalised = no3_pic_ez_detrended - np.ma.mean(no3_pic_ez_detrended, axis=0)
no3_pic_utz_normalised = no3_pic_utz_detrended - np.ma.mean(no3_pic_utz_detrended, axis=0) 

nst_pic_ez_normalised = nst_pic_ez_detrended - np.ma.mean(nst_pic_ez_detrended, axis=0)
nst_pic_utz_normalised = nst_pic_utz_detrended - np.ma.mean(nst_pic_utz_detrended, axis=0)

d15n_no3_pic_ez_normalised = d15n_no3_pic_ez_detrended - np.ma.mean(d15n_no3_pic_ez_detrended, axis=0)
d15n_no3_pic_utz_normalised = d15n_no3_pic_utz_detrended - np.ma.mean(d15n_no3_pic_utz_detrended, axis=0)

d15n_pom_pic_ez_normalised = d15n_pom_pic_ez_detrended - np.ma.mean(d15n_pom_pic_ez_detrended, axis=0)
d15n_pom_pic_utz_normalised = d15n_pom_pic_utz_detrended - np.ma.mean(d15n_pom_pic_utz_detrended, axis=0)


no3_picndep_ez_normalised = no3_picndep_ez_detrended - np.ma.mean(no3_pic_ez_detrended, axis=0)
no3_picndep_utz_normalised = no3_picndep_utz_detrended - np.ma.mean(no3_pic_utz_detrended, axis=0)

nst_picndep_ez_normalised = nst_picndep_ez_detrended - np.ma.mean(nst_pic_ez_detrended, axis=0)
nst_picndep_utz_normalised = nst_picndep_utz_detrended - np.ma.mean(nst_pic_utz_detrended, axis=0)

d15n_no3_picndep_ez_normalised = d15n_no3_picndep_ez_detrended - np.ma.mean(d15n_no3_pic_ez_detrended, axis=0)
d15n_no3_picndep_utz_normalised = d15n_no3_picndep_utz_detrended - np.ma.mean(d15n_no3_pic_utz_detrended, axis=0)

d15n_pom_picndep_ez_normalised = d15n_pom_picndep_ez_detrended - np.ma.mean(d15n_pom_pic_ez_detrended, axis=0)
d15n_pom_picndep_utz_normalised = d15n_pom_picndep_utz_detrended - np.ma.mean(d15n_pom_pic_utz_detrended, axis=0)


temp_fut_ez_normalised = temp_fut_ez_detrended - np.ma.mean(temp_pic_ez_detrended, axis=0)
temp_fut_utz_normalised = temp_fut_utz_detrended - np.ma.mean(temp_pic_utz_detrended, axis=0)

no3_fut_ez_normalised = no3_fut_ez_detrended - np.ma.mean(no3_pic_ez_detrended, axis=0)
no3_fut_utz_normalised = no3_fut_utz_detrended - np.ma.mean(no3_pic_utz_detrended, axis=0)

nst_fut_ez_normalised = nst_fut_ez_detrended - np.ma.mean(nst_pic_ez_detrended, axis=0)
nst_fut_utz_normalised = nst_fut_utz_detrended - np.ma.mean(nst_pic_utz_detrended, axis=0)

d15n_no3_fut_ez_normalised = d15n_no3_fut_ez_detrended - np.ma.mean(d15n_no3_pic_ez_detrended, axis=0)
d15n_no3_fut_utz_normalised = d15n_no3_fut_utz_detrended - np.ma.mean(d15n_no3_pic_utz_detrended, axis=0)

d15n_pom_fut_ez_normalised = d15n_pom_fut_ez_detrended - np.ma.mean(d15n_pom_pic_ez_detrended, axis=0)
d15n_pom_fut_utz_normalised = d15n_pom_fut_utz_detrended - np.ma.mean(d15n_pom_pic_utz_detrended, axis=0)


no3_futndep_ez_normalised = no3_futndep_ez_detrended - np.ma.mean(no3_pic_ez_detrended, axis=0)
no3_futndep_utz_normalised = no3_futndep_utz_detrended - np.ma.mean(no3_pic_utz_detrended, axis=0)

nst_futndep_ez_normalised = nst_futndep_ez_detrended - np.ma.mean(nst_pic_ez_detrended, axis=0)
nst_futndep_utz_normalised = nst_futndep_utz_detrended - np.ma.mean(nst_pic_utz_detrended, axis=0)

d15n_no3_futndep_ez_normalised = d15n_no3_futndep_ez_detrended - np.ma.mean(d15n_no3_pic_ez_detrended, axis=0)
d15n_no3_futndep_utz_normalised = d15n_no3_futndep_utz_detrended - np.ma.mean(d15n_no3_pic_utz_detrended, axis=0)

d15n_pom_futndep_ez_normalised = d15n_pom_futndep_ez_detrended - np.ma.mean(d15n_pom_pic_ez_detrended, axis=0)
d15n_pom_futndep_utz_normalised = d15n_pom_futndep_utz_detrended - np.ma.mean(d15n_pom_pic_utz_detrended, axis=0)



# find trends via decadal smoothing (11-year boxcar, or "flat", window)
def smooth(x, window_len=11, window='flat'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


fix_pic_smoothed = np.ma.masked_where(np.ma.getmask(fix_pic), np.zeros((250,180,360)))
fix_picndep_smoothed = np.ma.masked_where(np.ma.getmask(fix_picndep), np.zeros((250,180,360)))
fix_fut_smoothed = np.ma.masked_where(np.ma.getmask(fix_fut), np.zeros((250,180,360)))
fix_futndep_smoothed = np.ma.masked_where(np.ma.getmask(fix_futndep), np.zeros((250,180,360)))

npp_pic_smoothed = np.ma.masked_where(np.ma.getmask(npp_pic), np.zeros((250,180,360)))
npp_picndep_smoothed = np.ma.masked_where(np.ma.getmask(npp_picndep), np.zeros((250,180,360)))
npp_fut_smoothed = np.ma.masked_where(np.ma.getmask(npp_fut), np.zeros((250,180,360)))
npp_futndep_smoothed = np.ma.masked_where(np.ma.getmask(npp_futndep), np.zeros((250,180,360)))


temp_pic_ez_smoothed = np.ma.masked_where(np.ma.getmask(temp_pic_ez), np.zeros((250,180,360)))
temp_pic_utz_smoothed = np.ma.masked_where(np.ma.getmask(temp_pic_utz), np.zeros((250,180,360)))

no3_pic_ez_smoothed = np.ma.masked_where(np.ma.getmask(no3_pic_ez), np.zeros((250,180,360)))
no3_pic_utz_smoothed = np.ma.masked_where(np.ma.getmask(no3_pic_utz), np.zeros((250,180,360)))

nst_pic_ez_smoothed = np.ma.masked_where(np.ma.getmask(nst_pic_ez), np.zeros((250,180,360)))
nst_pic_utz_smoothed = np.ma.masked_where(np.ma.getmask(nst_pic_utz), np.zeros((250,180,360)))

d15n_no3_pic_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_pic_ez), np.zeros((250,180,360)))
d15n_no3_pic_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_pic_utz), np.zeros((250,180,360)))

d15n_pom_pic_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_pic_ez),np.zeros((250,180,360)))
d15n_pom_pic_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_pic_utz), np.zeros((250,180,360)))


no3_picndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(no3_picndep_ez), np.zeros((250,180,360)))
no3_picndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(no3_picndep_utz), np.zeros((250,180,360)))

nst_picndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(nst_picndep_ez), np.zeros((250,180,360)))
nst_picndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(nst_picndep_utz), np.zeros((250,180,360)))

d15n_no3_picndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_picndep_ez), np.zeros((250,180,360)))
d15n_no3_picndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_picndep_utz), np.zeros((250,180,360)))

d15n_pom_picndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_picndep_ez),np.zeros((250,180,360)))
d15n_pom_picndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_picndep_utz), np.zeros((250,180,360)))


temp_fut_ez_smoothed = np.ma.masked_where(np.ma.getmask(temp_fut_ez), np.zeros((250,180,360)))
temp_fut_utz_smoothed = np.ma.masked_where(np.ma.getmask(temp_fut_utz), np.zeros((250,180,360)))

no3_fut_ez_smoothed = np.ma.masked_where(np.ma.getmask(no3_fut_ez), np.zeros((250,180,360)))
no3_fut_utz_smoothed = np.ma.masked_where(np.ma.getmask(no3_fut_utz), np.zeros((250,180,360)))

nst_fut_ez_smoothed = np.ma.masked_where(np.ma.getmask(nst_fut_ez), np.zeros((250,180,360)))
nst_fut_utz_smoothed = np.ma.masked_where(np.ma.getmask(nst_fut_utz), np.zeros((250,180,360)))

d15n_no3_fut_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_fut_ez), np.zeros((250,180,360)))
d15n_no3_fut_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_fut_utz), np.zeros((250,180,360)))

d15n_pom_fut_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_fut_ez),np.zeros((250,180,360)))
d15n_pom_fut_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_fut_utz), np.zeros((250,180,360)))


no3_futndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(no3_futndep_ez), np.zeros((250,180,360)))
no3_futndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(no3_futndep_utz), np.zeros((250,180,360)))

nst_futndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(nst_futndep_ez), np.zeros((250,180,360)))
nst_futndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(nst_futndep_utz), np.zeros((250,180,360)))

d15n_no3_futndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_futndep_ez), np.zeros((250,180,360)))
d15n_no3_futndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_no3_futndep_utz), np.zeros((250,180,360)))

d15n_pom_futndep_ez_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_futndep_ez),np.zeros((250,180,360)))
d15n_pom_futndep_utz_smoothed = np.ma.masked_where(np.ma.getmask(d15n_pom_futndep_utz), np.zeros((250,180,360)))



for i in tqdm(np.arange(len(no3_pic_ez[0,0,:])), desc="longitudes", position=1):
    for j in np.arange(len(no3_pic_ez[0,:,0])):
        if np.ma.is_mask(fix_pic[0,j,i]) == False:
            fix_pic_smoothed[:,j,i] = smooth(fix_pic_normalised[:,j,i])[0:250]
            fix_picndep_smoothed[:,j,i] = smooth(fix_picndep_normalised[:,j,i])[0:250]
            fix_fut_smoothed[:,j,i] = smooth(fix_fut_normalised[:,j,i])[0:250]
            fix_futndep_smoothed[:,j,i] = smooth(fix_futndep_normalised[:,j,i])[0:250]
        
        if np.ma.is_mask(npp_pic[0,j,i]) == False:
            npp_pic_smoothed[:,j,i] = smooth(npp_pic_normalised[:,j,i])[0:250]
            npp_picndep_smoothed[:,j,i] = smooth(npp_picndep_normalised[:,j,i])[0:250]
            npp_fut_smoothed[:,j,i] = smooth(npp_fut_normalised[:,j,i])[0:250]
            npp_futndep_smoothed[:,j,i] = smooth(npp_futndep_normalised[:,j,i])[0:250]
        
        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            temp_pic_ez_smoothed[:,j,i] = smooth(temp_pic_ez_normalised[:,j,i])[0:250]
            temp_fut_ez_smoothed[:,j,i] = smooth(temp_fut_ez_normalised[:,j,i])[0:250]
        if np.ma.is_mask(temp_pic_utz[0,j,i]) == False:
            temp_pic_utz_smoothed[:,j,i] = smooth(temp_pic_utz_normalised[:,j,i])[0:250]
            temp_fut_utz_smoothed[:,j,i] = smooth(temp_fut_utz_normalised[:,j,i])[0:250]
               
        if np.ma.is_mask(no3_pic_ez[0,j,i]) == False:
            no3_pic_ez_smoothed[:,j,i] = smooth(no3_pic_ez_normalised[:,j,i])[0:250]
            no3_picndep_ez_smoothed[:,j,i] = smooth(no3_picndep_ez_normalised[:,j,i])[0:250]
            no3_fut_ez_smoothed[:,j,i] = smooth(no3_fut_ez_normalised[:,j,i])[0:250]
            no3_futndep_ez_smoothed[:,j,i] = smooth(no3_futndep_ez_normalised[:,j,i])[0:250]
        if np.ma.is_mask(no3_pic_utz[0,j,i]) == False:
            no3_pic_utz_smoothed[:,j,i] = smooth(no3_pic_utz_normalised[:,j,i])[0:250]
            no3_picndep_utz_smoothed[:,j,i] = smooth(no3_picndep_utz_normalised[:,j,i])[0:250]
            no3_fut_utz_smoothed[:,j,i] = smooth(no3_fut_utz_normalised[:,j,i])[0:250]
            no3_futndep_utz_smoothed[:,j,i] = smooth(no3_futndep_utz_normalised[:,j,i])[0:250]
                
        if np.ma.is_mask(nst_pic_ez[0,j,i]) == False:
            nst_pic_ez_smoothed[:,j,i] = smooth(nst_pic_ez_normalised[:,j,i])[0:250]
            nst_picndep_ez_smoothed[:,j,i] = smooth(nst_picndep_ez_normalised[:,j,i])[0:250]
            nst_fut_ez_smoothed[:,j,i] = smooth(nst_fut_ez_normalised[:,j,i])[0:250]
            nst_futndep_ez_smoothed[:,j,i] = smooth(nst_futndep_ez_normalised[:,j,i])[0:250]
        if np.ma.is_mask(nst_pic_utz[0,j,i]) == False:
            nst_pic_utz_smoothed[:,j,i] = smooth(nst_pic_utz_normalised[:,j,i])[0:250]
            nst_picndep_utz_smoothed[:,j,i] = smooth(nst_picndep_utz_normalised[:,j,i])[0:250]
            nst_fut_utz_smoothed[:,j,i] = smooth(nst_fut_utz_normalised[:,j,i])[0:250]
            nst_futndep_utz_smoothed[:,j,i] = smooth(nst_futndep_utz_normalised[:,j,i])[0:250]
      
        if np.ma.is_mask(d15n_no3_pic_ez[0,j,i]) == False:
            d15n_no3_pic_ez_smoothed[:,j,i] = smooth(d15n_no3_pic_ez_normalised[:,j,i])[0:250]
            d15n_no3_picndep_ez_smoothed[:,j,i] = smooth(d15n_no3_picndep_ez_normalised[:,j,i])[0:250]
            d15n_no3_fut_ez_smoothed[:,j,i] = smooth(d15n_no3_fut_ez_normalised[:,j,i])[0:250]
            d15n_no3_futndep_ez_smoothed[:,j,i] = smooth(d15n_no3_futndep_ez_normalised[:,j,i])[0:250]
        if np.ma.is_mask(d15n_no3_pic_utz[0,j,i]) == False:
            d15n_no3_pic_utz_smoothed[:,j,i] = smooth(d15n_no3_pic_utz_normalised[:,j,i])[0:250]
            d15n_no3_picndep_utz_smoothed[:,j,i] = smooth(d15n_no3_picndep_utz_normalised[:,j,i])[0:250]
            d15n_no3_fut_utz_smoothed[:,j,i] = smooth(d15n_no3_fut_utz_normalised[:,j,i])[0:250]
            d15n_no3_futndep_utz_smoothed[:,j,i] = smooth(d15n_no3_futndep_utz_normalised[:,j,i])[0:250]
               
        if np.ma.is_mask(d15n_pom_pic_ez[0,j,i]) == False:
            d15n_pom_pic_ez_smoothed[:,j,i] = smooth(d15n_pom_pic_ez_normalised[:,j,i])[0:250]
            d15n_pom_picndep_ez_smoothed[:,j,i] = smooth(d15n_pom_picndep_ez_normalised[:,j,i])[0:250]
            d15n_pom_fut_ez_smoothed[:,j,i] = smooth(d15n_pom_fut_ez_normalised[:,j,i])[0:250]
            d15n_pom_futndep_ez_smoothed[:,j,i] = smooth(d15n_pom_futndep_ez_normalised[:,j,i])[0:250]
        if np.ma.is_mask(d15n_pom_pic_utz[0,j,i]) == False:
            d15n_pom_pic_utz_smoothed[:,j,i] = smooth(d15n_pom_pic_utz_normalised[:,j,i])[0:250]
            d15n_pom_picndep_utz_smoothed[:,j,i] = smooth(d15n_pom_picndep_utz_normalised[:,j,i])[0:250]
            d15n_pom_fut_utz_smoothed[:,j,i] = smooth(d15n_pom_fut_utz_normalised[:,j,i])[0:250]
            d15n_pom_futndep_utz_smoothed[:,j,i] = smooth(d15n_pom_futndep_utz_normalised[:,j,i])[0:250]

print("Smoothing to find trends complete")


#%% check the outcome

lab = ['Nat', 'Nat + N dep', 'HistRCP8.5', 'HistRCP8.5 + N dep']

alf=0.7
i = 160; j = 110
print(lon[i], lat[j])

fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(2,3,1)
plt.title('Raw timeseries')
plt.plot(time,d15n_no3_pic_ez[:,j,i], color='k',alpha=alf, label=lab[0])
plt.plot(time,d15n_no3_picndep_ez[:,j,i], color='royalblue',alpha=alf, label=lab[1])
plt.plot(time,d15n_no3_fut_ez[:,j,i], color='goldenrod',alpha=alf, label=lab[2])
plt.plot(time,d15n_no3_futndep_ez[:,j,i], color='firebrick',alpha=alf, label=lab[3])
plt.ylabel('$\delta^{15}$N$_{NO_3}$')
plt.legend(frameon=False, loc='upper center', ncol=4, bbox_to_anchor=(1.8,1.2))
plt.subplot(2,3,2)
plt.title('Detrended')
plt.plot(time,d15n_no3_pic_ez_detrended[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_ez_detrended[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_ez_detrended[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_ez_detrended[:,j,i], color='firebrick',alpha=alf)
plt.subplot(2,3,3)
plt.title('Normalised and smoothed')
plt.plot(time,d15n_no3_pic_ez_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_ez_smoothed[:,j,i], color='k',linewidth=2)
plt.plot(time,d15n_no3_picndep_ez_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_ez_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_ez_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_ez_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_ez_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_ez_smoothed[:,j,i], color='firebrick', linewidth=2)

ax4 = plt.subplot(2,3,4)
plt.plot(time,d15n_no3_pic_utz[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_utz[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz[:,j,i], color='firebrick',alpha=alf)
plt.ylabel('$\delta^{15}$N$_{NO_3}$')
plt.xlabel('Year')
plt.subplot(2,3,5)
plt.plot(time,d15n_no3_pic_utz_detrended[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz_detrended[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_utz_detrended[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz_detrended[:,j,i], color='firebrick',alpha=alf)
plt.xlabel('Year')
plt.subplot(2,3,6)
plt.plot(time,d15n_no3_pic_utz_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_utz_smoothed[:,j,i], color='k', linewidth=2)
plt.plot(time,d15n_no3_picndep_utz_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_utz_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_utz_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_utz_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz_smoothed[:,j,i], color='firebrick', linewidth=2)
plt.xlabel('Year')

xx=-0.35;yy=0.5
plt.text(xx,yy,'Euphotic\nZone', ha='center', va='center', transform=ax1.transAxes)
plt.text(xx,yy,'Twilight\nZone', ha='center', va='center', transform=ax4.transAxes)


#savefig
fig.savefig("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//figures//processToE-detrend_normalise.png", dpi=300, bbox_to_inches='tight')

#%% calculate the differences in the normalised, smoothed trends to return the anthropogenic signal

fix_picndep_signal = fix_picndep_smoothed - fix_pic_smoothed
fix_fut_signal = fix_fut_smoothed - fix_pic_smoothed
fix_futndep_signal = fix_futndep_smoothed - fix_pic_smoothed

npp_picndep_signal = npp_picndep_smoothed - npp_pic_smoothed
npp_fut_signal = npp_fut_smoothed - npp_pic_smoothed
npp_futndep_signal = npp_futndep_smoothed - npp_pic_smoothed


temp_fut_signal_ez = temp_fut_ez_smoothed - temp_pic_ez_smoothed
temp_fut_signal_utz = temp_fut_utz_smoothed - temp_pic_utz_smoothed


no3_picndep_signal_ez = no3_picndep_ez_smoothed - no3_pic_ez_smoothed
no3_picndep_signal_utz = no3_picndep_utz_smoothed - no3_pic_utz_smoothed

nst_picndep_signal_ez = nst_picndep_ez_smoothed - nst_pic_ez_smoothed
nst_picndep_signal_utz = nst_picndep_utz_smoothed - nst_pic_utz_smoothed

d15n_no3_picndep_signal_ez = d15n_no3_picndep_ez_smoothed - d15n_no3_pic_ez_smoothed
d15n_no3_picndep_signal_utz = d15n_no3_picndep_utz_smoothed - d15n_no3_pic_utz_smoothed

d15n_pom_picndep_signal_ez = d15n_pom_picndep_ez_smoothed - d15n_pom_pic_ez_smoothed
d15n_pom_picndep_signal_utz = d15n_pom_picndep_utz_smoothed - d15n_pom_pic_utz_smoothed


no3_fut_signal_ez = no3_fut_ez_smoothed - no3_pic_ez_smoothed
no3_fut_signal_utz = no3_fut_utz_smoothed - no3_pic_utz_smoothed

nst_fut_signal_ez = nst_fut_ez_smoothed - nst_pic_ez_smoothed
nst_fut_signal_utz = nst_fut_utz_smoothed - nst_pic_utz_smoothed

d15n_no3_fut_signal_ez = d15n_no3_fut_ez_smoothed - d15n_no3_pic_ez_smoothed
d15n_no3_fut_signal_utz = d15n_no3_fut_utz_smoothed - d15n_no3_pic_utz_smoothed

d15n_pom_fut_signal_ez = d15n_pom_fut_ez_smoothed - d15n_pom_pic_ez_smoothed
d15n_pom_fut_signal_utz = d15n_pom_fut_utz_smoothed - d15n_pom_pic_utz_smoothed


no3_futndep_signal_ez = no3_futndep_ez_smoothed - no3_pic_ez_smoothed
no3_futndep_signal_utz = no3_futndep_utz_smoothed - no3_pic_utz_smoothed

nst_futndep_signal_ez = nst_futndep_ez_smoothed - nst_pic_ez_smoothed
nst_futndep_signal_utz = nst_futndep_utz_smoothed - nst_pic_utz_smoothed

d15n_no3_futndep_signal_ez = d15n_no3_futndep_ez_smoothed - d15n_no3_pic_ez_smoothed
d15n_no3_futndep_signal_utz = d15n_no3_futndep_utz_smoothed - d15n_no3_pic_utz_smoothed

d15n_pom_futndep_signal_ez = d15n_pom_futndep_ez_smoothed - d15n_pom_pic_ez_smoothed
d15n_pom_futndep_signal_utz = d15n_pom_futndep_utz_smoothed - d15n_pom_pic_utz_smoothed

print("Differences in trends complete")


#%% calucalte the metrics used (std and range) to quantify the noise

# calculate std
fix_std = np.ma.std(fix_pic_normalised,axis=0)
npp_std = np.ma.std(npp_pic_normalised,axis=0)

temp_ez_std = np.ma.std(temp_pic_ez_normalised,axis=0)
temp_utz_std = np.ma.std(temp_pic_utz_normalised,axis=0)

no3_ez_std = np.ma.std(no3_pic_ez_normalised,axis=0)
no3_utz_std = np.ma.std(no3_pic_utz_normalised,axis=0)

nst_ez_std = np.ma.std(nst_pic_ez_normalised,axis=0)
nst_utz_std = np.ma.std(nst_pic_utz_normalised,axis=0)

d15n_no3_ez_std = np.ma.std(d15n_no3_pic_ez_normalised,axis=0)
d15n_no3_utz_std = np.ma.std(d15n_no3_pic_utz_normalised,axis=0)

d15n_pom_ez_std = np.ma.std(d15n_pom_pic_ez_normalised,axis=0)
d15n_pom_utz_std = np.ma.std(d15n_pom_pic_utz_normalised,axis=0)

# calculate range
fix_range = ( np.ma.max(fix_pic_normalised,axis=0) - np.ma.min(fix_pic_normalised,axis=0) ) * 0.5
npp_range = ( np.ma.max(npp_pic_normalised,axis=0) - np.ma.min(npp_pic_normalised,axis=0) ) * 0.5

temp_ez_range = ( np.ma.max(temp_pic_ez_normalised,axis=0) - np.ma.min(temp_pic_ez_normalised,axis=0) ) * 0.5
temp_utz_range = ( np.ma.max(temp_pic_utz_normalised,axis=0) - np.ma.min(temp_pic_utz_normalised,axis=0) ) * 0.5

no3_ez_range = ( np.ma.max(no3_pic_ez_normalised,axis=0) - np.ma.min(no3_pic_ez_normalised,axis=0) ) * 0.5
no3_utz_range = ( np.ma.max(no3_pic_utz_normalised,axis=0) - np.ma.min(no3_pic_utz_normalised,axis=0) ) * 0.5

nst_ez_range = ( np.ma.max(nst_pic_ez_normalised,axis=0) - np.ma.min(nst_pic_ez_normalised,axis=0) ) * 0.5
nst_utz_range = ( np.ma.max(nst_pic_utz_normalised,axis=0) - np.ma.min(nst_pic_utz_normalised,axis=0) ) * 0.5

d15n_no3_ez_range = ( np.ma.max(d15n_no3_pic_ez_normalised,axis=0) - np.ma.min(d15n_no3_pic_ez_normalised,axis=0) ) * 0.5
d15n_no3_utz_range = ( np.ma.max(d15n_no3_pic_utz_normalised,axis=0) - np.ma.min(d15n_no3_pic_utz_normalised,axis=0) ) * 0.5

d15n_pom_ez_range = ( np.ma.max(d15n_pom_pic_ez_normalised,axis=0) - np.ma.min(d15n_pom_pic_ez_normalised,axis=0) ) * 0.5
d15n_pom_utz_range = ( np.ma.max(d15n_pom_pic_utz_normalised,axis=0) - np.ma.min(d15n_pom_pic_utz_normalised,axis=0) ) * 0.5

print("Noise estimates complete")


#%%

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.contourf(d15n_no3_ez_std*2 / d15n_no3_ez_range, levels=np.arange(0.5,1.51,0.05), cmap=cmo.balance, extend='both')
plt.subplot(1,2,2)
plt.contourf(d15n_no3_utz_std*2 / d15n_no3_utz_range, levels=np.arange(0.5,1.51,0.05), cmap=cmo.balance, extend='both')


#%% set values equal to 1 when the signal > noise, 0 when signal < 0

fix_picndep_detect_std1 = np.zeros((250,180,360))
fix_picndep_detect_std2 = np.zeros((250,180,360))
fix_picndep_detect_rang = np.zeros((250,180,360))
fix_fut_detect_std1 = np.zeros((250,180,360))
fix_fut_detect_std2 = np.zeros((250,180,360))
fix_fut_detect_rang = np.zeros((250,180,360))
fix_futndep_detect_std1 = np.zeros((250,180,360))
fix_futndep_detect_std2 = np.zeros((250,180,360))
fix_futndep_detect_rang = np.zeros((250,180,360))

npp_picndep_detect_std1 = np.zeros((250,180,360))
npp_picndep_detect_std2 = np.zeros((250,180,360))
npp_picndep_detect_rang = np.zeros((250,180,360))
npp_fut_detect_std1 = np.zeros((250,180,360))
npp_fut_detect_std2 = np.zeros((250,180,360))
npp_fut_detect_rang = np.zeros((250,180,360))
npp_futndep_detect_std1 = np.zeros((250,180,360))
npp_futndep_detect_std2 = np.zeros((250,180,360))
npp_futndep_detect_rang = np.zeros((250,180,360))


temp_fut_detect_std1_ez = np.zeros((250,180,360))
temp_fut_detect_std2_ez = np.zeros((250,180,360))
temp_fut_detect_rang_ez = np.zeros((250,180,360))
temp_fut_detect_std1_utz = np.zeros((250,180,360))
temp_fut_detect_std2_utz = np.zeros((250,180,360))
temp_fut_detect_rang_utz = np.zeros((250,180,360))


no3_picndep_detect_std1_ez = np.zeros((250,180,360))
no3_picndep_detect_std2_ez = np.zeros((250,180,360))
no3_picndep_detect_rang_ez = np.zeros((250,180,360))
no3_picndep_detect_std1_utz = np.zeros((250,180,360))
no3_picndep_detect_std2_utz = np.zeros((250,180,360))
no3_picndep_detect_rang_utz = np.zeros((250,180,360))

nst_picndep_detect_std1_ez = np.zeros((250,180,360))
nst_picndep_detect_std2_ez = np.zeros((250,180,360))
nst_picndep_detect_rang_ez = np.zeros((250,180,360))
nst_picndep_detect_std1_utz = np.zeros((250,180,360))
nst_picndep_detect_std2_utz = np.zeros((250,180,360))
nst_picndep_detect_rang_utz = np.zeros((250,180,360))

d15n_no3_picndep_detect_std1_ez = np.zeros((250,180,360))
d15n_no3_picndep_detect_std2_ez = np.zeros((250,180,360))
d15n_no3_picndep_detect_rang_ez = np.zeros((250,180,360))
d15n_no3_picndep_detect_std1_utz = np.zeros((250,180,360))
d15n_no3_picndep_detect_std2_utz = np.zeros((250,180,360))
d15n_no3_picndep_detect_rang_utz = np.zeros((250,180,360))

d15n_pom_picndep_detect_std1_ez = np.zeros((250,180,360))
d15n_pom_picndep_detect_std2_ez = np.zeros((250,180,360))
d15n_pom_picndep_detect_rang_ez = np.zeros((250,180,360))
d15n_pom_picndep_detect_std1_utz = np.zeros((250,180,360))
d15n_pom_picndep_detect_std2_utz = np.zeros((250,180,360))
d15n_pom_picndep_detect_rang_utz = np.zeros((250,180,360))


no3_fut_detect_std1_ez = np.zeros((250,180,360))
no3_fut_detect_std2_ez = np.zeros((250,180,360))
no3_fut_detect_rang_ez = np.zeros((250,180,360))
no3_fut_detect_std1_utz = np.zeros((250,180,360))
no3_fut_detect_std2_utz = np.zeros((250,180,360))
no3_fut_detect_rang_utz = np.zeros((250,180,360))

nst_fut_detect_std1_ez = np.zeros((250,180,360))
nst_fut_detect_std2_ez = np.zeros((250,180,360))
nst_fut_detect_rang_ez = np.zeros((250,180,360))
nst_fut_detect_std1_utz = np.zeros((250,180,360))
nst_fut_detect_std2_utz = np.zeros((250,180,360))
nst_fut_detect_rang_utz = np.zeros((250,180,360))

d15n_no3_fut_detect_std1_ez = np.zeros((250,180,360))
d15n_no3_fut_detect_std2_ez = np.zeros((250,180,360))
d15n_no3_fut_detect_rang_ez = np.zeros((250,180,360))
d15n_no3_fut_detect_std1_utz = np.zeros((250,180,360))
d15n_no3_fut_detect_std2_utz = np.zeros((250,180,360))
d15n_no3_fut_detect_rang_utz = np.zeros((250,180,360))

d15n_pom_fut_detect_std1_ez = np.zeros((250,180,360))
d15n_pom_fut_detect_std2_ez = np.zeros((250,180,360))
d15n_pom_fut_detect_rang_ez = np.zeros((250,180,360))
d15n_pom_fut_detect_std1_utz = np.zeros((250,180,360))
d15n_pom_fut_detect_std2_utz = np.zeros((250,180,360))
d15n_pom_fut_detect_rang_utz = np.zeros((250,180,360))


no3_futndep_detect_std1_ez = np.zeros((250,180,360))
no3_futndep_detect_std2_ez = np.zeros((250,180,360))
no3_futndep_detect_rang_ez = np.zeros((250,180,360))
no3_futndep_detect_std1_utz = np.zeros((250,180,360))
no3_futndep_detect_std2_utz = np.zeros((250,180,360))
no3_futndep_detect_rang_utz = np.zeros((250,180,360))

nst_futndep_detect_std1_ez = np.zeros((250,180,360))
nst_futndep_detect_std2_ez = np.zeros((250,180,360))
nst_futndep_detect_rang_ez = np.zeros((250,180,360))
nst_futndep_detect_std1_utz = np.zeros((250,180,360))
nst_futndep_detect_std2_utz = np.zeros((250,180,360))
nst_futndep_detect_rang_utz = np.zeros((250,180,360))

d15n_no3_futndep_detect_std1_ez = np.zeros((250,180,360))
d15n_no3_futndep_detect_std2_ez = np.zeros((250,180,360))
d15n_no3_futndep_detect_rang_ez = np.zeros((250,180,360))
d15n_no3_futndep_detect_std1_utz = np.zeros((250,180,360))
d15n_no3_futndep_detect_std2_utz = np.zeros((250,180,360))
d15n_no3_futndep_detect_rang_utz = np.zeros((250,180,360))

d15n_pom_futndep_detect_std1_ez = np.zeros((250,180,360))
d15n_pom_futndep_detect_std2_ez = np.zeros((250,180,360))
d15n_pom_futndep_detect_rang_ez = np.zeros((250,180,360))
d15n_pom_futndep_detect_std1_utz = np.zeros((250,180,360))
d15n_pom_futndep_detect_std2_utz = np.zeros((250,180,360))
d15n_pom_futndep_detect_rang_utz = np.zeros((250,180,360))


for i in tqdm(np.arange(len(no3_pic_ez[0,0,:])), desc="longitudes", position=1):
    for j in np.arange(len(no3_pic_ez[0,:,0])):

        if np.ma.is_mask(fix_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(fix_picndep_signal[:,j,i]) > fix_std[j,i]] = 1
            tmp2[np.abs(fix_picndep_signal[:,j,i]) > fix_std[j,i]*2] = 1
            tmp3[np.abs(fix_picndep_signal[:,j,i]) > fix_range[j,i]] = 1
            fix_picndep_detect_std1[:,j,i] = tmp1
            fix_picndep_detect_std2[:,j,i] = tmp2
            fix_picndep_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(fix_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(fix_fut_signal[:,j,i]) > fix_std[j,i]] = 1
            tmp2[np.abs(fix_fut_signal[:,j,i]) > fix_std[j,i]*2] = 1
            tmp3[np.abs(fix_fut_signal[:,j,i]) > fix_range[j,i]] = 1
            fix_fut_detect_std1[:,j,i] = tmp1
            fix_fut_detect_std2[:,j,i] = tmp2
            fix_fut_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(fix_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(fix_futndep_signal[:,j,i]) > fix_std[j,i]] = 1
            tmp2[np.abs(fix_futndep_signal[:,j,i]) > fix_std[j,i]*2] = 1
            tmp3[np.abs(fix_futndep_signal[:,j,i]) > fix_range[j,i]] = 1
            fix_futndep_detect_std1[:,j,i] = tmp1
            fix_futndep_detect_std2[:,j,i] = tmp2
            fix_futndep_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(npp_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(npp_picndep_signal[:,j,i]) > npp_std[j,i]] = 1
            tmp2[np.abs(npp_picndep_signal[:,j,i]) > npp_std[j,i]*2] = 1
            tmp3[np.abs(npp_picndep_signal[:,j,i]) > npp_range[j,i]] = 1
            npp_picndep_detect_std1[:,j,i] = tmp1
            npp_picndep_detect_std2[:,j,i] = tmp2
            npp_picndep_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(npp_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(npp_fut_signal[:,j,i]) > npp_std[j,i]] = 1
            tmp2[np.abs(npp_fut_signal[:,j,i]) > npp_std[j,i]*2] = 1
            tmp3[np.abs(npp_fut_signal[:,j,i]) > npp_range[j,i]] = 1
            npp_fut_detect_std1[:,j,i] = tmp1
            npp_fut_detect_std2[:,j,i] = tmp2
            npp_fut_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(npp_pic[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(npp_futndep_signal[:,j,i]) > npp_std[j,i]] = 1
            tmp2[np.abs(npp_futndep_signal[:,j,i]) > npp_std[j,i]*2] = 1
            tmp3[np.abs(npp_futndep_signal[:,j,i]) > npp_range[j,i]] = 1
            npp_futndep_detect_std1[:,j,i] = tmp1
            npp_futndep_detect_std2[:,j,i] = tmp2
            npp_futndep_detect_rang[:,j,i] = tmp3
        
        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(temp_fut_signal_ez[:,j,i]) > temp_ez_std[j,i]] = 1
            tmp2[np.abs(temp_fut_signal_ez[:,j,i]) > temp_ez_std[j,i]*2] = 1
            tmp3[np.abs(temp_fut_signal_ez[:,j,i]) > temp_ez_range[j,i]] = 1
            temp_fut_detect_std1_ez[:,j,i] = tmp1
            temp_fut_detect_std2_ez[:,j,i] = tmp2
            temp_fut_detect_rang_ez[:,j,i] = tmp3
        
        if np.ma.is_mask(temp_pic_utz[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(temp_fut_signal_utz[:,j,i]) > temp_utz_std[j,i]] = 1
            tmp2[np.abs(temp_fut_signal_utz[:,j,i]) > temp_utz_std[j,i]*2] = 1
            tmp3[np.abs(temp_fut_signal_utz[:,j,i]) > temp_utz_range[j,i]] = 1
            temp_fut_detect_std1_utz[:,j,i] = tmp1
            temp_fut_detect_std2_utz[:,j,i] = tmp2
            temp_fut_detect_rang_utz[:,j,i] = tmp3
        
        if np.ma.is_mask(no3_pic_ez[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_picndep_signal_ez[:,j,i]) > no3_ez_std[j,i]] = 1
            tmp2[np.abs(no3_picndep_signal_ez[:,j,i]) > no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(no3_picndep_signal_ez[:,j,i]) > no3_ez_range[j,i]] = 1
            no3_picndep_detect_std1_ez[:,j,i] = tmp1
            no3_picndep_detect_std2_ez[:,j,i] = tmp2
            no3_picndep_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_fut_signal_ez[:,j,i]) > no3_ez_std[j,i]] = 1
            tmp2[np.abs(no3_fut_signal_ez[:,j,i]) > no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(no3_fut_signal_ez[:,j,i]) > no3_ez_range[j,i]] = 1
            no3_fut_detect_std1_ez[:,j,i] = tmp1
            no3_fut_detect_std2_ez[:,j,i] = tmp2
            no3_fut_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_futndep_signal_ez[:,j,i]) > no3_ez_std[j,i]] = 1
            tmp2[np.abs(no3_futndep_signal_ez[:,j,i]) > no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(no3_futndep_signal_ez[:,j,i]) > no3_ez_range[j,i]] = 1
            no3_futndep_detect_std1_ez[:,j,i] = tmp1
            no3_futndep_detect_std2_ez[:,j,i] = tmp2
            no3_futndep_detect_rang_ez[:,j,i] = tmp3
        
        if np.ma.is_mask(no3_pic_utz[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_picndep_signal_utz[:,j,i]) > no3_utz_std[j,i]] = 1
            tmp2[np.abs(no3_picndep_signal_utz[:,j,i]) > no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(no3_picndep_signal_utz[:,j,i]) > no3_utz_range[j,i]] = 1
            no3_picndep_detect_std1_utz[:,j,i] = tmp1
            no3_picndep_detect_std2_utz[:,j,i] = tmp2
            no3_picndep_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_fut_signal_utz[:,j,i]) > no3_utz_std[j,i]] = 1
            tmp2[np.abs(no3_fut_signal_utz[:,j,i]) > no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(no3_fut_signal_utz[:,j,i]) > no3_utz_range[j,i]] = 1
            no3_fut_detect_std1_utz[:,j,i] = tmp1
            no3_fut_detect_std2_utz[:,j,i] = tmp2
            no3_fut_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(no3_futndep_signal_utz[:,j,i]) > no3_utz_std[j,i]] = 1
            tmp2[np.abs(no3_futndep_signal_utz[:,j,i]) > no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(no3_futndep_signal_utz[:,j,i]) > no3_utz_range[j,i]] = 1
            no3_futndep_detect_std1_utz[:,j,i] = tmp1
            no3_futndep_detect_std2_utz[:,j,i] = tmp2
            no3_futndep_detect_rang_utz[:,j,i] = tmp3
        
        
        if np.ma.is_mask(nst_pic_ez[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_picndep_signal_ez[:,j,i]) > nst_ez_std[j,i]] = 1
            tmp2[np.abs(nst_picndep_signal_ez[:,j,i]) > nst_ez_std[j,i]*2] = 1
            tmp3[np.abs(nst_picndep_signal_ez[:,j,i]) > nst_ez_range[j,i]] = 1
            nst_picndep_detect_std1_ez[:,j,i] = tmp1
            nst_picndep_detect_std2_ez[:,j,i] = tmp2
            nst_picndep_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_fut_signal_ez[:,j,i]) > nst_ez_std[j,i]] = 1
            tmp2[np.abs(nst_fut_signal_ez[:,j,i]) > nst_ez_std[j,i]*2] = 1
            tmp3[np.abs(nst_fut_signal_ez[:,j,i]) > nst_ez_range[j,i]] = 1
            nst_fut_detect_std1_ez[:,j,i] = tmp1
            nst_fut_detect_std2_ez[:,j,i] = tmp2
            nst_fut_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_futndep_signal_ez[:,j,i]) > nst_ez_std[j,i]] = 1
            tmp2[np.abs(nst_futndep_signal_ez[:,j,i]) > nst_ez_std[j,i]*2] = 1
            tmp3[np.abs(nst_futndep_signal_ez[:,j,i]) > nst_ez_range[j,i]] = 1
            nst_futndep_detect_std1_ez[:,j,i] = tmp1
            nst_futndep_detect_std2_ez[:,j,i] = tmp2
            nst_futndep_detect_rang_ez[:,j,i] = tmp3
        
        if np.ma.is_mask(nst_pic_utz[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_picndep_signal_utz[:,j,i]) > nst_utz_std[j,i]] = 1
            tmp2[np.abs(nst_picndep_signal_utz[:,j,i]) > nst_utz_std[j,i]*2] = 1
            tmp3[np.abs(nst_picndep_signal_utz[:,j,i]) > nst_utz_range[j,i]] = 1
            nst_picndep_detect_std1_utz[:,j,i] = tmp1
            nst_picndep_detect_std2_utz[:,j,i] = tmp2
            nst_picndep_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_fut_signal_utz[:,j,i]) > nst_utz_std[j,i]] = 1
            tmp2[np.abs(nst_fut_signal_utz[:,j,i]) > nst_utz_std[j,i]*2] = 1
            tmp3[np.abs(nst_fut_signal_utz[:,j,i]) > nst_utz_range[j,i]] = 1
            nst_fut_detect_std1_utz[:,j,i] = tmp1
            nst_fut_detect_std2_utz[:,j,i] = tmp2
            nst_fut_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(nst_futndep_signal_utz[:,j,i]) > nst_utz_std[j,i]] = 1
            tmp2[np.abs(nst_futndep_signal_utz[:,j,i]) > nst_utz_std[j,i]*2] = 1
            tmp3[np.abs(nst_futndep_signal_utz[:,j,i]) > nst_utz_range[j,i]] = 1
            nst_futndep_detect_std1_utz[:,j,i] = tmp1
            nst_futndep_detect_std2_utz[:,j,i] = tmp2
            nst_futndep_detect_rang_utz[:,j,i] = tmp3
        
        
        if np.ma.is_mask(d15n_no3_pic_ez[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_picndep_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_picndep_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_picndep_signal_ez[:,j,i]) > d15n_no3_ez_range[j,i]] = 1
            d15n_no3_picndep_detect_std1_ez[:,j,i] = tmp1
            d15n_no3_picndep_detect_std2_ez[:,j,i] = tmp2
            d15n_no3_picndep_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_fut_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_fut_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_fut_signal_ez[:,j,i]) > d15n_no3_ez_range[j,i]] = 1
            d15n_no3_fut_detect_std1_ez[:,j,i] = tmp1
            d15n_no3_fut_detect_std2_ez[:,j,i] = tmp2
            d15n_no3_fut_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_futndep_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_futndep_signal_ez[:,j,i]) > d15n_no3_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_futndep_signal_ez[:,j,i]) > d15n_no3_ez_range[j,i]] = 1
            d15n_no3_futndep_detect_std1_ez[:,j,i] = tmp1
            d15n_no3_futndep_detect_std2_ez[:,j,i] = tmp2
            d15n_no3_futndep_detect_rang_ez[:,j,i] = tmp3
        
        if np.ma.is_mask(d15n_no3_pic_utz[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_picndep_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_picndep_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_picndep_signal_utz[:,j,i]) > d15n_no3_utz_range[j,i]] = 1
            d15n_no3_picndep_detect_std1_utz[:,j,i] = tmp1
            d15n_no3_picndep_detect_std2_utz[:,j,i] = tmp2
            d15n_no3_picndep_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_fut_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_fut_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_fut_signal_utz[:,j,i]) > d15n_no3_utz_range[j,i]] = 1
            d15n_no3_fut_detect_std1_utz[:,j,i] = tmp1
            d15n_no3_fut_detect_std2_utz[:,j,i] = tmp2
            d15n_no3_fut_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_no3_futndep_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_no3_futndep_signal_utz[:,j,i]) > d15n_no3_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_no3_futndep_signal_utz[:,j,i]) > d15n_no3_utz_range[j,i]] = 1
            d15n_no3_futndep_detect_std1_utz[:,j,i] = tmp1
            d15n_no3_futndep_detect_std2_utz[:,j,i] = tmp2
            d15n_no3_futndep_detect_rang_utz[:,j,i] = tmp3
        
        
        if np.ma.is_mask(d15n_pom_pic_ez[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_picndep_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_picndep_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_picndep_signal_ez[:,j,i]) > d15n_pom_ez_range[j,i]] = 1
            d15n_pom_picndep_detect_std1_ez[:,j,i] = tmp1
            d15n_pom_picndep_detect_std2_ez[:,j,i] = tmp2
            d15n_pom_picndep_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_fut_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_fut_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_fut_signal_ez[:,j,i]) > d15n_pom_ez_range[j,i]] = 1
            d15n_pom_fut_detect_std1_ez[:,j,i] = tmp1
            d15n_pom_fut_detect_std2_ez[:,j,i] = tmp2
            d15n_pom_fut_detect_rang_ez[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_futndep_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_futndep_signal_ez[:,j,i]) > d15n_pom_ez_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_futndep_signal_ez[:,j,i]) > d15n_pom_ez_range[j,i]] = 1
            d15n_pom_futndep_detect_std1_ez[:,j,i] = tmp1
            d15n_pom_futndep_detect_std2_ez[:,j,i] = tmp2
            d15n_pom_futndep_detect_rang_ez[:,j,i] = tmp3
        
        if np.ma.is_mask(d15n_pom_pic_utz[0,j,i]) == False:
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_picndep_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_picndep_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_picndep_signal_utz[:,j,i]) > d15n_pom_utz_range[j,i]] = 1
            d15n_pom_picndep_detect_std1_utz[:,j,i] = tmp1
            d15n_pom_picndep_detect_std2_utz[:,j,i] = tmp2
            d15n_pom_picndep_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_fut_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_fut_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_fut_signal_utz[:,j,i]) > d15n_pom_utz_range[j,i]] = 1
            d15n_pom_fut_detect_std1_utz[:,j,i] = tmp1
            d15n_pom_fut_detect_std2_utz[:,j,i] = tmp2
            d15n_pom_fut_detect_rang_utz[:,j,i] = tmp3
            tmp1 = np.zeros((250)); tmp2 = np.zeros((250)); tmp3 = np.zeros((250));
            tmp1[np.abs(d15n_pom_futndep_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]] = 1
            tmp2[np.abs(d15n_pom_futndep_signal_utz[:,j,i]) > d15n_pom_utz_std[j,i]*2] = 1
            tmp3[np.abs(d15n_pom_futndep_signal_utz[:,j,i]) > d15n_pom_utz_range[j,i]] = 1
            d15n_pom_futndep_detect_std1_utz[:,j,i] = tmp1
            d15n_pom_futndep_detect_std2_utz[:,j,i] = tmp2
            d15n_pom_futndep_detect_rang_utz[:,j,i] = tmp3
        

print("Raw emergence of signal over noise complete")


#%% perform boxcar smoothing over 11 years to eliminate short-term fluctuations

fix_picndep_detectsmooth_std1 = np.zeros((250,180,360))
fix_picndep_detectsmooth_std2 = np.zeros((250,180,360))
fix_picndep_detectsmooth_rang = np.zeros((250,180,360))
fix_fut_detectsmooth_std1 = np.zeros((250,180,360))
fix_fut_detectsmooth_std2 = np.zeros((250,180,360))
fix_fut_detectsmooth_rang = np.zeros((250,180,360))
fix_futndep_detectsmooth_std1 = np.zeros((250,180,360))
fix_futndep_detectsmooth_std2 = np.zeros((250,180,360))
fix_futndep_detectsmooth_rang = np.zeros((250,180,360))

npp_picndep_detectsmooth_std1 = np.zeros((250,180,360))
npp_picndep_detectsmooth_std2 = np.zeros((250,180,360))
npp_picndep_detectsmooth_rang = np.zeros((250,180,360))
npp_fut_detectsmooth_std1 = np.zeros((250,180,360))
npp_fut_detectsmooth_std2 = np.zeros((250,180,360))
npp_fut_detectsmooth_rang = np.zeros((250,180,360))
npp_futndep_detectsmooth_std1 = np.zeros((250,180,360))
npp_futndep_detectsmooth_std2 = np.zeros((250,180,360))
npp_futndep_detectsmooth_rang = np.zeros((250,180,360))


temp_fut_detectsmooth_std1_ez = np.zeros((250,180,360))
temp_fut_detectsmooth_std2_ez = np.zeros((250,180,360))
temp_fut_detectsmooth_rang_ez = np.zeros((250,180,360))
temp_fut_detectsmooth_std1_utz = np.zeros((250,180,360))
temp_fut_detectsmooth_std2_utz = np.zeros((250,180,360))
temp_fut_detectsmooth_rang_utz = np.zeros((250,180,360))


no3_picndep_detectsmooth_std1_ez = np.zeros((250,180,360))
no3_picndep_detectsmooth_std2_ez = np.zeros((250,180,360))
no3_picndep_detectsmooth_rang_ez = np.zeros((250,180,360))
no3_picndep_detectsmooth_std1_utz = np.zeros((250,180,360))
no3_picndep_detectsmooth_std2_utz = np.zeros((250,180,360))
no3_picndep_detectsmooth_rang_utz = np.zeros((250,180,360))

nst_picndep_detectsmooth_std1_ez = np.zeros((250,180,360))
nst_picndep_detectsmooth_std2_ez = np.zeros((250,180,360))
nst_picndep_detectsmooth_rang_ez = np.zeros((250,180,360))
nst_picndep_detectsmooth_std1_utz = np.zeros((250,180,360))
nst_picndep_detectsmooth_std2_utz = np.zeros((250,180,360))
nst_picndep_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_no3_picndep_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_no3_picndep_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_no3_picndep_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_no3_picndep_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_no3_picndep_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_no3_picndep_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_pom_picndep_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_pom_picndep_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_pom_picndep_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_pom_picndep_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_pom_picndep_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_pom_picndep_detectsmooth_rang_utz = np.zeros((250,180,360))


no3_fut_detectsmooth_std1_ez = np.zeros((250,180,360))
no3_fut_detectsmooth_std2_ez = np.zeros((250,180,360))
no3_fut_detectsmooth_rang_ez = np.zeros((250,180,360))
no3_fut_detectsmooth_std1_utz = np.zeros((250,180,360))
no3_fut_detectsmooth_std2_utz = np.zeros((250,180,360))
no3_fut_detectsmooth_rang_utz = np.zeros((250,180,360))

nst_fut_detectsmooth_std1_ez = np.zeros((250,180,360))
nst_fut_detectsmooth_std2_ez = np.zeros((250,180,360))
nst_fut_detectsmooth_rang_ez = np.zeros((250,180,360))
nst_fut_detectsmooth_std1_utz = np.zeros((250,180,360))
nst_fut_detectsmooth_std2_utz = np.zeros((250,180,360))
nst_fut_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_no3_fut_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_no3_fut_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_no3_fut_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_no3_fut_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_no3_fut_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_no3_fut_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_pom_fut_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_pom_fut_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_pom_fut_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_pom_fut_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_pom_fut_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_pom_fut_detectsmooth_rang_utz = np.zeros((250,180,360))

no3_futndep_detectsmooth_std1_ez = np.zeros((250,180,360))
no3_futndep_detectsmooth_std2_ez = np.zeros((250,180,360))
no3_futndep_detectsmooth_rang_ez = np.zeros((250,180,360))
no3_futndep_detectsmooth_std1_utz = np.zeros((250,180,360))
no3_futndep_detectsmooth_std2_utz = np.zeros((250,180,360))
no3_futndep_detectsmooth_rang_utz = np.zeros((250,180,360))

nst_futndep_detectsmooth_std1_ez = np.zeros((250,180,360))
nst_futndep_detectsmooth_std2_ez = np.zeros((250,180,360))
nst_futndep_detectsmooth_rang_ez = np.zeros((250,180,360))
nst_futndep_detectsmooth_std1_utz = np.zeros((250,180,360))
nst_futndep_detectsmooth_std2_utz = np.zeros((250,180,360))
nst_futndep_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_no3_futndep_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_no3_futndep_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_no3_futndep_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_no3_futndep_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_no3_futndep_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_no3_futndep_detectsmooth_rang_utz = np.zeros((250,180,360))

d15n_pom_futndep_detectsmooth_std1_ez = np.zeros((250,180,360))
d15n_pom_futndep_detectsmooth_std2_ez = np.zeros((250,180,360))
d15n_pom_futndep_detectsmooth_rang_ez = np.zeros((250,180,360))
d15n_pom_futndep_detectsmooth_std1_utz = np.zeros((250,180,360))
d15n_pom_futndep_detectsmooth_std2_utz = np.zeros((250,180,360))
d15n_pom_futndep_detectsmooth_rang_utz = np.zeros((250,180,360))


for i in tqdm(np.arange(len(no3_pic_ez[0,0,:])), desc="longitudes", position=1):
    for j in np.arange(len(no3_pic_ez[0,:,0])):
        
        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            fix_picndep_detectsmooth_std1[:,j,i] = smooth(fix_picndep_detect_std1[:,j,i])[0:250]
            fix_picndep_detectsmooth_std2[:,j,i] = smooth(fix_picndep_detect_std2[:,j,i])[0:250]
            fix_picndep_detectsmooth_rang[:,j,i] = smooth(fix_picndep_detect_rang[:,j,i])[0:250]
            fix_fut_detectsmooth_std1[:,j,i] = smooth(fix_fut_detect_std1[:,j,i])[0:250]
            fix_fut_detectsmooth_std2[:,j,i] = smooth(fix_fut_detect_std2[:,j,i])[0:250]
            fix_fut_detectsmooth_rang[:,j,i] = smooth(fix_fut_detect_rang[:,j,i])[0:250]
            fix_futndep_detectsmooth_std1[:,j,i] = smooth(fix_futndep_detect_std1[:,j,i])[0:250]
            fix_futndep_detectsmooth_std2[:,j,i] = smooth(fix_futndep_detect_std2[:,j,i])[0:250]
            fix_futndep_detectsmooth_rang[:,j,i] = smooth(fix_futndep_detect_rang[:,j,i])[0:250]
            
        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            npp_picndep_detectsmooth_std1[:,j,i] = smooth(npp_picndep_detect_std1[:,j,i])[0:250]
            npp_picndep_detectsmooth_std2[:,j,i] = smooth(npp_picndep_detect_std2[:,j,i])[0:250]
            npp_picndep_detectsmooth_rang[:,j,i] = smooth(npp_picndep_detect_rang[:,j,i])[0:250]
            npp_fut_detectsmooth_std1[:,j,i] = smooth(npp_fut_detect_std1[:,j,i])[0:250]
            npp_fut_detectsmooth_std2[:,j,i] = smooth(npp_fut_detect_std2[:,j,i])[0:250]
            npp_fut_detectsmooth_rang[:,j,i] = smooth(npp_fut_detect_rang[:,j,i])[0:250]
            npp_futndep_detectsmooth_std1[:,j,i] = smooth(npp_futndep_detect_std1[:,j,i])[0:250]
            npp_futndep_detectsmooth_std2[:,j,i] = smooth(npp_futndep_detect_std2[:,j,i])[0:250]
            npp_futndep_detectsmooth_rang[:,j,i] = smooth(npp_futndep_detect_rang[:,j,i])[0:250]
            
        if np.ma.is_mask(temp_pic_ez[0,j,i]) == False:
            temp_fut_detectsmooth_std1_ez[:,j,i] = smooth(temp_fut_detect_std1_ez[:,j,i])[0:250]
            temp_fut_detectsmooth_std2_ez[:,j,i] = smooth(temp_fut_detect_std2_ez[:,j,i])[0:250]
            temp_fut_detectsmooth_rang_ez[:,j,i] = smooth(temp_fut_detect_rang_ez[:,j,i])[0:250]
            
        if np.ma.is_mask(temp_pic_utz[0,j,i]) == False:
            temp_fut_detectsmooth_std1_utz[:,j,i] = smooth(temp_fut_detect_std1_utz[:,j,i])[0:250]
            temp_fut_detectsmooth_std2_utz[:,j,i] = smooth(temp_fut_detect_std2_utz[:,j,i])[0:250]
            temp_fut_detectsmooth_rang_utz[:,j,i] = smooth(temp_fut_detect_rang_utz[:,j,i])[0:250]
            
           
        if np.ma.is_mask(no3_pic_ez[0,j,i]) == False:
            no3_picndep_detectsmooth_std1_ez[:,j,i] = smooth(no3_picndep_detect_std1_ez[:,j,i])[0:250]
            no3_picndep_detectsmooth_std2_ez[:,j,i] = smooth(no3_picndep_detect_std2_ez[:,j,i])[0:250]
            no3_picndep_detectsmooth_rang_ez[:,j,i] = smooth(no3_picndep_detect_rang_ez[:,j,i])[0:250]
            no3_fut_detectsmooth_std1_ez[:,j,i] = smooth(no3_fut_detect_std1_ez[:,j,i])[0:250]
            no3_fut_detectsmooth_std2_ez[:,j,i] = smooth(no3_fut_detect_std2_ez[:,j,i])[0:250]
            no3_fut_detectsmooth_rang_ez[:,j,i] = smooth(no3_fut_detect_rang_ez[:,j,i])[0:250]
            no3_futndep_detectsmooth_std1_ez[:,j,i] = smooth(no3_futndep_detect_std1_ez[:,j,i])[0:250]
            no3_futndep_detectsmooth_std2_ez[:,j,i] = smooth(no3_futndep_detect_std2_ez[:,j,i])[0:250]
            no3_futndep_detectsmooth_rang_ez[:,j,i] = smooth(no3_futndep_detect_rang_ez[:,j,i])[0:250]
        if np.ma.is_mask(no3_pic_utz[0,j,i]) == False:
            no3_picndep_detectsmooth_std1_utz[:,j,i] = smooth(no3_picndep_detect_std1_utz[:,j,i])[0:250]
            no3_picndep_detectsmooth_std2_utz[:,j,i] = smooth(no3_picndep_detect_std2_utz[:,j,i])[0:250]
            no3_picndep_detectsmooth_rang_utz[:,j,i] = smooth(no3_picndep_detect_rang_utz[:,j,i])[0:250]
            no3_fut_detectsmooth_std1_utz[:,j,i] = smooth(no3_fut_detect_std1_utz[:,j,i])[0:250]
            no3_fut_detectsmooth_std2_utz[:,j,i] = smooth(no3_fut_detect_std2_utz[:,j,i])[0:250]
            no3_fut_detectsmooth_rang_utz[:,j,i] = smooth(no3_fut_detect_rang_utz[:,j,i])[0:250]
            no3_futndep_detectsmooth_std1_utz[:,j,i] = smooth(no3_futndep_detect_std1_utz[:,j,i])[0:250]
            no3_futndep_detectsmooth_std2_utz[:,j,i] = smooth(no3_futndep_detect_std2_utz[:,j,i])[0:250]
            no3_futndep_detectsmooth_rang_utz[:,j,i] = smooth(no3_futndep_detect_rang_utz[:,j,i])[0:250]
        
        if np.ma.is_mask(nst_pic_ez[0,j,i]) == False:
            nst_picndep_detectsmooth_std1_ez[:,j,i] = smooth(nst_picndep_detect_std1_ez[:,j,i])[0:250]
            nst_picndep_detectsmooth_std2_ez[:,j,i] = smooth(nst_picndep_detect_std2_ez[:,j,i])[0:250]
            nst_picndep_detectsmooth_rang_ez[:,j,i] = smooth(nst_picndep_detect_rang_ez[:,j,i])[0:250]
            nst_fut_detectsmooth_std1_ez[:,j,i] = smooth(nst_fut_detect_std1_ez[:,j,i])[0:250]
            nst_fut_detectsmooth_std2_ez[:,j,i] = smooth(nst_fut_detect_std2_ez[:,j,i])[0:250]
            nst_fut_detectsmooth_rang_ez[:,j,i] = smooth(nst_fut_detect_rang_ez[:,j,i])[0:250]
            nst_futndep_detectsmooth_std1_ez[:,j,i] = smooth(nst_futndep_detect_std1_ez[:,j,i])[0:250]
            nst_futndep_detectsmooth_std2_ez[:,j,i] = smooth(nst_futndep_detect_std2_ez[:,j,i])[0:250]
            nst_futndep_detectsmooth_rang_ez[:,j,i] = smooth(nst_futndep_detect_rang_ez[:,j,i])[0:250]
        if np.ma.is_mask(nst_pic_utz[0,j,i]) == False:
            nst_picndep_detectsmooth_std1_utz[:,j,i] = smooth(nst_picndep_detect_std1_utz[:,j,i])[0:250]
            nst_picndep_detectsmooth_std2_utz[:,j,i] = smooth(nst_picndep_detect_std2_utz[:,j,i])[0:250]
            nst_picndep_detectsmooth_rang_utz[:,j,i] = smooth(nst_picndep_detect_rang_utz[:,j,i])[0:250]
            nst_fut_detectsmooth_std1_utz[:,j,i] = smooth(nst_fut_detect_std1_utz[:,j,i])[0:250]
            nst_fut_detectsmooth_std2_utz[:,j,i] = smooth(nst_fut_detect_std2_utz[:,j,i])[0:250]
            nst_fut_detectsmooth_rang_utz[:,j,i] = smooth(nst_fut_detect_rang_utz[:,j,i])[0:250]
            nst_futndep_detectsmooth_std1_utz[:,j,i] = smooth(nst_futndep_detect_std1_utz[:,j,i])[0:250]
            nst_futndep_detectsmooth_std2_utz[:,j,i] = smooth(nst_futndep_detect_std2_utz[:,j,i])[0:250]
            nst_futndep_detectsmooth_rang_utz[:,j,i] = smooth(nst_futndep_detect_rang_utz[:,j,i])[0:250]
        
        if np.ma.is_mask(d15n_no3_pic_ez[0,j,i]) == False:
            d15n_no3_picndep_detectsmooth_std1_ez[:,j,i] = smooth(d15n_no3_picndep_detect_std1_ez[:,j,i])[0:250]
            d15n_no3_picndep_detectsmooth_std2_ez[:,j,i] = smooth(d15n_no3_picndep_detect_std2_ez[:,j,i])[0:250]
            d15n_no3_picndep_detectsmooth_rang_ez[:,j,i] = smooth(d15n_no3_picndep_detect_rang_ez[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_std1_ez[:,j,i] = smooth(d15n_no3_fut_detect_std1_ez[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_std2_ez[:,j,i] = smooth(d15n_no3_fut_detect_std2_ez[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_rang_ez[:,j,i] = smooth(d15n_no3_fut_detect_rang_ez[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_std1_ez[:,j,i] = smooth(d15n_no3_futndep_detect_std1_ez[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_std2_ez[:,j,i] = smooth(d15n_no3_futndep_detect_std2_ez[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_rang_ez[:,j,i] = smooth(d15n_no3_futndep_detect_rang_ez[:,j,i])[0:250]
        if np.ma.is_mask(d15n_no3_pic_utz[0,j,i]) == False:
            d15n_no3_picndep_detectsmooth_std1_utz[:,j,i] = smooth(d15n_no3_picndep_detect_std1_utz[:,j,i])[0:250]
            d15n_no3_picndep_detectsmooth_std2_utz[:,j,i] = smooth(d15n_no3_picndep_detect_std2_utz[:,j,i])[0:250]
            d15n_no3_picndep_detectsmooth_rang_utz[:,j,i] = smooth(d15n_no3_picndep_detect_rang_utz[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_std1_utz[:,j,i] = smooth(d15n_no3_fut_detect_std1_utz[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_std2_utz[:,j,i] = smooth(d15n_no3_fut_detect_std2_utz[:,j,i])[0:250]
            d15n_no3_fut_detectsmooth_rang_utz[:,j,i] = smooth(d15n_no3_fut_detect_rang_utz[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_std1_utz[:,j,i] = smooth(d15n_no3_futndep_detect_std1_utz[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_std2_utz[:,j,i] = smooth(d15n_no3_futndep_detect_std2_utz[:,j,i])[0:250]
            d15n_no3_futndep_detectsmooth_rang_utz[:,j,i] = smooth(d15n_no3_futndep_detect_rang_utz[:,j,i])[0:250]
        
        if np.ma.is_mask(d15n_pom_pic_ez[0,j,i]) == False:
            d15n_pom_picndep_detectsmooth_std1_ez[:,j,i] = smooth(d15n_pom_picndep_detect_std1_ez[:,j,i])[0:250]
            d15n_pom_picndep_detectsmooth_std2_ez[:,j,i] = smooth(d15n_pom_picndep_detect_std2_ez[:,j,i])[0:250]
            d15n_pom_picndep_detectsmooth_rang_ez[:,j,i] = smooth(d15n_pom_picndep_detect_rang_ez[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_std1_ez[:,j,i] = smooth(d15n_pom_fut_detect_std1_ez[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_std2_ez[:,j,i] = smooth(d15n_pom_fut_detect_std2_ez[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_rang_ez[:,j,i] = smooth(d15n_pom_fut_detect_rang_ez[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_std1_ez[:,j,i] = smooth(d15n_pom_futndep_detect_std1_ez[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_std2_ez[:,j,i] = smooth(d15n_pom_futndep_detect_std2_ez[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_rang_ez[:,j,i] = smooth(d15n_pom_futndep_detect_rang_ez[:,j,i])[0:250]
        if np.ma.is_mask(d15n_pom_pic_utz[0,j,i]) == False:
            d15n_pom_picndep_detectsmooth_std1_utz[:,j,i] = smooth(d15n_pom_picndep_detect_std1_utz[:,j,i])[0:250]
            d15n_pom_picndep_detectsmooth_std2_utz[:,j,i] = smooth(d15n_pom_picndep_detect_std2_utz[:,j,i])[0:250]
            d15n_pom_picndep_detectsmooth_rang_utz[:,j,i] = smooth(d15n_pom_picndep_detect_rang_utz[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_std1_utz[:,j,i] = smooth(d15n_pom_fut_detect_std1_utz[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_std2_utz[:,j,i] = smooth(d15n_pom_fut_detect_std2_utz[:,j,i])[0:250]
            d15n_pom_fut_detectsmooth_rang_utz[:,j,i] = smooth(d15n_pom_fut_detect_rang_utz[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_std1_utz[:,j,i] = smooth(d15n_pom_futndep_detect_std1_utz[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_std2_utz[:,j,i] = smooth(d15n_pom_futndep_detect_std2_utz[:,j,i])[0:250]
            d15n_pom_futndep_detectsmooth_rang_utz[:,j,i] = smooth(d15n_pom_futndep_detect_rang_utz[:,j,i])[0:250]
        
print("Smoothing of emergences complete")


#%% check the outcome

lab = ['Nat', 'Nat + N dep', 'HistRCP8.5', 'HistRCP8.5 + N dep']

alf=0.7
i = 160; j = 110
print(lon[i], lat[j])


fig = plt.figure(figsize=(18,10))
ax1 = plt.subplot(2,4,1)
plt.title('Raw timeseries')
plt.plot(time,d15n_no3_pic_ez[:,j,i], color='k',alpha=alf, label=lab[0])
plt.plot(time,d15n_no3_picndep_ez[:,j,i], color='royalblue',alpha=alf, label=lab[1])
plt.plot(time,d15n_no3_fut_ez[:,j,i], color='goldenrod',alpha=alf, label=lab[2])
plt.plot(time,d15n_no3_futndep_ez[:,j,i], color='firebrick',alpha=alf, label=lab[3])
plt.legend(frameon=False, loc='upper center', ncol=4, bbox_to_anchor=(2.2,1.2))
plt.ylabel('$\delta^{15}$N$_{NO_3}$')
plt.subplot(2,4,2)
plt.title('Detrended')
plt.plot(time,d15n_no3_pic_ez_detrended[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_ez_detrended[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_ez_detrended[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_ez_detrended[:,j,i], color='firebrick',alpha=alf)
plt.subplot(2,4,3)
plt.title('Normalised and smoothed')
plt.plot(time,d15n_no3_pic_ez_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_ez_smoothed[:,j,i], color='k',linewidth=2)
plt.plot(time,d15n_no3_picndep_ez_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_ez_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_ez_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_ez_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_ez_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_ez_smoothed[:,j,i], color='firebrick', linewidth=2)
plt.subplot(2,4,4)
plt.title('Signal emergence')
#plt.plot(time,d15n_no3_picndep_detectsmooth_std1_ez[:,j,i], color='royalblue', linestyle=':')
plt.plot(time,d15n_no3_picndep_detectsmooth_std2_ez[:,j,i], color='royalblue', linestyle='-')
#plt.plot(time,d15n_no3_picndep_detectsmooth_rang_ez[:,j,i], color='royalblue', linestyle='-')
#plt.plot(time,d15n_no3_fut_detectsmooth_std1_ez[:,j,i], color='goldenrod', linestyle=':')
plt.plot(time,d15n_no3_fut_detectsmooth_std2_ez[:,j,i], color='goldenrod', linestyle='-')
#plt.plot(time,d15n_no3_fut_detectsmooth_rang_ez[:,j,i], color='goldenrod', linestyle='-')
#plt.plot(time,d15n_no3_futndep_detectsmooth_std1_ez[:,j,i], color='firebrick', linestyle=':')
plt.plot(time,d15n_no3_futndep_detectsmooth_std2_ez[:,j,i], color='firebrick', linestyle='-')
#plt.plot(time,d15n_no3_futndep_detectsmooth_rang_ez[:,j,i], color='firebrick', linestyle='-')


ax5 = plt.subplot(2,4,5)
plt.plot(time,d15n_no3_pic_utz[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_utz[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz[:,j,i], color='firebrick',alpha=alf)
plt.ylabel('$\delta^{15}$N$_{NO_3}$')
plt.xlabel('Year')
plt.subplot(2,4,6)
plt.plot(time,d15n_no3_pic_utz_detrended[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz_detrended[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_utz_detrended[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz_detrended[:,j,i], color='firebrick',alpha=alf)
plt.xlabel('Year')
plt.subplot(2,4,7)
plt.plot(time,d15n_no3_pic_utz_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_utz_smoothed[:,j,i], color='k', linewidth=2)
plt.plot(time,d15n_no3_picndep_utz_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_utz_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_utz_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_utz_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz_smoothed[:,j,i], color='firebrick', linewidth=2)
plt.xlabel('Year')
plt.subplot(2,4,8)
plt.title('Signal emergence')
#plt.plot(time,d15n_no3_picndep_detectsmooth_std1_utz[:,j,i], color='royalblue', linestyle=':')
plt.plot(time,d15n_no3_picndep_detectsmooth_std2_utz[:,j,i], color='royalblue', linestyle='-')
#plt.plot(time,d15n_no3_picndep_detectsmooth_rang_utz[:,j,i], color='royalblue', linestyle='-')
#plt.plot(time,d15n_no3_fut_detectsmooth_std1_utz[:,j,i], color='goldenrod', linestyle=':')
plt.plot(time,d15n_no3_fut_detectsmooth_std2_utz[:,j,i], color='goldenrod', linestyle='-')
#plt.plot(time,d15n_no3_fut_detectsmooth_rang_utz[:,j,i], color='goldenrod', linestyle='-')
#plt.plot(time,d15n_no3_futndep_detectsmooth_std1_utz[:,j,i], color='firebrick', linestyle=':')
plt.plot(time,d15n_no3_futndep_detectsmooth_std2_utz[:,j,i], color='firebrick', linestyle='-')
#plt.plot(time,d15n_no3_futndep_detectsmooth_rang_utz[:,j,i], color='firebrick', linestyle='-')
plt.xlabel('Year')

xx=-0.35;yy=0.5
plt.text(xx,yy,'Euphotic\nZone', ha='center', va='center', transform=ax1.transAxes)
plt.text(xx,yy,'Twilight\nZone', ha='center', va='center', transform=ax5.transAxes)


# savefig
fig.savefig("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//figures//processToE-detection.png", dpi=300, bbox_to_inches='tight')


#%% find year of emergence by selecting years when emergence is consistently above a threshold

# set all values above threshold equal to 1
threshold = 0.5

fix_picndep_detectsmooth_std1[fix_picndep_detectsmooth_std1 > threshold] = 1.0
fix_picndep_detectsmooth_std2[fix_picndep_detectsmooth_std2 > threshold] = 1.0
fix_picndep_detectsmooth_rang[fix_picndep_detectsmooth_rang > threshold] = 1.0
fix_fut_detectsmooth_std1[fix_fut_detectsmooth_std1 > threshold] = 1.0
fix_fut_detectsmooth_std2[fix_fut_detectsmooth_std2 > threshold] = 1.0
fix_fut_detectsmooth_rang[fix_fut_detectsmooth_rang > threshold] = 1.0
fix_futndep_detectsmooth_std1[fix_futndep_detectsmooth_std1 > threshold] = 1.0
fix_futndep_detectsmooth_std2[fix_futndep_detectsmooth_std2 > threshold] = 1.0
fix_futndep_detectsmooth_rang[fix_futndep_detectsmooth_rang > threshold] = 1.0

npp_picndep_detectsmooth_std1[npp_picndep_detectsmooth_std1 > threshold] = 1.0
npp_picndep_detectsmooth_std2[npp_picndep_detectsmooth_std2 > threshold] = 1.0
npp_picndep_detectsmooth_rang[npp_picndep_detectsmooth_rang > threshold] = 1.0
npp_fut_detectsmooth_std1[npp_fut_detectsmooth_std1 > threshold] = 1.0
npp_fut_detectsmooth_std2[npp_fut_detectsmooth_std2 > threshold] = 1.0
npp_fut_detectsmooth_rang[npp_fut_detectsmooth_rang > threshold] = 1.0
npp_futndep_detectsmooth_std1[npp_futndep_detectsmooth_std1 > threshold] = 1.0
npp_futndep_detectsmooth_std2[npp_futndep_detectsmooth_std2 > threshold] = 1.0
npp_futndep_detectsmooth_rang[npp_futndep_detectsmooth_rang > threshold] = 1.0

temp_fut_detectsmooth_std1_ez[temp_fut_detectsmooth_std1_ez > threshold] = 1.0
temp_fut_detectsmooth_std2_ez[temp_fut_detectsmooth_std2_ez > threshold] = 1.0
temp_fut_detectsmooth_rang_ez[temp_fut_detectsmooth_rang_ez > threshold] = 1.0
temp_fut_detectsmooth_std1_utz[temp_fut_detectsmooth_std1_utz > threshold] = 1.0
temp_fut_detectsmooth_std2_utz[temp_fut_detectsmooth_std2_utz > threshold] = 1.0
temp_fut_detectsmooth_rang_utz[temp_fut_detectsmooth_rang_utz > threshold] = 1.0


no3_picndep_detectsmooth_std1_ez[no3_picndep_detectsmooth_std1_ez > threshold] = 1.0
no3_picndep_detectsmooth_std2_ez[no3_picndep_detectsmooth_std2_ez > threshold] = 1.0
no3_picndep_detectsmooth_rang_ez[no3_picndep_detectsmooth_rang_ez > threshold] = 1.0
no3_picndep_detectsmooth_std1_utz[no3_picndep_detectsmooth_std1_utz > threshold] = 1.0
no3_picndep_detectsmooth_std2_utz[no3_picndep_detectsmooth_std2_utz > threshold] = 1.0
no3_picndep_detectsmooth_rang_utz[no3_picndep_detectsmooth_rang_utz > threshold] = 1.0

nst_picndep_detectsmooth_std1_ez[nst_picndep_detectsmooth_std1_ez > threshold] = 1.0
nst_picndep_detectsmooth_std2_ez[nst_picndep_detectsmooth_std2_ez > threshold] = 1.0
nst_picndep_detectsmooth_rang_ez[nst_picndep_detectsmooth_rang_ez > threshold] = 1.0
nst_picndep_detectsmooth_std1_utz[nst_picndep_detectsmooth_std1_utz > threshold] = 1.0
nst_picndep_detectsmooth_std2_utz[nst_picndep_detectsmooth_std2_utz > threshold] = 1.0
nst_picndep_detectsmooth_rang_utz[nst_picndep_detectsmooth_rang_utz > threshold] = 1.0

d15n_no3_picndep_detectsmooth_std1_ez[d15n_no3_picndep_detectsmooth_std1_ez > threshold] = 1.0
d15n_no3_picndep_detectsmooth_std2_ez[d15n_no3_picndep_detectsmooth_std2_ez > threshold] = 1.0
d15n_no3_picndep_detectsmooth_rang_ez[d15n_no3_picndep_detectsmooth_rang_ez > threshold] = 1.0
d15n_no3_picndep_detectsmooth_std1_utz[d15n_no3_picndep_detectsmooth_std1_utz > threshold] = 1.0
d15n_no3_picndep_detectsmooth_std2_utz[d15n_no3_picndep_detectsmooth_std2_utz > threshold] = 1.0
d15n_no3_picndep_detectsmooth_rang_utz[d15n_no3_picndep_detectsmooth_rang_utz > threshold] = 1.0

d15n_pom_picndep_detectsmooth_std1_ez[d15n_pom_picndep_detectsmooth_std1_ez > threshold] = 1.0
d15n_pom_picndep_detectsmooth_std2_ez[d15n_pom_picndep_detectsmooth_std2_ez > threshold] = 1.0
d15n_pom_picndep_detectsmooth_rang_ez[d15n_pom_picndep_detectsmooth_rang_ez > threshold] = 1.0
d15n_pom_picndep_detectsmooth_std1_utz[d15n_pom_picndep_detectsmooth_std1_utz > threshold] = 1.0
d15n_pom_picndep_detectsmooth_std2_utz[d15n_pom_picndep_detectsmooth_std2_utz > threshold] = 1.0
d15n_pom_picndep_detectsmooth_rang_utz[d15n_pom_picndep_detectsmooth_rang_utz > threshold] = 1.0


no3_fut_detectsmooth_std1_ez[no3_fut_detectsmooth_std1_ez > threshold] = 1.0
no3_fut_detectsmooth_std2_ez[no3_fut_detectsmooth_std2_ez > threshold] = 1.0
no3_fut_detectsmooth_rang_ez[no3_fut_detectsmooth_rang_ez > threshold] = 1.0
no3_fut_detectsmooth_std1_utz[no3_fut_detectsmooth_std1_utz > threshold] = 1.0
no3_fut_detectsmooth_std2_utz[no3_fut_detectsmooth_std2_utz > threshold] = 1.0
no3_fut_detectsmooth_rang_utz[no3_fut_detectsmooth_rang_utz > threshold] = 1.0

nst_fut_detectsmooth_std1_ez[nst_fut_detectsmooth_std1_ez > threshold] = 1.0
nst_fut_detectsmooth_std2_ez[nst_fut_detectsmooth_std2_ez > threshold] = 1.0
nst_fut_detectsmooth_rang_ez[nst_fut_detectsmooth_rang_ez > threshold] = 1.0
nst_fut_detectsmooth_std1_utz[nst_fut_detectsmooth_std1_utz > threshold] = 1.0
nst_fut_detectsmooth_std2_utz[nst_fut_detectsmooth_std2_utz > threshold] = 1.0
nst_fut_detectsmooth_rang_utz[nst_fut_detectsmooth_rang_utz > threshold] = 1.0

d15n_no3_fut_detectsmooth_std1_ez[d15n_no3_fut_detectsmooth_std1_ez > threshold] = 1.0
d15n_no3_fut_detectsmooth_std2_ez[d15n_no3_fut_detectsmooth_std2_ez > threshold] = 1.0
d15n_no3_fut_detectsmooth_rang_ez[d15n_no3_fut_detectsmooth_rang_ez > threshold] = 1.0
d15n_no3_fut_detectsmooth_std1_utz[d15n_no3_fut_detectsmooth_std1_utz > threshold] = 1.0
d15n_no3_fut_detectsmooth_std2_utz[d15n_no3_fut_detectsmooth_std2_utz > threshold] = 1.0
d15n_no3_fut_detectsmooth_rang_utz[d15n_no3_fut_detectsmooth_rang_utz > threshold] = 1.0

d15n_pom_fut_detectsmooth_std1_ez[d15n_pom_fut_detectsmooth_std1_ez > threshold] = 1.0
d15n_pom_fut_detectsmooth_std2_ez[d15n_pom_fut_detectsmooth_std2_ez > threshold] = 1.0
d15n_pom_fut_detectsmooth_rang_ez[d15n_pom_fut_detectsmooth_rang_ez > threshold] = 1.0
d15n_pom_fut_detectsmooth_std1_utz[d15n_pom_fut_detectsmooth_std1_utz > threshold] = 1.0
d15n_pom_fut_detectsmooth_std2_utz[d15n_pom_fut_detectsmooth_std2_utz > threshold] = 1.0
d15n_pom_fut_detectsmooth_rang_utz[d15n_pom_fut_detectsmooth_rang_utz > threshold] = 1.0


no3_futndep_detectsmooth_std1_ez[no3_futndep_detectsmooth_std1_ez > threshold] = 1.0
no3_futndep_detectsmooth_std2_ez[no3_futndep_detectsmooth_std2_ez > threshold] = 1.0
no3_futndep_detectsmooth_rang_ez[no3_futndep_detectsmooth_rang_ez > threshold] = 1.0
no3_futndep_detectsmooth_std1_utz[no3_futndep_detectsmooth_std1_utz > threshold] = 1.0
no3_futndep_detectsmooth_std2_utz[no3_futndep_detectsmooth_std2_utz > threshold] = 1.0
no3_futndep_detectsmooth_rang_utz[no3_futndep_detectsmooth_rang_utz > threshold] = 1.0

nst_futndep_detectsmooth_std1_ez[nst_futndep_detectsmooth_std1_ez > threshold] = 1.0
nst_futndep_detectsmooth_std2_ez[nst_futndep_detectsmooth_std2_ez > threshold] = 1.0
nst_futndep_detectsmooth_rang_ez[nst_futndep_detectsmooth_rang_ez > threshold] = 1.0
nst_futndep_detectsmooth_std1_utz[nst_futndep_detectsmooth_std1_utz > threshold] = 1.0
nst_futndep_detectsmooth_std2_utz[nst_futndep_detectsmooth_std2_utz > threshold] = 1.0
nst_futndep_detectsmooth_rang_utz[nst_futndep_detectsmooth_rang_utz > threshold] = 1.0

d15n_no3_futndep_detectsmooth_std1_ez[d15n_no3_futndep_detectsmooth_std1_ez > threshold] = 1.0
d15n_no3_futndep_detectsmooth_std2_ez[d15n_no3_futndep_detectsmooth_std2_ez > threshold] = 1.0
d15n_no3_futndep_detectsmooth_rang_ez[d15n_no3_futndep_detectsmooth_rang_ez > threshold] = 1.0
d15n_no3_futndep_detectsmooth_std1_utz[d15n_no3_futndep_detectsmooth_std1_utz > threshold] = 1.0
d15n_no3_futndep_detectsmooth_std2_utz[d15n_no3_futndep_detectsmooth_std2_utz > threshold] = 1.0
d15n_no3_futndep_detectsmooth_rang_utz[d15n_no3_futndep_detectsmooth_rang_utz > threshold] = 1.0

d15n_pom_futndep_detectsmooth_std1_ez[d15n_pom_futndep_detectsmooth_std1_ez > threshold] = 1.0
d15n_pom_futndep_detectsmooth_std2_ez[d15n_pom_futndep_detectsmooth_std2_ez > threshold] = 1.0
d15n_pom_futndep_detectsmooth_rang_ez[d15n_pom_futndep_detectsmooth_rang_ez > threshold] = 1.0
d15n_pom_futndep_detectsmooth_std1_utz[d15n_pom_futndep_detectsmooth_std1_utz > threshold] = 1.0
d15n_pom_futndep_detectsmooth_std2_utz[d15n_pom_futndep_detectsmooth_std2_utz > threshold] = 1.0
d15n_pom_futndep_detectsmooth_rang_utz[d15n_pom_futndep_detectsmooth_rang_utz > threshold] = 1.0


# obtain year to year differences
fix_picndep_yearchange_std1 = np.zeros((250,180,360))
fix_picndep_yearchange_std2 = np.zeros((250,180,360))
fix_picndep_yearchange_rang = np.zeros((250,180,360))
fix_fut_yearchange_std1 = np.zeros((250,180,360))
fix_fut_yearchange_std2 = np.zeros((250,180,360))
fix_fut_yearchange_rang = np.zeros((250,180,360))
fix_futndep_yearchange_std1 = np.zeros((250,180,360))
fix_futndep_yearchange_std2 = np.zeros((250,180,360))
fix_futndep_yearchange_rang = np.zeros((250,180,360))

fix_picndep_yearchange_std1[1:250,:,:] = fix_picndep_detectsmooth_std1[1:250,:,:] - fix_picndep_detectsmooth_std1[0:249,:,:]
fix_picndep_yearchange_std2[1:250,:,:] = fix_picndep_detectsmooth_std2[1:250,:,:] - fix_picndep_detectsmooth_std2[0:249,:,:]
fix_picndep_yearchange_rang[1:250,:,:] = fix_picndep_detectsmooth_rang[1:250,:,:] - fix_picndep_detectsmooth_rang[0:249,:,:]
fix_fut_yearchange_std1[1:250,:,:] = fix_fut_detectsmooth_std1[1:250,:,:] - fix_fut_detectsmooth_std1[0:249,:,:]
fix_fut_yearchange_std2[1:250,:,:] = fix_fut_detectsmooth_std2[1:250,:,:] - fix_fut_detectsmooth_std2[0:249,:,:]
fix_fut_yearchange_rang[1:250,:,:] = fix_fut_detectsmooth_rang[1:250,:,:] - fix_fut_detectsmooth_rang[0:249,:,:]
fix_futndep_yearchange_std1[1:250,:,:] = fix_futndep_detectsmooth_std1[1:250,:,:] - fix_futndep_detectsmooth_std1[0:249,:,:]
fix_futndep_yearchange_std2[1:250,:,:] = fix_futndep_detectsmooth_std2[1:250,:,:] - fix_futndep_detectsmooth_std2[0:249,:,:]
fix_futndep_yearchange_rang[1:250,:,:] = fix_futndep_detectsmooth_rang[1:250,:,:] - fix_futndep_detectsmooth_rang[0:249,:,:]

npp_picndep_yearchange_std1 = np.zeros((250,180,360))
npp_picndep_yearchange_std2 = np.zeros((250,180,360))
npp_picndep_yearchange_rang = np.zeros((250,180,360))
npp_fut_yearchange_std1 = np.zeros((250,180,360))
npp_fut_yearchange_std2 = np.zeros((250,180,360))
npp_fut_yearchange_rang = np.zeros((250,180,360))
npp_futndep_yearchange_std1 = np.zeros((250,180,360))
npp_futndep_yearchange_std2 = np.zeros((250,180,360))
npp_futndep_yearchange_rang = np.zeros((250,180,360))

npp_picndep_yearchange_std1[1:250,:,:] = npp_picndep_detectsmooth_std1[1:250,:,:] - npp_picndep_detectsmooth_std1[0:249,:,:]
npp_picndep_yearchange_std2[1:250,:,:] = npp_picndep_detectsmooth_std2[1:250,:,:] - npp_picndep_detectsmooth_std2[0:249,:,:]
npp_picndep_yearchange_rang[1:250,:,:] = npp_picndep_detectsmooth_rang[1:250,:,:] - npp_picndep_detectsmooth_rang[0:249,:,:]
npp_fut_yearchange_std1[1:250,:,:] = npp_fut_detectsmooth_std1[1:250,:,:] - npp_fut_detectsmooth_std1[0:249,:,:]
npp_fut_yearchange_std2[1:250,:,:] = npp_fut_detectsmooth_std2[1:250,:,:] - npp_fut_detectsmooth_std2[0:249,:,:]
npp_fut_yearchange_rang[1:250,:,:] = npp_fut_detectsmooth_rang[1:250,:,:] - npp_fut_detectsmooth_rang[0:249,:,:]
npp_futndep_yearchange_std1[1:250,:,:] = npp_futndep_detectsmooth_std1[1:250,:,:] - npp_futndep_detectsmooth_std1[0:249,:,:]
npp_futndep_yearchange_std2[1:250,:,:] = npp_futndep_detectsmooth_std2[1:250,:,:] - npp_futndep_detectsmooth_std2[0:249,:,:]
npp_futndep_yearchange_rang[1:250,:,:] = npp_futndep_detectsmooth_rang[1:250,:,:] - npp_futndep_detectsmooth_rang[0:249,:,:]

temp_fut_yearchange_std1_ez = np.zeros((250,180,360))
temp_fut_yearchange_std2_ez = np.zeros((250,180,360))
temp_fut_yearchange_rang_ez = np.zeros((250,180,360))
temp_fut_yearchange_std1_utz = np.zeros((250,180,360))
temp_fut_yearchange_std2_utz = np.zeros((250,180,360))
temp_fut_yearchange_rang_utz = np.zeros((250,180,360))

temp_fut_yearchange_std1_ez[1:250,:,:] = temp_fut_detectsmooth_std1_ez[1:250,:,:] - temp_fut_detectsmooth_std1_ez[0:249,:,:]
temp_fut_yearchange_std2_ez[1:250,:,:] = temp_fut_detectsmooth_std2_ez[1:250,:,:] - temp_fut_detectsmooth_std2_ez[0:249,:,:]
temp_fut_yearchange_rang_ez[1:250,:,:] = temp_fut_detectsmooth_rang_ez[1:250,:,:] - temp_fut_detectsmooth_rang_ez[0:249,:,:]
temp_fut_yearchange_std1_utz[1:250,:,:] = temp_fut_detectsmooth_std1_utz[1:250,:,:] - temp_fut_detectsmooth_std1_utz[0:249,:,:]
temp_fut_yearchange_std2_utz[1:250,:,:] = temp_fut_detectsmooth_std2_utz[1:250,:,:] - temp_fut_detectsmooth_std2_utz[0:249,:,:]
temp_fut_yearchange_rang_utz[1:250,:,:] = temp_fut_detectsmooth_rang_utz[1:250,:,:] - temp_fut_detectsmooth_rang_utz[0:249,:,:]


no3_picndep_yearchange_std1_ez = np.zeros((250,180,360))
no3_picndep_yearchange_std2_ez = np.zeros((250,180,360))
no3_picndep_yearchange_rang_ez = np.zeros((250,180,360))
no3_picndep_yearchange_std1_utz = np.zeros((250,180,360))
no3_picndep_yearchange_std2_utz = np.zeros((250,180,360))
no3_picndep_yearchange_rang_utz = np.zeros((250,180,360))

no3_picndep_yearchange_std1_ez[1:250,:,:] = no3_picndep_detectsmooth_std1_ez[1:250,:,:] - no3_picndep_detectsmooth_std1_ez[0:249,:,:]
no3_picndep_yearchange_std2_ez[1:250,:,:] = no3_picndep_detectsmooth_std2_ez[1:250,:,:] - no3_picndep_detectsmooth_std2_ez[0:249,:,:]
no3_picndep_yearchange_rang_ez[1:250,:,:] = no3_picndep_detectsmooth_rang_ez[1:250,:,:] - no3_picndep_detectsmooth_rang_ez[0:249,:,:]
no3_picndep_yearchange_std1_utz[1:250,:,:] = no3_picndep_detectsmooth_std1_utz[1:250,:,:] - no3_picndep_detectsmooth_std1_utz[0:249,:,:]
no3_picndep_yearchange_std2_utz[1:250,:,:] = no3_picndep_detectsmooth_std2_utz[1:250,:,:] - no3_picndep_detectsmooth_std2_utz[0:249,:,:]
no3_picndep_yearchange_rang_utz[1:250,:,:] = no3_picndep_detectsmooth_rang_utz[1:250,:,:] - no3_picndep_detectsmooth_rang_utz[0:249,:,:]

nst_picndep_yearchange_std1_ez = np.zeros((250,180,360))
nst_picndep_yearchange_std2_ez = np.zeros((250,180,360))
nst_picndep_yearchange_rang_ez = np.zeros((250,180,360))
nst_picndep_yearchange_std1_utz = np.zeros((250,180,360))
nst_picndep_yearchange_std2_utz = np.zeros((250,180,360))
nst_picndep_yearchange_rang_utz = np.zeros((250,180,360))

nst_picndep_yearchange_std1_ez[1:250,:,:] = nst_picndep_detectsmooth_std1_ez[1:250,:,:] - nst_picndep_detectsmooth_std1_ez[0:249,:,:]
nst_picndep_yearchange_std2_ez[1:250,:,:] = nst_picndep_detectsmooth_std2_ez[1:250,:,:] - nst_picndep_detectsmooth_std2_ez[0:249,:,:]
nst_picndep_yearchange_rang_ez[1:250,:,:] = nst_picndep_detectsmooth_rang_ez[1:250,:,:] - nst_picndep_detectsmooth_rang_ez[0:249,:,:]
nst_picndep_yearchange_std1_utz[1:250,:,:] = nst_picndep_detectsmooth_std1_utz[1:250,:,:] - nst_picndep_detectsmooth_std1_utz[0:249,:,:]
nst_picndep_yearchange_std2_utz[1:250,:,:] = nst_picndep_detectsmooth_std2_utz[1:250,:,:] - nst_picndep_detectsmooth_std2_utz[0:249,:,:]
nst_picndep_yearchange_rang_utz[1:250,:,:] = nst_picndep_detectsmooth_rang_utz[1:250,:,:] - nst_picndep_detectsmooth_rang_utz[0:249,:,:]

d15n_no3_picndep_yearchange_std1_ez = np.zeros((250,180,360))
d15n_no3_picndep_yearchange_std2_ez = np.zeros((250,180,360))
d15n_no3_picndep_yearchange_rang_ez = np.zeros((250,180,360))
d15n_no3_picndep_yearchange_std1_utz = np.zeros((250,180,360))
d15n_no3_picndep_yearchange_std2_utz = np.zeros((250,180,360))
d15n_no3_picndep_yearchange_rang_utz = np.zeros((250,180,360))

d15n_no3_picndep_yearchange_std1_ez[1:250,:,:] = d15n_no3_picndep_detectsmooth_std1_ez[1:250,:,:] - d15n_no3_picndep_detectsmooth_std1_ez[0:249,:,:]
d15n_no3_picndep_yearchange_std2_ez[1:250,:,:] = d15n_no3_picndep_detectsmooth_std2_ez[1:250,:,:] - d15n_no3_picndep_detectsmooth_std2_ez[0:249,:,:]
d15n_no3_picndep_yearchange_rang_ez[1:250,:,:] = d15n_no3_picndep_detectsmooth_rang_ez[1:250,:,:] - d15n_no3_picndep_detectsmooth_rang_ez[0:249,:,:]
d15n_no3_picndep_yearchange_std1_utz[1:250,:,:] = d15n_no3_picndep_detectsmooth_std1_utz[1:250,:,:] - d15n_no3_picndep_detectsmooth_std1_utz[0:249,:,:]
d15n_no3_picndep_yearchange_std2_utz[1:250,:,:] = d15n_no3_picndep_detectsmooth_std2_utz[1:250,:,:] - d15n_no3_picndep_detectsmooth_std2_utz[0:249,:,:]
d15n_no3_picndep_yearchange_rang_utz[1:250,:,:] = d15n_no3_picndep_detectsmooth_rang_utz[1:250,:,:] - d15n_no3_picndep_detectsmooth_rang_utz[0:249,:,:]

d15n_pom_picndep_yearchange_std1_ez = np.zeros((250,180,360))
d15n_pom_picndep_yearchange_std2_ez = np.zeros((250,180,360))
d15n_pom_picndep_yearchange_rang_ez = np.zeros((250,180,360))
d15n_pom_picndep_yearchange_std1_utz = np.zeros((250,180,360))
d15n_pom_picndep_yearchange_std2_utz = np.zeros((250,180,360))
d15n_pom_picndep_yearchange_rang_utz = np.zeros((250,180,360))

d15n_pom_picndep_yearchange_std1_ez[1:250,:,:] = d15n_pom_picndep_detectsmooth_std1_ez[1:250,:,:] - d15n_pom_picndep_detectsmooth_std1_ez[0:249,:,:]
d15n_pom_picndep_yearchange_std2_ez[1:250,:,:] = d15n_pom_picndep_detectsmooth_std2_ez[1:250,:,:] - d15n_pom_picndep_detectsmooth_std2_ez[0:249,:,:]
d15n_pom_picndep_yearchange_rang_ez[1:250,:,:] = d15n_pom_picndep_detectsmooth_rang_ez[1:250,:,:] - d15n_pom_picndep_detectsmooth_rang_ez[0:249,:,:]
d15n_pom_picndep_yearchange_std1_utz[1:250,:,:] = d15n_pom_picndep_detectsmooth_std1_utz[1:250,:,:] - d15n_pom_picndep_detectsmooth_std1_utz[0:249,:,:]
d15n_pom_picndep_yearchange_std2_utz[1:250,:,:] = d15n_pom_picndep_detectsmooth_std2_utz[1:250,:,:] - d15n_pom_picndep_detectsmooth_std2_utz[0:249,:,:]
d15n_pom_picndep_yearchange_rang_utz[1:250,:,:] = d15n_pom_picndep_detectsmooth_rang_utz[1:250,:,:] - d15n_pom_picndep_detectsmooth_rang_utz[0:249,:,:]


no3_fut_yearchange_std1_ez = np.zeros((250,180,360))
no3_fut_yearchange_std2_ez = np.zeros((250,180,360))
no3_fut_yearchange_rang_ez = np.zeros((250,180,360))
no3_fut_yearchange_std1_utz = np.zeros((250,180,360))
no3_fut_yearchange_std2_utz = np.zeros((250,180,360))
no3_fut_yearchange_rang_utz = np.zeros((250,180,360))

no3_fut_yearchange_std1_ez[1:250,:,:] = no3_fut_detectsmooth_std1_ez[1:250,:,:] - no3_fut_detectsmooth_std1_ez[0:249,:,:]
no3_fut_yearchange_std2_ez[1:250,:,:] = no3_fut_detectsmooth_std2_ez[1:250,:,:] - no3_fut_detectsmooth_std2_ez[0:249,:,:]
no3_fut_yearchange_rang_ez[1:250,:,:] = no3_fut_detectsmooth_rang_ez[1:250,:,:] - no3_fut_detectsmooth_rang_ez[0:249,:,:]
no3_fut_yearchange_std1_utz[1:250,:,:] = no3_fut_detectsmooth_std1_utz[1:250,:,:] - no3_fut_detectsmooth_std1_utz[0:249,:,:]
no3_fut_yearchange_std2_utz[1:250,:,:] = no3_fut_detectsmooth_std2_utz[1:250,:,:] - no3_fut_detectsmooth_std2_utz[0:249,:,:]
no3_fut_yearchange_rang_utz[1:250,:,:] = no3_fut_detectsmooth_rang_utz[1:250,:,:] - no3_fut_detectsmooth_rang_utz[0:249,:,:]

nst_fut_yearchange_std1_ez = np.zeros((250,180,360))
nst_fut_yearchange_std2_ez = np.zeros((250,180,360))
nst_fut_yearchange_rang_ez = np.zeros((250,180,360))
nst_fut_yearchange_std1_utz = np.zeros((250,180,360))
nst_fut_yearchange_std2_utz = np.zeros((250,180,360))
nst_fut_yearchange_rang_utz = np.zeros((250,180,360))

nst_fut_yearchange_std1_ez[1:250,:,:] = nst_fut_detectsmooth_std1_ez[1:250,:,:] - nst_fut_detectsmooth_std1_ez[0:249,:,:]
nst_fut_yearchange_std2_ez[1:250,:,:] = nst_fut_detectsmooth_std2_ez[1:250,:,:] - nst_fut_detectsmooth_std2_ez[0:249,:,:]
nst_fut_yearchange_rang_ez[1:250,:,:] = nst_fut_detectsmooth_rang_ez[1:250,:,:] - nst_fut_detectsmooth_rang_ez[0:249,:,:]
nst_fut_yearchange_std1_utz[1:250,:,:] = nst_fut_detectsmooth_std1_utz[1:250,:,:] - nst_fut_detectsmooth_std1_utz[0:249,:,:]
nst_fut_yearchange_std2_utz[1:250,:,:] = nst_fut_detectsmooth_std2_utz[1:250,:,:] - nst_fut_detectsmooth_std2_utz[0:249,:,:]
nst_fut_yearchange_rang_utz[1:250,:,:] = nst_fut_detectsmooth_rang_utz[1:250,:,:] - nst_fut_detectsmooth_rang_utz[0:249,:,:]

d15n_no3_fut_yearchange_std1_ez = np.zeros((250,180,360))
d15n_no3_fut_yearchange_std2_ez = np.zeros((250,180,360))
d15n_no3_fut_yearchange_rang_ez = np.zeros((250,180,360))
d15n_no3_fut_yearchange_std1_utz = np.zeros((250,180,360))
d15n_no3_fut_yearchange_std2_utz = np.zeros((250,180,360))
d15n_no3_fut_yearchange_rang_utz = np.zeros((250,180,360))

d15n_no3_fut_yearchange_std1_ez[1:250,:,:] = d15n_no3_fut_detectsmooth_std1_ez[1:250,:,:] - d15n_no3_fut_detectsmooth_std1_ez[0:249,:,:]
d15n_no3_fut_yearchange_std2_ez[1:250,:,:] = d15n_no3_fut_detectsmooth_std2_ez[1:250,:,:] - d15n_no3_fut_detectsmooth_std2_ez[0:249,:,:]
d15n_no3_fut_yearchange_rang_ez[1:250,:,:] = d15n_no3_fut_detectsmooth_rang_ez[1:250,:,:] - d15n_no3_fut_detectsmooth_rang_ez[0:249,:,:]
d15n_no3_fut_yearchange_std1_utz[1:250,:,:] = d15n_no3_fut_detectsmooth_std1_utz[1:250,:,:] - d15n_no3_fut_detectsmooth_std1_utz[0:249,:,:]
d15n_no3_fut_yearchange_std2_utz[1:250,:,:] = d15n_no3_fut_detectsmooth_std2_utz[1:250,:,:] - d15n_no3_fut_detectsmooth_std2_utz[0:249,:,:]
d15n_no3_fut_yearchange_rang_utz[1:250,:,:] = d15n_no3_fut_detectsmooth_rang_utz[1:250,:,:] - d15n_no3_fut_detectsmooth_rang_utz[0:249,:,:]

d15n_pom_fut_yearchange_std1_ez = np.zeros((250,180,360))
d15n_pom_fut_yearchange_std2_ez = np.zeros((250,180,360))
d15n_pom_fut_yearchange_rang_ez = np.zeros((250,180,360))
d15n_pom_fut_yearchange_std1_utz = np.zeros((250,180,360))
d15n_pom_fut_yearchange_std2_utz = np.zeros((250,180,360))
d15n_pom_fut_yearchange_rang_utz = np.zeros((250,180,360))

d15n_pom_fut_yearchange_std1_ez[1:250,:,:] = d15n_pom_fut_detectsmooth_std1_ez[1:250,:,:] - d15n_pom_fut_detectsmooth_std1_ez[0:249,:,:]
d15n_pom_fut_yearchange_std2_ez[1:250,:,:] = d15n_pom_fut_detectsmooth_std2_ez[1:250,:,:] - d15n_pom_fut_detectsmooth_std2_ez[0:249,:,:]
d15n_pom_fut_yearchange_rang_ez[1:250,:,:] = d15n_pom_fut_detectsmooth_rang_ez[1:250,:,:] - d15n_pom_fut_detectsmooth_rang_ez[0:249,:,:]
d15n_pom_fut_yearchange_std1_utz[1:250,:,:] = d15n_pom_fut_detectsmooth_std1_utz[1:250,:,:] - d15n_pom_fut_detectsmooth_std1_utz[0:249,:,:]
d15n_pom_fut_yearchange_std2_utz[1:250,:,:] = d15n_pom_fut_detectsmooth_std2_utz[1:250,:,:] - d15n_pom_fut_detectsmooth_std2_utz[0:249,:,:]
d15n_pom_fut_yearchange_rang_utz[1:250,:,:] = d15n_pom_fut_detectsmooth_rang_utz[1:250,:,:] - d15n_pom_fut_detectsmooth_rang_utz[0:249,:,:]


no3_futndep_yearchange_std1_ez = np.zeros((250,180,360))
no3_futndep_yearchange_std2_ez = np.zeros((250,180,360))
no3_futndep_yearchange_rang_ez = np.zeros((250,180,360))
no3_futndep_yearchange_std1_utz = np.zeros((250,180,360))
no3_futndep_yearchange_std2_utz = np.zeros((250,180,360))
no3_futndep_yearchange_rang_utz = np.zeros((250,180,360))

no3_futndep_yearchange_std1_ez[1:250,:,:] = no3_futndep_detectsmooth_std1_ez[1:250,:,:] - no3_futndep_detectsmooth_std1_ez[0:249,:,:]
no3_futndep_yearchange_std2_ez[1:250,:,:] = no3_futndep_detectsmooth_std2_ez[1:250,:,:] - no3_futndep_detectsmooth_std2_ez[0:249,:,:]
no3_futndep_yearchange_rang_ez[1:250,:,:] = no3_futndep_detectsmooth_rang_ez[1:250,:,:] - no3_futndep_detectsmooth_rang_ez[0:249,:,:]
no3_futndep_yearchange_std1_utz[1:250,:,:] = no3_futndep_detectsmooth_std1_utz[1:250,:,:] - no3_futndep_detectsmooth_std1_utz[0:249,:,:]
no3_futndep_yearchange_std2_utz[1:250,:,:] = no3_futndep_detectsmooth_std2_utz[1:250,:,:] - no3_futndep_detectsmooth_std2_utz[0:249,:,:]
no3_futndep_yearchange_rang_utz[1:250,:,:] = no3_futndep_detectsmooth_rang_utz[1:250,:,:] - no3_futndep_detectsmooth_rang_utz[0:249,:,:]

nst_futndep_yearchange_std1_ez = np.zeros((250,180,360))
nst_futndep_yearchange_std2_ez = np.zeros((250,180,360))
nst_futndep_yearchange_rang_ez = np.zeros((250,180,360))
nst_futndep_yearchange_std1_utz = np.zeros((250,180,360))
nst_futndep_yearchange_std2_utz = np.zeros((250,180,360))
nst_futndep_yearchange_rang_utz = np.zeros((250,180,360))

nst_futndep_yearchange_std1_ez[1:250,:,:] = nst_futndep_detectsmooth_std1_ez[1:250,:,:] - nst_futndep_detectsmooth_std1_ez[0:249,:,:]
nst_futndep_yearchange_std2_ez[1:250,:,:] = nst_futndep_detectsmooth_std2_ez[1:250,:,:] - nst_futndep_detectsmooth_std2_ez[0:249,:,:]
nst_futndep_yearchange_rang_ez[1:250,:,:] = nst_futndep_detectsmooth_rang_ez[1:250,:,:] - nst_futndep_detectsmooth_rang_ez[0:249,:,:]
nst_futndep_yearchange_std1_utz[1:250,:,:] = nst_futndep_detectsmooth_std1_utz[1:250,:,:] - nst_futndep_detectsmooth_std1_utz[0:249,:,:]
nst_futndep_yearchange_std2_utz[1:250,:,:] = nst_futndep_detectsmooth_std2_utz[1:250,:,:] - nst_futndep_detectsmooth_std2_utz[0:249,:,:]
nst_futndep_yearchange_rang_utz[1:250,:,:] = nst_futndep_detectsmooth_rang_utz[1:250,:,:] - nst_futndep_detectsmooth_rang_utz[0:249,:,:]

d15n_no3_futndep_yearchange_std1_ez = np.zeros((250,180,360))
d15n_no3_futndep_yearchange_std2_ez = np.zeros((250,180,360))
d15n_no3_futndep_yearchange_rang_ez = np.zeros((250,180,360))
d15n_no3_futndep_yearchange_std1_utz = np.zeros((250,180,360))
d15n_no3_futndep_yearchange_std2_utz = np.zeros((250,180,360))
d15n_no3_futndep_yearchange_rang_utz = np.zeros((250,180,360))

d15n_no3_futndep_yearchange_std1_ez[1:250,:,:] = d15n_no3_futndep_detectsmooth_std1_ez[1:250,:,:] - d15n_no3_futndep_detectsmooth_std1_ez[0:249,:,:]
d15n_no3_futndep_yearchange_std2_ez[1:250,:,:] = d15n_no3_futndep_detectsmooth_std2_ez[1:250,:,:] - d15n_no3_futndep_detectsmooth_std2_ez[0:249,:,:]
d15n_no3_futndep_yearchange_rang_ez[1:250,:,:] = d15n_no3_futndep_detectsmooth_rang_ez[1:250,:,:] - d15n_no3_futndep_detectsmooth_rang_ez[0:249,:,:]
d15n_no3_futndep_yearchange_std1_utz[1:250,:,:] = d15n_no3_futndep_detectsmooth_std1_utz[1:250,:,:] - d15n_no3_futndep_detectsmooth_std1_utz[0:249,:,:]
d15n_no3_futndep_yearchange_std2_utz[1:250,:,:] = d15n_no3_futndep_detectsmooth_std2_utz[1:250,:,:] - d15n_no3_futndep_detectsmooth_std2_utz[0:249,:,:]
d15n_no3_futndep_yearchange_rang_utz[1:250,:,:] = d15n_no3_futndep_detectsmooth_rang_utz[1:250,:,:] - d15n_no3_futndep_detectsmooth_rang_utz[0:249,:,:]

d15n_pom_futndep_yearchange_std1_ez = np.zeros((250,180,360))
d15n_pom_futndep_yearchange_std2_ez = np.zeros((250,180,360))
d15n_pom_futndep_yearchange_rang_ez = np.zeros((250,180,360))
d15n_pom_futndep_yearchange_std1_utz = np.zeros((250,180,360))
d15n_pom_futndep_yearchange_std2_utz = np.zeros((250,180,360))
d15n_pom_futndep_yearchange_rang_utz = np.zeros((250,180,360))

d15n_pom_futndep_yearchange_std1_ez[1:250,:,:] = d15n_pom_futndep_detectsmooth_std1_ez[1:250,:,:] - d15n_pom_futndep_detectsmooth_std1_ez[0:249,:,:]
d15n_pom_futndep_yearchange_std2_ez[1:250,:,:] = d15n_pom_futndep_detectsmooth_std2_ez[1:250,:,:] - d15n_pom_futndep_detectsmooth_std2_ez[0:249,:,:]
d15n_pom_futndep_yearchange_rang_ez[1:250,:,:] = d15n_pom_futndep_detectsmooth_rang_ez[1:250,:,:] - d15n_pom_futndep_detectsmooth_rang_ez[0:249,:,:]
d15n_pom_futndep_yearchange_std1_utz[1:250,:,:] = d15n_pom_futndep_detectsmooth_std1_utz[1:250,:,:] - d15n_pom_futndep_detectsmooth_std1_utz[0:249,:,:]
d15n_pom_futndep_yearchange_std2_utz[1:250,:,:] = d15n_pom_futndep_detectsmooth_std2_utz[1:250,:,:] - d15n_pom_futndep_detectsmooth_std2_utz[0:249,:,:]
d15n_pom_futndep_yearchange_rang_utz[1:250,:,:] = d15n_pom_futndep_detectsmooth_rang_utz[1:250,:,:] - d15n_pom_futndep_detectsmooth_rang_utz[0:249,:,:]


# return timeseries with only 1s and 0s, where the first 1 is the ToE, 
# and where any negative differences cause all prior times to be 0

fix_picndep_toe_std1 = np.zeros((250,180,360))
fix_picndep_toe_std2 = np.zeros((250,180,360))
fix_picndep_toe_rang = np.zeros((250,180,360))
fix_fut_toe_std1 = np.zeros((250,180,360))
fix_fut_toe_std2 = np.zeros((250,180,360))
fix_fut_toe_rang = np.zeros((250,180,360))
fix_futndep_toe_std1 = np.zeros((250,180,360))
fix_futndep_toe_std2 = np.zeros((250,180,360))
fix_futndep_toe_rang = np.zeros((250,180,360))

npp_picndep_toe_std1 = np.zeros((250,180,360))
npp_picndep_toe_std2 = np.zeros((250,180,360))
npp_picndep_toe_rang = np.zeros((250,180,360))
npp_fut_toe_std1 = np.zeros((250,180,360))
npp_fut_toe_std2 = np.zeros((250,180,360))
npp_fut_toe_rang = np.zeros((250,180,360))
npp_futndep_toe_std1 = np.zeros((250,180,360))
npp_futndep_toe_std2 = np.zeros((250,180,360))
npp_futndep_toe_rang = np.zeros((250,180,360))

temp_fut_toe_std1_ez = np.zeros((250,180,360))
temp_fut_toe_std2_ez = np.zeros((250,180,360))
temp_fut_toe_rang_ez = np.zeros((250,180,360))
temp_fut_toe_std1_utz = np.zeros((250,180,360))
temp_fut_toe_std2_utz = np.zeros((250,180,360))
temp_fut_toe_rang_utz = np.zeros((250,180,360))


no3_picndep_toe_std1_ez = np.zeros((250,180,360))
no3_picndep_toe_std2_ez = np.zeros((250,180,360))
no3_picndep_toe_rang_ez = np.zeros((250,180,360))
no3_picndep_toe_std1_utz = np.zeros((250,180,360))
no3_picndep_toe_std2_utz = np.zeros((250,180,360))
no3_picndep_toe_rang_utz = np.zeros((250,180,360))

nst_picndep_toe_std1_ez = np.zeros((250,180,360))
nst_picndep_toe_std2_ez = np.zeros((250,180,360))
nst_picndep_toe_rang_ez = np.zeros((250,180,360))
nst_picndep_toe_std1_utz = np.zeros((250,180,360))
nst_picndep_toe_std2_utz = np.zeros((250,180,360))
nst_picndep_toe_rang_utz = np.zeros((250,180,360))

d15n_no3_picndep_toe_std1_ez = np.zeros((250,180,360))
d15n_no3_picndep_toe_std2_ez = np.zeros((250,180,360))
d15n_no3_picndep_toe_rang_ez = np.zeros((250,180,360))
d15n_no3_picndep_toe_std1_utz = np.zeros((250,180,360))
d15n_no3_picndep_toe_std2_utz = np.zeros((250,180,360))
d15n_no3_picndep_toe_rang_utz = np.zeros((250,180,360))

d15n_pom_picndep_toe_std1_ez = np.zeros((250,180,360))
d15n_pom_picndep_toe_std2_ez = np.zeros((250,180,360))
d15n_pom_picndep_toe_rang_ez = np.zeros((250,180,360))
d15n_pom_picndep_toe_std1_utz = np.zeros((250,180,360))
d15n_pom_picndep_toe_std2_utz = np.zeros((250,180,360))
d15n_pom_picndep_toe_rang_utz = np.zeros((250,180,360))


no3_fut_toe_std1_ez = np.zeros((250,180,360))
no3_fut_toe_std2_ez = np.zeros((250,180,360))
no3_fut_toe_rang_ez = np.zeros((250,180,360))
no3_fut_toe_std1_utz = np.zeros((250,180,360))
no3_fut_toe_std2_utz = np.zeros((250,180,360))
no3_fut_toe_rang_utz = np.zeros((250,180,360))

nst_fut_toe_std1_ez = np.zeros((250,180,360))
nst_fut_toe_std2_ez = np.zeros((250,180,360))
nst_fut_toe_rang_ez = np.zeros((250,180,360))
nst_fut_toe_std1_utz = np.zeros((250,180,360))
nst_fut_toe_std2_utz = np.zeros((250,180,360))
nst_fut_toe_rang_utz = np.zeros((250,180,360))

d15n_no3_fut_toe_std1_ez = np.zeros((250,180,360))
d15n_no3_fut_toe_std2_ez = np.zeros((250,180,360))
d15n_no3_fut_toe_rang_ez = np.zeros((250,180,360))
d15n_no3_fut_toe_std1_utz = np.zeros((250,180,360))
d15n_no3_fut_toe_std2_utz = np.zeros((250,180,360))
d15n_no3_fut_toe_rang_utz = np.zeros((250,180,360))

d15n_pom_fut_toe_std1_ez = np.zeros((250,180,360))
d15n_pom_fut_toe_std2_ez = np.zeros((250,180,360))
d15n_pom_fut_toe_rang_ez = np.zeros((250,180,360))
d15n_pom_fut_toe_std1_utz = np.zeros((250,180,360))
d15n_pom_fut_toe_std2_utz = np.zeros((250,180,360))
d15n_pom_fut_toe_rang_utz = np.zeros((250,180,360))


no3_futndep_toe_std1_ez = np.zeros((250,180,360))
no3_futndep_toe_std2_ez = np.zeros((250,180,360))
no3_futndep_toe_rang_ez = np.zeros((250,180,360))
no3_futndep_toe_std1_utz = np.zeros((250,180,360))
no3_futndep_toe_std2_utz = np.zeros((250,180,360))
no3_futndep_toe_rang_utz = np.zeros((250,180,360))

nst_futndep_toe_std1_ez = np.zeros((250,180,360))
nst_futndep_toe_std2_ez = np.zeros((250,180,360))
nst_futndep_toe_rang_ez = np.zeros((250,180,360))
nst_futndep_toe_std1_utz = np.zeros((250,180,360))
nst_futndep_toe_std2_utz = np.zeros((250,180,360))
nst_futndep_toe_rang_utz = np.zeros((250,180,360))

d15n_no3_futndep_toe_std1_ez = np.zeros((250,180,360))
d15n_no3_futndep_toe_std2_ez = np.zeros((250,180,360))
d15n_no3_futndep_toe_rang_ez = np.zeros((250,180,360))
d15n_no3_futndep_toe_std1_utz = np.zeros((250,180,360))
d15n_no3_futndep_toe_std2_utz = np.zeros((250,180,360))
d15n_no3_futndep_toe_rang_utz = np.zeros((250,180,360))

d15n_pom_futndep_toe_std1_ez = np.zeros((250,180,360))
d15n_pom_futndep_toe_std2_ez = np.zeros((250,180,360))
d15n_pom_futndep_toe_rang_ez = np.zeros((250,180,360))
d15n_pom_futndep_toe_std1_utz = np.zeros((250,180,360))
d15n_pom_futndep_toe_std2_utz = np.zeros((250,180,360))
d15n_pom_futndep_toe_rang_utz = np.zeros((250,180,360))



for i in tqdm(np.arange(len(no3_pic_ez[0,0,:])), desc="longitudes", position=2):
    for j in np.arange(len(no3_pic_ez[0,:,0])):
        for t in np.arange(250):
            if fix_picndep_detectsmooth_std1[t,j,i] == 1:
                fix_picndep_toe_std1[t,j,i] = 1
            if fix_picndep_yearchange_std1[t,j,i] < 0:
                fix_picndep_toe_std1[0:t,j,i] = 0  
            if fix_picndep_detectsmooth_std2[t,j,i] == 1:
                fix_picndep_toe_std2[t,j,i] = 1
            if fix_picndep_yearchange_std2[t,j,i] < 0:
                fix_picndep_toe_std2[0:t,j,i] = 0  
            if fix_picndep_detectsmooth_rang[t,j,i] == 1:
                fix_picndep_toe_rang[t,j,i] = 1
            if fix_picndep_yearchange_rang[t,j,i] < 0:
                fix_picndep_toe_rang[0:t,j,i] = 0  
            if fix_fut_detectsmooth_std1[t,j,i] == 1:
                fix_fut_toe_std1[t,j,i] = 1
            if fix_fut_yearchange_std1[t,j,i] < 0:
                fix_fut_toe_std1[0:t,j,i] = 0  
            if fix_fut_detectsmooth_std2[t,j,i] == 1:
                fix_fut_toe_std2[t,j,i] = 1
            if fix_fut_yearchange_std2[t,j,i] < 0:
                fix_fut_toe_std2[0:t,j,i] = 0  
            if fix_fut_detectsmooth_rang[t,j,i] == 1:
                fix_fut_toe_rang[t,j,i] = 1
            if fix_fut_yearchange_rang[t,j,i] < 0:
                fix_fut_toe_rang[0:t,j,i] = 0  
            if fix_futndep_detectsmooth_std1[t,j,i] == 1:
                fix_futndep_toe_std1[t,j,i] = 1
            if fix_futndep_yearchange_std1[t,j,i] < 0:
                fix_futndep_toe_std1[0:t,j,i] = 0  
            if fix_futndep_detectsmooth_std2[t,j,i] == 1:
                fix_futndep_toe_std2[t,j,i] = 1
            if fix_futndep_yearchange_std2[t,j,i] < 0:
                fix_futndep_toe_std2[0:t,j,i] = 0  
            if fix_futndep_detectsmooth_rang[t,j,i] == 1:
                fix_futndep_toe_rang[t,j,i] = 1
            if fix_futndep_yearchange_rang[t,j,i] < 0:
                fix_futndep_toe_rang[0:t,j,i] = 0  
    
            if npp_picndep_detectsmooth_std1[t,j,i] == 1:
                npp_picndep_toe_std1[t,j,i] = 1
            if npp_picndep_yearchange_std1[t,j,i] < 0:
                npp_picndep_toe_std1[0:t,j,i] = 0  
            if npp_picndep_detectsmooth_std2[t,j,i] == 1:
                npp_picndep_toe_std2[t,j,i] = 1
            if npp_picndep_yearchange_std2[t,j,i] < 0:
                npp_picndep_toe_std2[0:t,j,i] = 0  
            if npp_picndep_detectsmooth_rang[t,j,i] == 1:
                npp_picndep_toe_rang[t,j,i] = 1
            if npp_picndep_yearchange_rang[t,j,i] < 0:
                npp_picndep_toe_rang[0:t,j,i] = 0  
            if npp_fut_detectsmooth_std1[t,j,i] == 1:
                npp_fut_toe_std1[t,j,i] = 1
            if npp_fut_yearchange_std1[t,j,i] < 0:
                npp_fut_toe_std1[0:t,j,i] = 0  
            if npp_fut_detectsmooth_std2[t,j,i] == 1:
                npp_fut_toe_std2[t,j,i] = 1
            if npp_fut_yearchange_std2[t,j,i] < 0:
                npp_fut_toe_std2[0:t,j,i] = 0  
            if npp_fut_detectsmooth_rang[t,j,i] == 1:
                npp_fut_toe_rang[t,j,i] = 1
            if npp_fut_yearchange_rang[t,j,i] < 0:
                npp_fut_toe_rang[0:t,j,i] = 0  
            if npp_futndep_detectsmooth_std1[t,j,i] == 1:
                npp_futndep_toe_std1[t,j,i] = 1
            if npp_futndep_yearchange_std1[t,j,i] < 0:
                npp_futndep_toe_std1[0:t,j,i] = 0  
            if npp_futndep_detectsmooth_std2[t,j,i] == 1:
                npp_futndep_toe_std2[t,j,i] = 1
            if npp_futndep_yearchange_std2[t,j,i] < 0:
                npp_futndep_toe_std2[0:t,j,i] = 0  
            if npp_futndep_detectsmooth_rang[t,j,i] == 1:
                npp_futndep_toe_rang[t,j,i] = 1
            if npp_futndep_yearchange_rang[t,j,i] < 0:
                npp_futndep_toe_rang[0:t,j,i] = 0  
    
            if temp_fut_detectsmooth_std1_ez[t,j,i] == 1:
                temp_fut_toe_std1_ez[t,j,i] = 1
            if temp_fut_yearchange_std1_ez[t,j,i] < 0:
                temp_fut_toe_std1_ez[0:t,j,i] = 0  
            if temp_fut_detectsmooth_std2_ez[t,j,i] == 1:
                temp_fut_toe_std2_ez[t,j,i] = 1
            if temp_fut_yearchange_std2_ez[t,j,i] < 0:
                temp_fut_toe_std2_ez[0:t,j,i] = 0  
            if temp_fut_detectsmooth_rang_ez[t,j,i] == 1:
                temp_fut_toe_rang_ez[t,j,i] = 1
            if temp_fut_yearchange_rang_ez[t,j,i] < 0:
                temp_fut_toe_rang_ez[0:t,j,i] = 0  
    
            if temp_fut_detectsmooth_std1_utz[t,j,i] == 1:
                temp_fut_toe_std1_utz[t,j,i] = 1
            if temp_fut_yearchange_std1_utz[t,j,i] < 0:
                temp_fut_toe_std1_utz[0:t,j,i] = 0  
            if temp_fut_detectsmooth_std2_utz[t,j,i] == 1:
                temp_fut_toe_std2_utz[t,j,i] = 1
            if temp_fut_yearchange_std2_utz[t,j,i] < 0:
                temp_fut_toe_std2_utz[0:t,j,i] = 0  
            if temp_fut_detectsmooth_rang_utz[t,j,i] == 1:
                temp_fut_toe_rang_utz[t,j,i] = 1
            if temp_fut_yearchange_rang_utz[t,j,i] < 0:
                temp_fut_toe_rang_utz[0:t,j,i] = 0  
    
            
            if no3_picndep_detectsmooth_std1_ez[t,j,i] == 1:
                no3_picndep_toe_std1_ez[t,j,i] = 1
            if no3_picndep_yearchange_std1_ez[t,j,i] < 0:
                no3_picndep_toe_std1_ez[0:t,j,i] = 0  
            if no3_picndep_detectsmooth_std2_ez[t,j,i] == 1:
                no3_picndep_toe_std2_ez[t,j,i] = 1
            if no3_picndep_yearchange_std2_ez[t,j,i] < 0:
                no3_picndep_toe_std2_ez[0:t,j,i] = 0  
            if no3_picndep_detectsmooth_rang_ez[t,j,i] == 1:
                no3_picndep_toe_rang_ez[t,j,i] = 1
            if no3_picndep_yearchange_rang_ez[t,j,i] < 0:
                no3_picndep_toe_rang_ez[0:t,j,i] = 0  
    
            if no3_picndep_detectsmooth_std1_utz[t,j,i] == 1:
                no3_picndep_toe_std1_utz[t,j,i] = 1
            if no3_picndep_yearchange_std1_utz[t,j,i] < 0:
                no3_picndep_toe_std1_utz[0:t,j,i] = 0  
            if no3_picndep_detectsmooth_std2_utz[t,j,i] == 1:
                no3_picndep_toe_std2_utz[t,j,i] = 1
            if no3_picndep_yearchange_std2_utz[t,j,i] < 0:
                no3_picndep_toe_std2_utz[0:t,j,i] = 0  
            if no3_picndep_detectsmooth_rang_utz[t,j,i] == 1:
                no3_picndep_toe_rang_utz[t,j,i] = 1
            if no3_picndep_yearchange_rang_utz[t,j,i] < 0:
                no3_picndep_toe_rang_utz[0:t,j,i] = 0  
    

            if nst_picndep_detectsmooth_std1_ez[t,j,i] == 1:
                nst_picndep_toe_std1_ez[t,j,i] = 1
            if nst_picndep_yearchange_std1_ez[t,j,i] < 0:
                nst_picndep_toe_std1_ez[0:t,j,i] = 0  
            if nst_picndep_detectsmooth_std2_ez[t,j,i] == 1:
                nst_picndep_toe_std2_ez[t,j,i] = 1
            if nst_picndep_yearchange_std2_ez[t,j,i] < 0:
                nst_picndep_toe_std2_ez[0:t,j,i] = 0  
            if nst_picndep_detectsmooth_rang_ez[t,j,i] == 1:
                nst_picndep_toe_rang_ez[t,j,i] = 1
            if nst_picndep_yearchange_rang_ez[t,j,i] < 0:
                nst_picndep_toe_rang_ez[0:t,j,i] = 0  
    
            if nst_picndep_detectsmooth_std1_utz[t,j,i] == 1:
                nst_picndep_toe_std1_utz[t,j,i] = 1
            if nst_picndep_yearchange_std1_utz[t,j,i] < 0:
                nst_picndep_toe_std1_utz[0:t,j,i] = 0  
            if nst_picndep_detectsmooth_std2_utz[t,j,i] == 1:
                nst_picndep_toe_std2_utz[t,j,i] = 1
            if nst_picndep_yearchange_std2_utz[t,j,i] < 0:
                nst_picndep_toe_std2_utz[0:t,j,i] = 0  
            if nst_picndep_detectsmooth_rang_utz[t,j,i] == 1:
                nst_picndep_toe_rang_utz[t,j,i] = 1
            if nst_picndep_yearchange_rang_utz[t,j,i] < 0:
                nst_picndep_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_no3_picndep_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_no3_picndep_toe_std1_ez[t,j,i] = 1
            if d15n_no3_picndep_yearchange_std1_ez[t,j,i] < 0:
                d15n_no3_picndep_toe_std1_ez[0:t,j,i] = 0  
            if d15n_no3_picndep_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_no3_picndep_toe_std2_ez[t,j,i] = 1
            if d15n_no3_picndep_yearchange_std2_ez[t,j,i] < 0:
                d15n_no3_picndep_toe_std2_ez[0:t,j,i] = 0  
            if d15n_no3_picndep_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_no3_picndep_toe_rang_ez[t,j,i] = 1
            if d15n_no3_picndep_yearchange_rang_ez[t,j,i] < 0:
                d15n_no3_picndep_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_no3_picndep_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_no3_picndep_toe_std1_utz[t,j,i] = 1
            if d15n_no3_picndep_yearchange_std1_utz[t,j,i] < 0:
                d15n_no3_picndep_toe_std1_utz[0:t,j,i] = 0  
            if d15n_no3_picndep_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_no3_picndep_toe_std2_utz[t,j,i] = 1
            if d15n_no3_picndep_yearchange_std2_utz[t,j,i] < 0:
                d15n_no3_picndep_toe_std2_utz[0:t,j,i] = 0  
            if d15n_no3_picndep_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_no3_picndep_toe_rang_utz[t,j,i] = 1
            if d15n_no3_picndep_yearchange_rang_utz[t,j,i] < 0:
                d15n_no3_picndep_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_pom_picndep_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_pom_picndep_toe_std1_ez[t,j,i] = 1
            if d15n_pom_picndep_yearchange_std1_ez[t,j,i] < 0:
                d15n_pom_picndep_toe_std1_ez[0:t,j,i] = 0  
            if d15n_pom_picndep_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_pom_picndep_toe_std2_ez[t,j,i] = 1
            if d15n_pom_picndep_yearchange_std2_ez[t,j,i] < 0:
                d15n_pom_picndep_toe_std2_ez[0:t,j,i] = 0  
            if d15n_pom_picndep_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_pom_picndep_toe_rang_ez[t,j,i] = 1
            if d15n_pom_picndep_yearchange_rang_ez[t,j,i] < 0:
                d15n_pom_picndep_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_pom_picndep_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_pom_picndep_toe_std1_utz[t,j,i] = 1
            if d15n_pom_picndep_yearchange_std1_utz[t,j,i] < 0:
                d15n_pom_picndep_toe_std1_utz[0:t,j,i] = 0  
            if d15n_pom_picndep_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_pom_picndep_toe_std2_utz[t,j,i] = 1
            if d15n_pom_picndep_yearchange_std2_utz[t,j,i] < 0:
                d15n_pom_picndep_toe_std2_utz[0:t,j,i] = 0  
            if d15n_pom_picndep_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_pom_picndep_toe_rang_utz[t,j,i] = 1
            if d15n_pom_picndep_yearchange_rang_utz[t,j,i] < 0:
                d15n_pom_picndep_toe_rang_utz[0:t,j,i] = 0  
    


            if no3_fut_detectsmooth_std1_ez[t,j,i] == 1:
                no3_fut_toe_std1_ez[t,j,i] = 1
            if no3_fut_yearchange_std1_ez[t,j,i] < 0:
                no3_fut_toe_std1_ez[0:t,j,i] = 0  
            if no3_fut_detectsmooth_std2_ez[t,j,i] == 1:
                no3_fut_toe_std2_ez[t,j,i] = 1
            if no3_fut_yearchange_std2_ez[t,j,i] < 0:
                no3_fut_toe_std2_ez[0:t,j,i] = 0  
            if no3_fut_detectsmooth_rang_ez[t,j,i] == 1:
                no3_fut_toe_rang_ez[t,j,i] = 1
            if no3_fut_yearchange_rang_ez[t,j,i] < 0:
                no3_fut_toe_rang_ez[0:t,j,i] = 0  
    
            if no3_fut_detectsmooth_std1_utz[t,j,i] == 1:
                no3_fut_toe_std1_utz[t,j,i] = 1
            if no3_fut_yearchange_std1_utz[t,j,i] < 0:
                no3_fut_toe_std1_utz[0:t,j,i] = 0  
            if no3_fut_detectsmooth_std2_utz[t,j,i] == 1:
                no3_fut_toe_std2_utz[t,j,i] = 1
            if no3_fut_yearchange_std2_utz[t,j,i] < 0:
                no3_fut_toe_std2_utz[0:t,j,i] = 0  
            if no3_fut_detectsmooth_rang_utz[t,j,i] == 1:
                no3_fut_toe_rang_utz[t,j,i] = 1
            if no3_fut_yearchange_rang_utz[t,j,i] < 0:
                no3_fut_toe_rang_utz[0:t,j,i] = 0  
    

            if nst_fut_detectsmooth_std1_ez[t,j,i] == 1:
                nst_fut_toe_std1_ez[t,j,i] = 1
            if nst_fut_yearchange_std1_ez[t,j,i] < 0:
                nst_fut_toe_std1_ez[0:t,j,i] = 0  
            if nst_fut_detectsmooth_std2_ez[t,j,i] == 1:
                nst_fut_toe_std2_ez[t,j,i] = 1
            if nst_fut_yearchange_std2_ez[t,j,i] < 0:
                nst_fut_toe_std2_ez[0:t,j,i] = 0  
            if nst_fut_detectsmooth_rang_ez[t,j,i] == 1:
                nst_fut_toe_rang_ez[t,j,i] = 1
            if nst_fut_yearchange_rang_ez[t,j,i] < 0:
                nst_fut_toe_rang_ez[0:t,j,i] = 0  
    
            if nst_fut_detectsmooth_std1_utz[t,j,i] == 1:
                nst_fut_toe_std1_utz[t,j,i] = 1
            if nst_fut_yearchange_std1_utz[t,j,i] < 0:
                nst_fut_toe_std1_utz[0:t,j,i] = 0  
            if nst_fut_detectsmooth_std2_utz[t,j,i] == 1:
                nst_fut_toe_std2_utz[t,j,i] = 1
            if nst_fut_yearchange_std2_utz[t,j,i] < 0:
                nst_fut_toe_std2_utz[0:t,j,i] = 0  
            if nst_fut_detectsmooth_rang_utz[t,j,i] == 1:
                nst_fut_toe_rang_utz[t,j,i] = 1
            if nst_fut_yearchange_rang_utz[t,j,i] < 0:
                nst_fut_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_no3_fut_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_no3_fut_toe_std1_ez[t,j,i] = 1
            if d15n_no3_fut_yearchange_std1_ez[t,j,i] < 0:
                d15n_no3_fut_toe_std1_ez[0:t,j,i] = 0  
            if d15n_no3_fut_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_no3_fut_toe_std2_ez[t,j,i] = 1
            if d15n_no3_fut_yearchange_std2_ez[t,j,i] < 0:
                d15n_no3_fut_toe_std2_ez[0:t,j,i] = 0  
            if d15n_no3_fut_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_no3_fut_toe_rang_ez[t,j,i] = 1
            if d15n_no3_fut_yearchange_rang_ez[t,j,i] < 0:
                d15n_no3_fut_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_no3_fut_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_no3_fut_toe_std1_utz[t,j,i] = 1
            if d15n_no3_fut_yearchange_std1_utz[t,j,i] < 0:
                d15n_no3_fut_toe_std1_utz[0:t,j,i] = 0  
            if d15n_no3_fut_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_no3_fut_toe_std2_utz[t,j,i] = 1
            if d15n_no3_fut_yearchange_std2_utz[t,j,i] < 0:
                d15n_no3_fut_toe_std2_utz[0:t,j,i] = 0  
            if d15n_no3_fut_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_no3_fut_toe_rang_utz[t,j,i] = 1
            if d15n_no3_fut_yearchange_rang_utz[t,j,i] < 0:
                d15n_no3_fut_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_pom_fut_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_pom_fut_toe_std1_ez[t,j,i] = 1
            if d15n_pom_fut_yearchange_std1_ez[t,j,i] < 0:
                d15n_pom_fut_toe_std1_ez[0:t,j,i] = 0  
            if d15n_pom_fut_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_pom_fut_toe_std2_ez[t,j,i] = 1
            if d15n_pom_fut_yearchange_std2_ez[t,j,i] < 0:
                d15n_pom_fut_toe_std2_ez[0:t,j,i] = 0  
            if d15n_pom_fut_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_pom_fut_toe_rang_ez[t,j,i] = 1
            if d15n_pom_fut_yearchange_rang_ez[t,j,i] < 0:
                d15n_pom_fut_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_pom_fut_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_pom_fut_toe_std1_utz[t,j,i] = 1
            if d15n_pom_fut_yearchange_std1_utz[t,j,i] < 0:
                d15n_pom_fut_toe_std1_utz[0:t,j,i] = 0  
            if d15n_pom_fut_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_pom_fut_toe_std2_utz[t,j,i] = 1
            if d15n_pom_fut_yearchange_std2_utz[t,j,i] < 0:
                d15n_pom_fut_toe_std2_utz[0:t,j,i] = 0  
            if d15n_pom_fut_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_pom_fut_toe_rang_utz[t,j,i] = 1
            if d15n_pom_fut_yearchange_rang_utz[t,j,i] < 0:
                d15n_pom_fut_toe_rang_utz[0:t,j,i] = 0  
    

            if no3_futndep_detectsmooth_std1_ez[t,j,i] == 1:
                no3_futndep_toe_std1_ez[t,j,i] = 1
            if no3_futndep_yearchange_std1_ez[t,j,i] < 0:
                no3_futndep_toe_std1_ez[0:t,j,i] = 0  
            if no3_futndep_detectsmooth_std2_ez[t,j,i] == 1:
                no3_futndep_toe_std2_ez[t,j,i] = 1
            if no3_futndep_yearchange_std2_ez[t,j,i] < 0:
                no3_futndep_toe_std2_ez[0:t,j,i] = 0  
            if no3_futndep_detectsmooth_rang_ez[t,j,i] == 1:
                no3_futndep_toe_rang_ez[t,j,i] = 1
            if no3_futndep_yearchange_rang_ez[t,j,i] < 0:
                no3_futndep_toe_rang_ez[0:t,j,i] = 0  
    
            if no3_futndep_detectsmooth_std1_utz[t,j,i] == 1:
                no3_futndep_toe_std1_utz[t,j,i] = 1
            if no3_futndep_yearchange_std1_utz[t,j,i] < 0:
                no3_futndep_toe_std1_utz[0:t,j,i] = 0  
            if no3_futndep_detectsmooth_std2_utz[t,j,i] == 1:
                no3_futndep_toe_std2_utz[t,j,i] = 1
            if no3_futndep_yearchange_std2_utz[t,j,i] < 0:
                no3_futndep_toe_std2_utz[0:t,j,i] = 0  
            if no3_futndep_detectsmooth_rang_utz[t,j,i] == 1:
                no3_futndep_toe_rang_utz[t,j,i] = 1
            if no3_futndep_yearchange_rang_utz[t,j,i] < 0:
                no3_futndep_toe_rang_utz[0:t,j,i] = 0  
    

            if nst_futndep_detectsmooth_std1_ez[t,j,i] == 1:
                nst_futndep_toe_std1_ez[t,j,i] = 1
            if nst_futndep_yearchange_std1_ez[t,j,i] < 0:
                nst_futndep_toe_std1_ez[0:t,j,i] = 0  
            if nst_futndep_detectsmooth_std2_ez[t,j,i] == 1:
                nst_futndep_toe_std2_ez[t,j,i] = 1
            if nst_futndep_yearchange_std2_ez[t,j,i] < 0:
                nst_futndep_toe_std2_ez[0:t,j,i] = 0  
            if nst_futndep_detectsmooth_rang_ez[t,j,i] == 1:
                nst_futndep_toe_rang_ez[t,j,i] = 1
            if nst_futndep_yearchange_rang_ez[t,j,i] < 0:
                nst_futndep_toe_rang_ez[0:t,j,i] = 0  
    
            if nst_futndep_detectsmooth_std1_utz[t,j,i] == 1:
                nst_futndep_toe_std1_utz[t,j,i] = 1
            if nst_futndep_yearchange_std1_utz[t,j,i] < 0:
                nst_futndep_toe_std1_utz[0:t,j,i] = 0  
            if nst_futndep_detectsmooth_std2_utz[t,j,i] == 1:
                nst_futndep_toe_std2_utz[t,j,i] = 1
            if nst_futndep_yearchange_std2_utz[t,j,i] < 0:
                nst_futndep_toe_std2_utz[0:t,j,i] = 0  
            if nst_futndep_detectsmooth_rang_utz[t,j,i] == 1:
                nst_futndep_toe_rang_utz[t,j,i] = 1
            if nst_futndep_yearchange_rang_utz[t,j,i] < 0:
                nst_futndep_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_no3_futndep_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_no3_futndep_toe_std1_ez[t,j,i] = 1
            if d15n_no3_futndep_yearchange_std1_ez[t,j,i] < 0:
                d15n_no3_futndep_toe_std1_ez[0:t,j,i] = 0  
            if d15n_no3_futndep_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_no3_futndep_toe_std2_ez[t,j,i] = 1
            if d15n_no3_futndep_yearchange_std2_ez[t,j,i] < 0:
                d15n_no3_futndep_toe_std2_ez[0:t,j,i] = 0  
            if d15n_no3_futndep_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_no3_futndep_toe_rang_ez[t,j,i] = 1
            if d15n_no3_futndep_yearchange_rang_ez[t,j,i] < 0:
                d15n_no3_futndep_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_no3_futndep_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_no3_futndep_toe_std1_utz[t,j,i] = 1
            if d15n_no3_futndep_yearchange_std1_utz[t,j,i] < 0:
                d15n_no3_futndep_toe_std1_utz[0:t,j,i] = 0  
            if d15n_no3_futndep_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_no3_futndep_toe_std2_utz[t,j,i] = 1
            if d15n_no3_futndep_yearchange_std2_utz[t,j,i] < 0:
                d15n_no3_futndep_toe_std2_utz[0:t,j,i] = 0  
            if d15n_no3_futndep_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_no3_futndep_toe_rang_utz[t,j,i] = 1
            if d15n_no3_futndep_yearchange_rang_utz[t,j,i] < 0:
                d15n_no3_futndep_toe_rang_utz[0:t,j,i] = 0  
    

            if d15n_pom_futndep_detectsmooth_std1_ez[t,j,i] == 1:
                d15n_pom_futndep_toe_std1_ez[t,j,i] = 1
            if d15n_pom_futndep_yearchange_std1_ez[t,j,i] < 0:
                d15n_pom_futndep_toe_std1_ez[0:t,j,i] = 0  
            if d15n_pom_futndep_detectsmooth_std2_ez[t,j,i] == 1:
                d15n_pom_futndep_toe_std2_ez[t,j,i] = 1
            if d15n_pom_futndep_yearchange_std2_ez[t,j,i] < 0:
                d15n_pom_futndep_toe_std2_ez[0:t,j,i] = 0  
            if d15n_pom_futndep_detectsmooth_rang_ez[t,j,i] == 1:
                d15n_pom_futndep_toe_rang_ez[t,j,i] = 1
            if d15n_pom_futndep_yearchange_rang_ez[t,j,i] < 0:
                d15n_pom_futndep_toe_rang_ez[0:t,j,i] = 0  
    
            if d15n_pom_futndep_detectsmooth_std1_utz[t,j,i] == 1:
                d15n_pom_futndep_toe_std1_utz[t,j,i] = 1
            if d15n_pom_futndep_yearchange_std1_utz[t,j,i] < 0:
                d15n_pom_futndep_toe_std1_utz[0:t,j,i] = 0  
            if d15n_pom_futndep_detectsmooth_std2_utz[t,j,i] == 1:
                d15n_pom_futndep_toe_std2_utz[t,j,i] = 1
            if d15n_pom_futndep_yearchange_std2_utz[t,j,i] < 0:
                d15n_pom_futndep_toe_std2_utz[0:t,j,i] = 0  
            if d15n_pom_futndep_detectsmooth_rang_utz[t,j,i] == 1:
                d15n_pom_futndep_toe_rang_utz[t,j,i] = 1
            if d15n_pom_futndep_yearchange_rang_utz[t,j,i] < 0:
                d15n_pom_futndep_toe_rang_utz[0:t,j,i] = 0  
    


print("Robust, consistent emergences identified")


#%% print the ToEs!

ToE_picndep = {}
ToE_fut = {}
ToE_futndep = {}

# get the minimum (first) year that each time series equals one
ToE_picndep['fix_toeyear_std1']= np.ma.min(np.ma.masked_where(fix_picndep_toe_std1==0, fix_picndep_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['fix_toeyear_std2']= np.ma.min(np.ma.masked_where(fix_picndep_toe_std2==0, fix_picndep_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['fix_toeyear_rang']= np.ma.min(np.ma.masked_where(fix_picndep_toe_rang==0, fix_picndep_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_picndep['npp_toeyear_std1']= np.ma.min(np.ma.masked_where(npp_picndep_toe_std1==0, npp_picndep_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['npp_toeyear_std2']= np.ma.min(np.ma.masked_where(npp_picndep_toe_std2==0, npp_picndep_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['npp_toeyear_rang']= np.ma.min(np.ma.masked_where(npp_picndep_toe_rang==0, npp_picndep_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_picndep['no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(no3_picndep_toe_std1_ez==0, no3_picndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(no3_picndep_toe_std2_ez==0, no3_picndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(no3_picndep_toe_rang_ez==0, no3_picndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(no3_picndep_toe_std1_utz==0, no3_picndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(no3_picndep_toe_std2_utz==0, no3_picndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(no3_picndep_toe_rang_utz==0, no3_picndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_picndep['nst_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(nst_picndep_toe_std1_ez==0, nst_picndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['nst_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(nst_picndep_toe_std2_ez==0, nst_picndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['nst_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(nst_picndep_toe_rang_ez==0, nst_picndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['nst_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(nst_picndep_toe_std1_utz==0, nst_picndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['nst_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(nst_picndep_toe_std2_utz==0, nst_picndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['nst_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(nst_picndep_toe_rang_utz==0, nst_picndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_picndep['d15n_no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_std1_ez==0, d15n_no3_picndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_std2_ez==0, d15n_no3_picndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_rang_ez==0, d15n_no3_picndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_std1_utz==0, d15n_no3_picndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_std2_utz==0, d15n_no3_picndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_no3_picndep_toe_rang_utz==0, d15n_no3_picndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_picndep['d15n_pom_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_std1_ez==0, d15n_pom_picndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_pom_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_std2_ez==0, d15n_pom_picndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_pom_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_rang_ez==0, d15n_pom_picndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_pom_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_std1_utz==0, d15n_pom_picndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_pom_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_std2_utz==0, d15n_pom_picndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_picndep['d15n_pom_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_pom_picndep_toe_rang_utz==0, d15n_pom_picndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)


ToE_fut['fix_toeyear_std1']= np.ma.min(np.ma.masked_where(fix_fut_toe_std1==0, fix_fut_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['fix_toeyear_std2']= np.ma.min(np.ma.masked_where(fix_fut_toe_std2==0, fix_fut_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['fix_toeyear_rang']= np.ma.min(np.ma.masked_where(fix_fut_toe_rang==0, fix_fut_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['npp_toeyear_std1']= np.ma.min(np.ma.masked_where(npp_fut_toe_std1==0, npp_fut_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['npp_toeyear_std2']= np.ma.min(np.ma.masked_where(npp_fut_toe_std2==0, npp_fut_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['npp_toeyear_rang']= np.ma.min(np.ma.masked_where(npp_fut_toe_rang==0, npp_fut_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['temp_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(temp_fut_toe_std1_ez==0, temp_fut_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['temp_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(temp_fut_toe_std2_ez==0, temp_fut_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['temp_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(temp_fut_toe_rang_ez==0, temp_fut_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['temp_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(temp_fut_toe_std1_utz==0, temp_fut_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['temp_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(temp_fut_toe_std2_utz==0, temp_fut_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['temp_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(temp_fut_toe_rang_utz==0, temp_fut_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(no3_fut_toe_std1_ez==0, no3_fut_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(no3_fut_toe_std2_ez==0, no3_fut_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(no3_fut_toe_rang_ez==0, no3_fut_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(no3_fut_toe_std1_utz==0, no3_fut_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(no3_fut_toe_std2_utz==0, no3_fut_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(no3_fut_toe_rang_utz==0, no3_fut_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['nst_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(nst_fut_toe_std1_ez==0, nst_fut_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['nst_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(nst_fut_toe_std2_ez==0, nst_fut_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['nst_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(nst_fut_toe_rang_ez==0, nst_fut_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['nst_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(nst_fut_toe_std1_utz==0, nst_fut_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['nst_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(nst_fut_toe_std2_utz==0, nst_fut_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['nst_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(nst_fut_toe_rang_utz==0, nst_fut_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['d15n_no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_std1_ez==0, d15n_no3_fut_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_std2_ez==0, d15n_no3_fut_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_rang_ez==0, d15n_no3_fut_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_std1_utz==0, d15n_no3_fut_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_std2_utz==0, d15n_no3_fut_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_no3_fut_toe_rang_utz==0, d15n_no3_fut_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_fut['d15n_pom_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_std1_ez==0, d15n_pom_fut_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_pom_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_std2_ez==0, d15n_pom_fut_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_pom_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_rang_ez==0, d15n_pom_fut_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_pom_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_std1_utz==0, d15n_pom_fut_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_pom_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_std2_utz==0, d15n_pom_fut_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_fut['d15n_pom_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_pom_fut_toe_rang_utz==0, d15n_pom_fut_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)


ToE_futndep['fix_toeyear_std1']= np.ma.min(np.ma.masked_where(fix_futndep_toe_std1==0, fix_futndep_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['fix_toeyear_std2']= np.ma.min(np.ma.masked_where(fix_futndep_toe_std2==0, fix_futndep_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['fix_toeyear_rang']= np.ma.min(np.ma.masked_where(fix_futndep_toe_rang==0, fix_futndep_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_futndep['npp_toeyear_std1']= np.ma.min(np.ma.masked_where(npp_futndep_toe_std1==0, npp_futndep_toe_std1) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['npp_toeyear_std2']= np.ma.min(np.ma.masked_where(npp_futndep_toe_std2==0, npp_futndep_toe_std2) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['npp_toeyear_rang']= np.ma.min(np.ma.masked_where(npp_futndep_toe_rang==0, npp_futndep_toe_rang) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_futndep['no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(no3_futndep_toe_std1_ez==0, no3_futndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(no3_futndep_toe_std2_ez==0, no3_futndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(no3_futndep_toe_rang_ez==0, no3_futndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(no3_futndep_toe_std1_utz==0, no3_futndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(no3_futndep_toe_std2_utz==0, no3_futndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(no3_futndep_toe_rang_utz==0, no3_futndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_futndep['nst_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(nst_futndep_toe_std1_ez==0, nst_futndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['nst_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(nst_futndep_toe_std2_ez==0, nst_futndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['nst_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(nst_futndep_toe_rang_ez==0, nst_futndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['nst_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(nst_futndep_toe_std1_utz==0, nst_futndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['nst_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(nst_futndep_toe_std2_utz==0, nst_futndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['nst_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(nst_futndep_toe_rang_utz==0, nst_futndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_futndep['d15n_no3_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_std1_ez==0, d15n_no3_futndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_no3_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_std2_ez==0, d15n_no3_futndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_no3_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_rang_ez==0, d15n_no3_futndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_no3_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_std1_utz==0, d15n_no3_futndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_no3_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_std2_utz==0, d15n_no3_futndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_no3_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_no3_futndep_toe_rang_utz==0, d15n_no3_futndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)

ToE_futndep['d15n_pom_toeyear_std1_ez']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_std1_ez==0, d15n_pom_futndep_toe_std1_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_pom_toeyear_std2_ez']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_std2_ez==0, d15n_pom_futndep_toe_std2_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_pom_toeyear_rang_ez']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_rang_ez==0, d15n_pom_futndep_toe_rang_ez) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_pom_toeyear_std1_utz']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_std1_utz==0, d15n_pom_futndep_toe_std1_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_pom_toeyear_std2_utz']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_std2_utz==0, d15n_pom_futndep_toe_std2_utz) * time[:,np.newaxis,np.newaxis],axis=0)
ToE_futndep['d15n_pom_toeyear_rang_utz']= np.ma.min(np.ma.masked_where(d15n_pom_futndep_toe_rang_utz==0, d15n_pom_futndep_toe_rang_utz) * time[:,np.newaxis,np.newaxis],axis=0)


print("Year of Emergence identified")

#%% check the outcome


lab = ['Nat', 'Nat + N dep', 'Climate change', 'Climate change + N dep']

off = 0.01
alf=0.7; alf1 = 0.2
i = 160; j = 110
print(lon[i], lat[j])
fslab = 14; fstic = 12

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot(2,3,1)
ax1.tick_params(labelsize=fstic)
plt.title('Raw timeseries', fontsize=fslab)
plt.plot(time,d15n_no3_pic_ez[:,j,i], color='k',alpha=alf, label=lab[0])
plt.plot(time,d15n_no3_picndep_ez[:,j,i], color='royalblue',alpha=alf, label=lab[1])
plt.plot(time,d15n_no3_fut_ez[:,j,i], color='goldenrod',alpha=alf, label=lab[2])
plt.plot(time,d15n_no3_futndep_ez[:,j,i], color='firebrick',alpha=alf, label=lab[3])
plt.legend(frameon=False, loc='upper center', ncol=4, bbox_to_anchor=(1.8,1.3), fontsize=fslab)
plt.ylabel('$\delta^{15}$N$_{NO_3}$', fontsize=fslab)
ax2 = plt.subplot(2,3,2)
ax2.tick_params(labelsize=fstic)
plt.title('Detrended, normalised and smoothed', fontsize=fslab)
plt.plot(time,d15n_no3_pic_ez_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_ez_smoothed[:,j,i], color='k',linewidth=2)
plt.plot(time,d15n_no3_picndep_ez_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_ez_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_ez_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_ez_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_ez_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_ez_smoothed[:,j,i], color='firebrick', linewidth=2)
ax3 = plt.subplot(2,3,3)
ax3.tick_params(labelsize=fstic)
plt.title('Signal emergence', fontsize=fslab)
plt.plot(time,d15n_no3_picndep_toe_std2_ez[:,j,i]+off, color='royalblue', linestyle='-')
plt.plot(time,d15n_no3_fut_toe_std2_ez[:,j,i], color='goldenrod', linestyle='-')
plt.plot(time,d15n_no3_futndep_toe_std2_ez[:,j,i]-off, color='firebrick', linestyle='-')
if np.ma.is_masked(ToE_picndep['d15n_no3_toeyear_std2_ez'][j,i]) == False:
    plt.text(1.01,0.8, 'ToE(std*2) = %d'%(ToE_picndep['d15n_no3_toeyear_std2_ez'][j,i]), transform=ax3.transAxes, color='royalblue', fontsize=fslab)
if np.ma.is_masked(ToE_fut['d15n_no3_toeyear_std2_ez'][j,i]) == False:
    plt.text(1.01,0.8, 'ToE(std*2) = %d'%(ToE_fut['d15n_no3_toeyear_std2_ez'][j,i]), transform=ax3.transAxes, color='goldenrod', fontsize=fslab)
if np.ma.is_masked(ToE_futndep['d15n_no3_toeyear_std2_ez'][j,i]) == False:
    plt.text(1.01,0.5, 'ToE(std*2) = %d'%(ToE_futndep['d15n_no3_toeyear_std2_ez'][j,i]), transform=ax3.transAxes, color='firebrick', fontsize=fslab)
plt.ylim(-0.1,1.1)

ax4 = plt.subplot(2,3,4)
ax4.tick_params(labelsize=fstic)
plt.plot(time,d15n_no3_pic_utz[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_fut_utz[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz[:,j,i], color='firebrick',alpha=alf)
plt.ylabel('$\delta^{15}$N$_{NO_3}$', fontsize=fslab)
plt.xlabel('Year', fontsize=fslab)
ax5 = plt.subplot(2,3,5)
ax5.tick_params(labelsize=fstic)
plt.plot(time,d15n_no3_pic_utz_normalised[:,j,i], color='k',alpha=alf)
plt.plot(time,d15n_no3_pic_utz_smoothed[:,j,i], color='k', linewidth=2)
plt.plot(time,d15n_no3_picndep_utz_normalised[:,j,i], color='royalblue',alpha=alf)
plt.plot(time,d15n_no3_picndep_utz_smoothed[:,j,i], color='royalblue', linewidth=2)
plt.plot(time,d15n_no3_fut_utz_normalised[:,j,i], color='goldenrod',alpha=alf)
plt.plot(time,d15n_no3_fut_utz_smoothed[:,j,i], color='goldenrod', linewidth=2)
plt.plot(time,d15n_no3_futndep_utz_normalised[:,j,i], color='firebrick',alpha=alf)
plt.plot(time,d15n_no3_futndep_utz_smoothed[:,j,i], color='firebrick', linewidth=2)
plt.xlabel('Year', fontsize=fslab)
ax6 = plt.subplot(2,3,6)
ax6.tick_params(labelsize=fstic)
plt.plot(time,d15n_no3_picndep_toe_std2_utz[:,j,i]+off, color='royalblue', linestyle='-')
plt.plot(time,d15n_no3_fut_toe_std2_utz[:,j,i], color='goldenrod', linestyle='-')
plt.plot(time,d15n_no3_futndep_toe_std2_utz[:,j,i]-off, color='firebrick', linestyle='-')
if np.ma.is_masked(ToE_picndep['d15n_no3_toeyear_std2_utz'][j,i]) == False:
    plt.text(1.01,0.8, 'ToE(std*2) = %d'%(ToE_picndep['d15n_no3_toeyear_std2_utz'][j,i]), transform=ax6.transAxes, color='royalblue', fontsize=fslab)
if np.ma.is_masked(ToE_fut['d15n_no3_toeyear_std2_utz'][j,i]) == False:
    plt.text(1.01,0.65, 'ToE(std*2) = %d'%(ToE_fut['d15n_no3_toeyear_std2_utz'][j,i]), transform=ax6.transAxes, color='goldenrod', fontsize=fslab)
if np.ma.is_masked(ToE_futndep['d15n_no3_toeyear_std2_utz'][j,i]) == False:
    plt.text(1.01,0.5, 'ToE(std*2) = %d'%(ToE_futndep['d15n_no3_toeyear_std2_utz'][j,i]), transform=ax6.transAxes, color='firebrick', fontsize=fslab)
plt.ylim(-0.1,1.1)
plt.xlabel('Year', fontsize=fslab)

xx=-0.45;yy=0.5
plt.text(xx,yy,'Euphotic\nZone', ha='center', va='center', transform=ax1.transAxes, fontsize=fslab)
plt.text(xx,yy,'Twilight\nZone', ha='center', va='center', transform=ax4.transAxes, fontsize=fslab)

plt.subplots_adjust(left=0.15, right=0.85)

# savefig
fig.savefig("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures//fig-supp12.png", dpi=300, bbox_inches='tight')
fig.savefig("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures//fig-supp12_trans.png", dpi=300, bbox_inches='tight', transparent=True)


#%% save files

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data")

os.remove('ETOPO_ToE_picndep_depthzones.nc')
## activate line below if overwriting file
data = nc.Dataset('ETOPO_ToE_picndep_depthzones.nc', 'w', format='NETCDF4_CLASSIC')

xd = data.createDimension('x', (360))
yd = data.createDimension('y', (180))

vlon = data.createVariable('lon', np.float64, ('y','x'))
vlat = data.createVariable('lat', np.float64, ('y','x'))
vToE_fix = data.createVariable('fix', np.float64, ('y','x'))
vToE_npp = data.createVariable('npp', np.float64, ('y','x'))
vToE_no3_ez = data.createVariable('no3_ez', np.float64, ('y','x'))
vToE_no3_utz = data.createVariable('no3_utz', np.float64, ('y','x'))
vToE_nst_ez = data.createVariable('nst_ez', np.float64, ('y','x'))
vToE_nst_utz = data.createVariable('nst_utz', np.float64, ('y','x'))
vToE_d15n_no3_ez = data.createVariable('d15n_no3_ez', np.float64, ('y','x'))
vToE_d15n_no3_utz = data.createVariable('d15n_no3_utz', np.float64, ('y','x'))
vToE_d15n_pom_ez = data.createVariable('d15n_pom_ez', np.float64, ('y','x'))
vToE_d15n_pom_utz = data.createVariable('d15n_pom_utz', np.float64, ('y','x'))

data.description = 'Time of Emergence of biogeochemical fields for Buchanan & Tagliabue (2020) Biogeosciences'
data.history = "Created by Pearse J. Buchanan on 6th May 2020"
data.source = "Output from NEMO-PISCES ocean model"

vlon.units = "degrees_east"
vlat.units = "degrees_north"
vToE_fix.units = "year"
vToE_npp.units = "year"
vToE_no3_ez.units = "year"
vToE_no3_utz.units = "year"
vToE_nst_ez.units = "year"
vToE_nst_utz.units = "year"
vToE_d15n_no3_ez.units = "year"
vToE_d15n_no3_utz.units = "year"
vToE_d15n_pom_ez.units = "year"
vToE_d15n_pom_utz.units = "year"

vlon.standard_name = "longitude"
vlat.standard_name = "latitude"

vlon.long_name = "longitude"
vlat.long_name = "latitude"

vlon.axis = "X"
vlat.axis = "Y"

vToE_fix.coordinates = "x y"
vToE_npp.coordinates = "x y"
vToE_no3_ez.coordinates = "x y"
vToE_no3_utz.coordinates = "x y"
vToE_nst_ez.coordinates = "x y"
vToE_nst_utz.coordinates = "x y"
vToE_d15n_no3_ez.coordinates = "x y"
vToE_d15n_no3_utz.coordinates = "x y"
vToE_d15n_pom_ez.coordinates = "x y"
vToE_d15n_pom_utz.coordinates = "x y"

lon2 = lon+360
lons,lats = np.meshgrid(lon2,lat)

vlon[:] = lons
vlat[:] = lats
vToE_fix[:,:] = ToE_picndep['fix_toeyear_std2'][:,:]
vToE_npp[:,:] = ToE_picndep['npp_toeyear_std2'][:,:]
vToE_no3_ez[:,:] = ToE_picndep['no3_toeyear_std2_ez'][:,:]
vToE_no3_utz[:,:] = ToE_picndep['no3_toeyear_std2_utz'][:,:]
vToE_nst_ez[:,:] = ToE_picndep['nst_toeyear_std2_ez'][:,:]
vToE_nst_utz[:,:] = ToE_picndep['nst_toeyear_std2_utz'][:,:]
vToE_d15n_no3_ez[:,:] = ToE_picndep['d15n_no3_toeyear_std2_ez'][:,:]
vToE_d15n_no3_utz[:,:] = ToE_picndep['d15n_no3_toeyear_std2_utz'][:,:]
vToE_d15n_pom_ez[:,:] = ToE_picndep['d15n_pom_toeyear_std2_ez'][:,:]
vToE_d15n_pom_utz[:,:] = ToE_picndep['d15n_pom_toeyear_std2_utz'][:,:]


data.close()


#%% 

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data")

os.remove('ETOPO_ToE_fut_depthzones.nc')
## activate line below if overwriting file
data = nc.Dataset('ETOPO_ToE_fut_depthzones.nc', 'w', format='NETCDF4_CLASSIC')

xd = data.createDimension('x', (360))
yd = data.createDimension('y', (180))

vlon = data.createVariable('lon', np.float64, ('y','x'))
vlat = data.createVariable('lat', np.float64, ('y','x'))
vToE_fix = data.createVariable('fix', np.float64, ('y','x'))
vToE_npp = data.createVariable('npp', np.float64, ('y','x'))
vToE_temp_ez = data.createVariable('temp_ez', np.float64, ('y','x'))
vToE_temp_utz = data.createVariable('temp_utz', np.float64, ('y','x'))
vToE_no3_ez = data.createVariable('no3_ez', np.float64, ('y','x'))
vToE_no3_utz = data.createVariable('no3_utz', np.float64, ('y','x'))
vToE_nst_ez = data.createVariable('nst_ez', np.float64, ('y','x'))
vToE_nst_utz = data.createVariable('nst_utz', np.float64, ('y','x'))
vToE_d15n_no3_ez = data.createVariable('d15n_no3_ez', np.float64, ('y','x'))
vToE_d15n_no3_utz = data.createVariable('d15n_no3_utz', np.float64, ('y','x'))
vToE_d15n_pom_ez = data.createVariable('d15n_pom_ez', np.float64, ('y','x'))
vToE_d15n_pom_utz = data.createVariable('d15n_pom_utz', np.float64, ('y','x'))

data.description = 'Time of Emergence of biogeochemical fields for Buchanan & Tagliabue (2020) Biogeosciences'
data.history = "Created by Pearse J. Buchanan on 6th May 2020"
data.source = "Output from NEMO-PISCES ocean model"

vlon.units = "degrees_east"
vlat.units = "degrees_north"
vToE_fix.units = "year"
vToE_npp.units = "year"
vToE_temp_ez.units = "year"
vToE_temp_utz.units = "year"
vToE_no3_ez.units = "year"
vToE_no3_utz.units = "year"
vToE_nst_ez.units = "year"
vToE_nst_utz.units = "year"
vToE_d15n_no3_ez.units = "year"
vToE_d15n_no3_utz.units = "year"
vToE_d15n_pom_ez.units = "year"
vToE_d15n_pom_utz.units = "year"

vlon.standard_name = "longitude"
vlat.standard_name = "latitude"

vlon.long_name = "longitude"
vlat.long_name = "latitude"

vlon.axis = "X"
vlat.axis = "Y"

vToE_fix.coordinates = "x y"
vToE_npp.coordinates = "x y"
vToE_temp_ez.coordinates = "x y"
vToE_temp_utz.coordinates = "x y"
vToE_no3_ez.coordinates = "x y"
vToE_no3_utz.coordinates = "x y"
vToE_nst_ez.coordinates = "x y"
vToE_nst_utz.coordinates = "x y"
vToE_d15n_no3_ez.coordinates = "x y"
vToE_d15n_no3_utz.coordinates = "x y"
vToE_d15n_pom_ez.coordinates = "x y"
vToE_d15n_pom_utz.coordinates = "x y"

lon2 = lon+360
lons,lats = np.meshgrid(lon2,lat)

vlon[:] = lons
vlat[:] = lats
vToE_fix[:,:] = ToE_fut['fix_toeyear_std2'][:,:]
vToE_npp[:,:] = ToE_fut['npp_toeyear_std2'][:,:]
vToE_temp_ez[:,:] = ToE_fut['temp_toeyear_std2_ez'][:,:]
vToE_temp_utz[:,:] = ToE_fut['temp_toeyear_std2_utz'][:,:]
vToE_no3_ez[:,:] = ToE_fut['no3_toeyear_std2_ez'][:,:]
vToE_no3_utz[:,:] = ToE_fut['no3_toeyear_std2_utz'][:,:]
vToE_nst_ez[:,:] = ToE_fut['nst_toeyear_std2_ez'][:,:]
vToE_nst_utz[:,:] = ToE_fut['nst_toeyear_std2_utz'][:,:]
vToE_d15n_no3_ez[:,:] = ToE_fut['d15n_no3_toeyear_std2_ez'][:,:]
vToE_d15n_no3_utz[:,:] = ToE_fut['d15n_no3_toeyear_std2_utz'][:,:]
vToE_d15n_pom_ez[:,:] = ToE_fut['d15n_pom_toeyear_std2_ez'][:,:]
vToE_d15n_pom_utz[:,:] = ToE_fut['d15n_pom_toeyear_std2_utz'][:,:]


data.close()


#%%

os.remove('ETOPO_ToE_futndep_depthzones.nc')
## activate line below if overwriting file
data = nc.Dataset('ETOPO_ToE_futndep_depthzones.nc', 'w', format='NETCDF4_CLASSIC')

xd = data.createDimension('x', (360))
yd = data.createDimension('y', (180))

vlon = data.createVariable('lon', np.float64, ('y','x'))
vlat = data.createVariable('lat', np.float64, ('y','x'))
vToE_fix = data.createVariable('fix', np.float64, ('y','x'))
vToE_npp = data.createVariable('npp', np.float64, ('y','x'))
vToE_no3_ez = data.createVariable('no3_ez', np.float64, ('y','x'))
vToE_no3_utz = data.createVariable('no3_utz', np.float64, ('y','x'))
vToE_nst_ez = data.createVariable('nst_ez', np.float64, ('y','x'))
vToE_nst_utz = data.createVariable('nst_utz', np.float64, ('y','x'))
vToE_d15n_no3_ez = data.createVariable('d15n_no3_ez', np.float64, ('y','x'))
vToE_d15n_no3_utz = data.createVariable('d15n_no3_utz', np.float64, ('y','x'))
vToE_d15n_pom_ez = data.createVariable('d15n_pom_ez', np.float64, ('y','x'))
vToE_d15n_pom_utz = data.createVariable('d15n_pom_utz', np.float64, ('y','x'))

data.description = 'Time of Emergence of biogeochemical fields for Buchanan & Tagliabue (2020) Biogeosciences'
data.history = "Created by Pearse J. Buchanan on 6th May 2020"
data.source = "Output from NEMO-PISCES ocean model"

vlon.units = "degrees_east"
vlat.units = "degrees_north"
vToE_fix.units = "year"
vToE_npp.units = "year"
vToE_no3_ez.units = "year"
vToE_no3_utz.units = "year"
vToE_nst_ez.units = "year"
vToE_nst_utz.units = "year"
vToE_d15n_no3_ez.units = "year"
vToE_d15n_no3_utz.units = "year"
vToE_d15n_pom_ez.units = "year"
vToE_d15n_pom_utz.units = "year"

vlon.standard_name = "longitude"
vlat.standard_name = "latitude"

vlon.long_name = "longitude"
vlat.long_name = "latitude"

vlon.axis = "X"
vlat.axis = "Y"

vToE_fix.coordinates = "x y"
vToE_npp.coordinates = "x y"
vToE_no3_ez.coordinates = "x y"
vToE_no3_utz.coordinates = "x y"
vToE_nst_ez.coordinates = "x y"
vToE_nst_utz.coordinates = "x y"
vToE_d15n_no3_ez.coordinates = "x y"
vToE_d15n_no3_utz.coordinates = "x y"
vToE_d15n_pom_ez.coordinates = "x y"
vToE_d15n_pom_utz.coordinates = "x y"

lon2 = lon+360
lons,lats = np.meshgrid(lon2,lat)

vlon[:] = lons
vlat[:] = lats
vToE_fix[:,:] = ToE_futndep['fix_toeyear_std2'][:,:]
vToE_npp[:,:] = ToE_futndep['npp_toeyear_std2'][:,:]
vToE_no3_ez[:,:] = ToE_futndep['no3_toeyear_std2_ez'][:,:]
vToE_no3_utz[:,:] = ToE_futndep['no3_toeyear_std2_utz'][:,:]
vToE_nst_ez[:,:] = ToE_futndep['nst_toeyear_std2_ez'][:,:]
vToE_nst_utz[:,:] = ToE_futndep['nst_toeyear_std2_utz'][:,:]
vToE_d15n_no3_ez[:,:] = ToE_futndep['d15n_no3_toeyear_std2_ez'][:,:]
vToE_d15n_no3_utz[:,:] = ToE_futndep['d15n_no3_toeyear_std2_utz'][:,:]
vToE_d15n_pom_ez[:,:] = ToE_futndep['d15n_pom_toeyear_std2_ez'][:,:]
vToE_d15n_pom_utz[:,:] = ToE_futndep['d15n_pom_toeyear_std2_utz'][:,:]


data.close()
