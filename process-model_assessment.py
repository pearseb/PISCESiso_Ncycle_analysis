# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:44:53 2017

@author: pearseb
"""

#%% imports

from __future__ import unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import netCDF4 as nc
import cmocean
import cmocean.cm as cmo
from scipy.optimize import curve_fit
from matplotlib.animation import ArtistAnimation
import seaborn as sb
sb.set(style='ticks')

import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


#%% load model data

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\data_for_publication')

#data = nc.Dataset('ETOPO_future_ndep_d15Nno3_1986-2005average.nc', 'r')
data = nc.Dataset('ETOPO_spinup_d15Nno3.nc', 'r')
no3 = np.ma.squeeze(data.variables['NO3'][...])
no3_15 = np.ma.squeeze(data.variables['NO3_15'][...])
d15n = (no3_15/no3-1)*1000
lon = data.variables['ETOPO60X'][...]
lat = data.variables['ETOPO60Y'][...]
dep = data.variables['deptht'][...]

land = np.ma.getmask(d15n)
land = np.ma.masked_where(land==False, land)
land[land==True] = 1.0

lons = np.append(lon[-1], lon[:])
lons[0] = 19.5
lons, lats = np.meshgrid(lons,lat)


# load model basin masks for later use
data = nc.Dataset('ETOPO_ORCA2.0_Basins_float.nc', 'r')
atlmask = data.variables['atlmsk'][...]
pacmask = data.variables['pacmsk'][...]
indmask = data.variables['indmsk'][...]
aramask = data.variables['aramsk'][...]
arpmask = data.variables['arpmsk'][...]
glomask = data.variables['glomsk'][...]
glomask_msf = data.variables['glomsk_msf'][...]
medmask = data.variables['medmsk'][...]

# load model grid information for later use
data = nc.Dataset('ETOPO_ORCA2.0.full_grid.nc', 'r')
vol = data.variables['volume'][...]
area = data.variables['area'][...]
thickness = vol/area

data.close()


#%% get observations

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\data_for_publication')
npfile = np.load('RafterTuerena_watercolumn_d15N_no3_gridded.npz')
npfile.files
d15n_obs = npfile['arr_0']

# mask obs where model is land
d15n_obs = np.ma.masked_where(np.ma.getmask(d15n),d15n_obs)
# mask zero values
d15n_obs = np.ma.masked_where(d15n_obs==0.0, d15n_obs)
mask = np.ma.getmask(d15n_obs)

# mask model output for a 1:1 comparison with the data
d15n_mod = np.ma.masked_where(mask, d15n)

# also mask the NO3 concentration data for plotting
no3_mod = np.ma.masked_where(mask, no3)


#%% check that the 1:1 comparison is perfect

print(len(np.ma.ravel(d15n_obs)), len(np.ma.ravel(d15n_mod)), len(np.ma.ravel(no3_mod)))
print(np.ma.count_masked(d15n_obs), np.ma.count_masked(d15n_mod), np.ma.count_masked(no3_mod))
print("values for direct model-data comparison after averaging measurements within each grid cell")
print(len(np.ma.ravel(d15n_obs))-np.ma.count_masked(d15n_obs), len(np.ma.ravel(d15n_mod))-np.ma.count_masked(d15n_mod), len(np.ma.ravel(no3_mod))-np.ma.count_masked(no3_mod))

# produce 1D arrays of the data
d15n_obs_1d = np.ma.ravel(d15n_obs)
d15n_mod_1d = np.ma.ravel(d15n_mod)
no3_mod_1d = np.ma.ravel(no3_mod)


#%% quick visualisation

fslab = 15
fstic = 13

fig = plt.figure(facecolor='w', figsize=(8,12))
ax1 = fig.gca()
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(labelsize=fstic)

p1 = plt.scatter(d15n_obs_1d, d15n_mod_1d, c=no3_mod_1d, cmap=cmo.deep, alpha=0.4, edgecolor='grey', linewidth=0.5)
plt.plot((-5,45),(-5,45),'k--', alpha=0.5, zorder=0, linewidth=1)

plt.ylabel('Simulated $\delta^{15}$N$_{NO_3}$ (\u2030)', fontsize=fslab, labelpad=10)
plt.xlabel('Measured $\delta^{15}$N$_{NO_3}$ (\u2030)', fontsize=fslab, labelpad=10)
plt.yticks(np.arange(0,41,5), np.arange(0,41,5), fontsize=fstic)
plt.xticks(np.arange(0,41,5), np.arange(0,41,5), fontsize=fstic)
plt.xlim(-5,35); plt.ylim(-5,35)

cbax = fig.add_axes([0.2,0.8,0.5,0.03])
cbar = plt.colorbar(p1, cax=cbax, orientation='horizontal', ticks=np.arange(0,51,5))
cbar.set_label('NO$_3$ concentration (mmol m$^{-1}$', fontsize=fslab, labelpad=5)
cbar.ax.set_yticklabels(np.arange(0,51,5))
cbar.ax.tick_params(labelsize=fstic)

plt.text(0.35,0.7, "N = %i"%(len(np.ma.ravel(d15n_obs))-np.ma.count_masked(d15n_obs)), fontsize=fslab, transform=ax1.transAxes)


#%%

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDoc\\my articles\\d15N and d13C in PISCES\\scripts_for_publication\\supplementary_figures')
fig.savefig('fig-supp2.png', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp2.pdf', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp2_trans.png', dpi=300, bbox_inches='tight', transparent=True)


#%%


'''

After this initial model-data comparison, it is important to look at model-data fits 
within different regions of the ocean. This includes different ocean basins and
different depth sections.

The following code produces visualisations of model-data fit for these different
regions. Finally, it calculates univariate measures of fit (correlation, RSME, STD),
such that Taylor Diagrams can then be made.

'''


#%% now that we have the averages for each equivalent grid point, I can take weighted averages in space

### depth space
d15n_obs_100m = np.ma.average(d15n_obs[0:10,...], weights=thickness[0:10,...], axis=0)      # 0-100
d15n_obs_200m = np.ma.average(d15n_obs[10:16,...], weights=thickness[10:16,...], axis=0)    # 100-200
d15n_obs_400m = np.ma.average(d15n_obs[16:19,...], weights=thickness[16:19,...], axis=0)    # 200-400
d15n_obs_500m = d15n_obs[19,...]                                                            # 508
d15n_obs_750m = d15n_obs[20,...]                                                            # 729
d15n_obs_1500m = np.ma.average(d15n_obs[21:23,...], weights=thickness[21:23,...], axis=0)   # 1000-1600
d15n_obs_3000m = np.ma.average(d15n_obs[23:26,...], weights=thickness[23:26,...], axis=0)   # 1600-3000
d15n_obs_5000m = np.ma.average(d15n_obs[25:31,...], weights=thickness[25:31,...], axis=0)   # 3000-5000

# put back onto 1d arrays for scattering
llons, llats = np.meshgrid(lon, lat)
maskobs100 = np.ma.getmask(d15n_obs_100m)
maskobs200 = np.ma.getmask(d15n_obs_200m)
maskobs400 = np.ma.getmask(d15n_obs_400m)
maskobs500 = np.ma.getmask(d15n_obs_500m)
maskobs750 = np.ma.getmask(d15n_obs_750m)
maskobs1500 = np.ma.getmask(d15n_obs_1500m)
maskobs3000 = np.ma.getmask(d15n_obs_3000m)
maskobs5000 = np.ma.getmask(d15n_obs_5000m)

llons100 = np.ma.masked_where(maskobs100, llons); llats100 = np.ma.masked_where(maskobs100, llats)
llons200 = np.ma.masked_where(maskobs200, llons); llats200 = np.ma.masked_where(maskobs200, llats)
llons400 = np.ma.masked_where(maskobs400, llons); llats400 = np.ma.masked_where(maskobs400, llats)
llons500 = np.ma.masked_where(maskobs500, llons); llats500 = np.ma.masked_where(maskobs500, llats)
llons750 = np.ma.masked_where(maskobs750, llons); llats750 = np.ma.masked_where(maskobs750, llats)
llons1500 = np.ma.masked_where(maskobs1500, llons); llats1500 = np.ma.masked_where(maskobs1500, llats)
llons3000 = np.ma.masked_where(maskobs3000, llons); llats3000 = np.ma.masked_where(maskobs3000, llats)
llons5000 = np.ma.masked_where(maskobs5000, llons); llats5000 = np.ma.masked_where(maskobs5000, llats)

d15n_obs_100m_1d = np.ma.MaskedArray.compressed(d15n_obs_100m); llons100_1d = np.ma.MaskedArray.compressed(llons100); llats100_1d = np.ma.MaskedArray.compressed(llats100)
d15n_obs_200m_1d = np.ma.MaskedArray.compressed(d15n_obs_200m); llons200_1d = np.ma.MaskedArray.compressed(llons200); llats200_1d = np.ma.MaskedArray.compressed(llats200)
d15n_obs_400m_1d = np.ma.MaskedArray.compressed(d15n_obs_400m); llons400_1d = np.ma.MaskedArray.compressed(llons400); llats400_1d = np.ma.MaskedArray.compressed(llats400)
d15n_obs_500m_1d = np.ma.MaskedArray.compressed(d15n_obs_500m); llons500_1d = np.ma.MaskedArray.compressed(llons500); llats500_1d = np.ma.MaskedArray.compressed(llats500)
d15n_obs_750m_1d = np.ma.MaskedArray.compressed(d15n_obs_750m); llons750_1d = np.ma.MaskedArray.compressed(llons750); llats750_1d = np.ma.MaskedArray.compressed(llats750)
d15n_obs_1500m_1d = np.ma.MaskedArray.compressed(d15n_obs_1500m); llons1500_1d = np.ma.MaskedArray.compressed(llons1500); llats1500_1d = np.ma.MaskedArray.compressed(llats1500)
d15n_obs_3000m_1d = np.ma.MaskedArray.compressed(d15n_obs_3000m); llons3000_1d = np.ma.MaskedArray.compressed(llons3000); llats3000_1d = np.ma.MaskedArray.compressed(llats3000)
d15n_obs_5000m_1d = np.ma.MaskedArray.compressed(d15n_obs_5000m); llons5000_1d = np.ma.MaskedArray.compressed(llons5000); llats5000_1d = np.ma.MaskedArray.compressed(llats5000)


#%% get the model averages in depth space too

d15n_mod_100m = np.ma.average(d15n[0:10,...], weights=thickness[0:10,...], axis=0)      # 0-100
d15n_mod_200m = np.ma.average(d15n[10:16,...], weights=thickness[10:16,...], axis=0)    # 100-200
d15n_mod_400m = np.ma.average(d15n[16:19,...], weights=thickness[16:19,...], axis=0)    # 200-400
d15n_mod_500m = d15n[19,...]                                                            # 508
d15n_mod_750m = d15n[20,...]                                                            # 729
d15n_mod_1500m = np.ma.average(d15n[21:23,...], weights=thickness[21:23,...], axis=0)   # 1000-1600
d15n_mod_3000m = np.ma.average(d15n[23:26,...], weights=thickness[23:26,...], axis=0)   # 1600-3000
d15n_mod_5000m = np.ma.average(d15n[25:31,...], weights=thickness[25:31,...], axis=0)   # 3000-5000


#%% create non-linear colormap

from matplotlib.colors import LinearSegmentedColormap

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels/ self.levels.max()
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._y = np.linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)
        
        
levs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25]
cmap_lin = cmocean.tools.lighten(cmo.thermal, 0.85)
cmap_nonlin = nlcmap(cmap_lin, levs)

levs2 = [2, 3, 3.5, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.5, 7, 8, 9, 10, 12]
cmap_nonlin2 = nlcmap(cmap_lin, levs2)


#%%

# 1. create colour palettes
z100 = cmap_nonlin(np.linspace(0,1, len(levs)))
z200 = cmap_nonlin(np.linspace(0,1, len(levs)))
z400 = cmap_nonlin(np.linspace(0,1, len(levs)))
z500 = cmap_nonlin(np.linspace(0,1, len(levs)))
z750 = cmap_nonlin(np.linspace(0,1, len(levs2)))
z1500 = cmap_nonlin2(np.linspace(0,1, len(levs2)))
z3000 = cmap_nonlin(np.linspace(0,1, len(levs2)))
z5000 = cmap_nonlin2(np.linspace(0,1, len(levs2)))

# 2. bin the data into these colours
colors100 = np.zeros((len(d15n_obs_100m_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_100m_1d):
        if o >= n:
            colors100[j] = z100[i]
            if o < m:
                print(o)
                colors100[j] = z100[i]
colors200 = np.zeros((len(d15n_obs_200m_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_200m_1d):
        if o >= n:
            colors200[j] = z200[i]
            if o < m:
                print(o)
                colors200[j] = z200[i]
colors400 = np.zeros((len(d15n_obs_400m_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_400m_1d):
        if o >= n:
            colors400[j] = z400[i]
            if o < m:
                print(o)
                colors400[j] = z400[i]
colors500 = np.zeros((len(d15n_obs_500m_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_500m_1d):
        if o >= n:
            colors500[j] = z500[i]
            if o < m:
                print(o)
                colors500[j] = z500[i]
colors750 = np.zeros((len(d15n_obs_750m_1d),4))
for i,n in enumerate(levs2[0:-1]):
    m = levs2[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_750m_1d):
        if o >= n:
            colors750[j] = z750[i]
            if o < m:
                print(o)
                colors750[j] = z750[i]
colors1500 = np.zeros((len(d15n_obs_1500m_1d),4))
for i,n in enumerate(levs2[0:-1]):
    m = levs2[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_1500m_1d):
        if o >= n:
            colors1500[j] = z1500[i]
            if o < m:
                print(o)
                colors1500[j] = z1500[i]
colors3000 = np.zeros((len(d15n_obs_3000m_1d),4))
for i,n in enumerate(levs2[0:-1]):
    m = levs2[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_3000m_1d):
        if o >= n:
            colors3000[j] = z3000[i]
            if o < m:
                print(o)
                colors3000[j] = z3000[i]
colors5000 = np.zeros((len(d15n_obs_5000m_1d),4))
for i,n in enumerate(levs2[0:-1]):
    m = levs2[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_5000m_1d):
        if o >= n:
            colors5000[j] = z5000[i]
            if o < m:
                print(o)
                colors5000[j] = z5000[i]
                


#%% prepare figures

lat_labels = ['80$^{\circ}$S', ' ', '40$^{\circ}$S', ' ', '0$^{\circ}$', \
        ' ', '40$^{\circ}$N', ' ', '80$^{\circ}$N']
lon_labels = ['0$^{\circ}$E', '60$^{\circ}$E', '120$^{\circ}$E', '180$^{\circ}$E', '240$^{\circ}$E', '300$^{\circ}$E']


#%% prepare array for wrapping around longitudes

d15n_mod_100ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_100m[:,-1], (180,1)), d15n_mod_100m[:,:] ), axis=1 )
d15n_mod_200ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_200m[:,-1], (180,1)), d15n_mod_200m[:,:] ), axis=1 )
d15n_mod_400ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_400m[:,-1], (180,1)), d15n_mod_400m[:,:] ), axis=1 )
d15n_mod_500ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_500m[:,-1], (180,1)), d15n_mod_500m[:,:] ), axis=1 )
d15n_mod_750ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_750m[:,-1], (180,1)), d15n_mod_750m[:,:] ), axis=1 )
d15n_mod_1500ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_1500m[:,-1], (180,1)), d15n_mod_1500m[:,:] ), axis=1 )
d15n_mod_3000ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_3000m[:,-1], (180,1)), d15n_mod_3000m[:,:] ), axis=1 )
d15n_mod_5000ms = np.ma.concatenate( ( np.ma.reshape(d15n_mod_5000m[:,-1], (180,1)), d15n_mod_5000m[:,:] ), axis=1 )


#%% make figure

proj = ccrs.Robinson(central_longitude=0.0)

fslab = 15
fstic = 13
alfsc = 0.75
edc='grey'
lw=0.5
ms = 20

fig = plt.figure(facecolor='w', figsize=(11,20))
gs = GridSpec(4,2)

conts = np.array([0, 2, 4, 6, 8, 10, 15, 20, 25]); conts2 = np.array([3, 4, 4.5, 5, 5.5, 6, 7, 8, 10])

ax1 = plt.subplot(gs[0,0], projection=proj)
plt.title('0 - 100 m', family='sans-serif', fontsize=fslab)
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax1.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plot1 = plt.contourf(lons,lats,d15n_mod_100ms, cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both', transform=ccrs.PlateCarree())
#CS00 = plt.contour(lons,lats, d15n_mod_100ms, levels=conts, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[0], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons100_1d, llats100_1d, s=ms, c=colors100, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax2 = plt.subplot(gs[0,1], projection=proj)
plt.title('100 - 200 m', family='sans-serif', fontsize=fslab)
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plt.contourf(lons,lats,d15n_mod_200ms, cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both', transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[10], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons200_1d, llats200_1d, s=ms, c=colors200, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax3 = plt.subplot(gs[1,0], projection=proj)
plt.title('200 - 400 m', family='sans-serif', fontsize=fslab)
gl = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax3.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1, zorder=0)
plt.contourf(lons,lats,d15n_mod_400ms, cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both', transform=ccrs.PlateCarree())
#CS01 = plt.contour(lons,lats, d15n_mod_500ms, levels=conts, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[16], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons400_1d, llats400_1d, s=ms, c=colors400, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax4 = plt.subplot(gs[1,1], projection=proj)
plt.title('500 m', family='sans-serif', fontsize=fslab)
gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax4.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plt.contourf(lons,lats,d15n_mod_500ms, cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both', transform=ccrs.PlateCarree())
#CS01 = plt.contour(lons,lats, d15n_mod_500ms, levels=conts, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[19], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons500_1d, llats500_1d, s=ms, c=colors500, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax5 = plt.subplot(gs[2,0], projection=proj)
plt.title('750 m', family='sans-serif', fontsize=fslab)
gl = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax5.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plot2 = plt.contourf(lons,lats,d15n_mod_750ms, cmap=cmap_nonlin, corner_mask=True, levels=levs2, extend='both', transform=ccrs.PlateCarree())
#CS10 = plt.contour(lons,lats, d15n_mod_1500ms, levels=conts, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[20], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons750_1d, llats750_1d, s=ms, c=colors750, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax6 = plt.subplot(gs[2,1], projection=proj)
plt.title('1000 - 1500 m', family='sans-serif', fontsize=fslab)
gl = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax6.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plot2 = plt.contourf(lons,lats,d15n_mod_1500ms, cmap=cmap_nonlin, corner_mask=True, levels=levs2, extend='both', transform=ccrs.PlateCarree())
#CS10 = plt.contour(lons,lats, d15n_mod_1500ms, levels=conts, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[21], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons1500_1d, llats1500_1d, s=ms, c=colors1500, marker='o',  lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax7 = plt.subplot(gs[3,0], projection=proj)
plt.title('1500 - 3000 m', family='sans-serif', fontsize=fslab)
gl = ax7.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax7.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plt.contourf(lons,lats,d15n_mod_3000ms, cmap=cmap_nonlin, corner_mask=True, levels=levs2, extend='both', transform=ccrs.PlateCarree())
#CS11 = plt.contour(lons,lats, d15n_mod_5000ms, levels=conts2, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[22], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons3000_1d, llats3000_1d, s=ms, c=colors3000, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())

ax8 = plt.subplot(gs[3,1], projection=proj)
plt.title('3000 - 5000 m', family='sans-serif', fontsize=fslab)
gl = ax8.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax8.gridlines(linestyle='--', linewidth=0.5, color='grey', alpha=0.1)
plt.contourf(lons,lats,d15n_mod_5000ms, cmap=cmap_nonlin, corner_mask=True, levels=levs2, extend='both', transform=ccrs.PlateCarree())
#CS11 = plt.contour(lons,lats, d15n_mod_5000ms, levels=conts2, colors='k', alpha=0.75, linewidths=1, zorder=1, transform=ccrs.PlateCarree())
plt.pcolormesh(lons,lats,land[26], cmap='Greys_r', zorder=2, transform=ccrs.PlateCarree())
scat = plt.scatter(llons5000_1d, llats5000_1d, s=ms, c=colors5000, marker='o', lw=lw, zorder=2, alpha=alfsc, edgecolor=edc, transform=ccrs.PlateCarree())


plt.subplots_adjust(wspace=0.05, left=0.05, top=0.95, bottom=0.05, right=0.87)


#%%

cbar_ax1 = fig.add_axes([0.88, 0.58, 0.025, 0.32])
cbar1 = fig.colorbar(plot1, cax=cbar_ax1, orientation='vertical', ticks=levs[::2])
cbar1.set_label('$\delta^{15}$N$_{NO_3}$ (\u2030)', fontsize=fslab, family='sans-serif')
plt.yticks(levs[::2],levs[::2], fontsize=fstic)

cbar_ax2 = fig.add_axes([0.88, 0.13, 0.025, 0.32])
cbar2 = fig.colorbar(plot2, cax=cbar_ax2, orientation='vertical', ticks=levs2[::2])
cbar2.set_label('$\delta^{15}$N$_{NO_3}$ (\u2030)', fontsize=fslab, family='sans-serif')
plt.yticks(levs2[::2],levs2[::2], fontsize=fstic)


#%%

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp2_alt.pdf', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp2_alt.png', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp2_alt_trans.png', dpi=300, bbox_inches='tight', transparent=True)



#%% now that we have the averages for each equivalent grid point, I can take weighted averages in space

# first make 3D masks
basins = np.zeros(np.shape(d15n))
for x in np.arange(len(basins[0,0,:])):
    for y in np.arange(len(basins[0,:,0])):
        if atlmask[y,x] == 1:
            basins[:,y,x] = 1
        if pacmask[y,x] == 1:
            basins[:,y,x] = 2
        if indmask[y,x] == 1:
            basins[:,y,x] = 3
        if aramask[y,x] == 1:
            basins[:,y,x] = 4
        if arpmask[y,x] == 1:
            basins[:,y,x] = 5

glob = np.zeros(np.shape(d15n))
for x in np.arange(len(basins[0,0,:])):
    for y in np.arange(len(basins[0,:,0])):
        if glomask[y,x] == 1:
            glob[:,y,x] = 1



### meridional space
d15n_obs_Glo = np.ma.masked_where(glob!=1, d15n_obs)
d15n_obs_Atl = np.ma.masked_where(basins!=1, d15n_obs)
d15n_obs_Pac = np.ma.masked_where(basins!=2, d15n_obs)         
d15n_obs_Ind = np.ma.masked_where(basins!=3, d15n_obs)         
d15n_obs_ArA = np.ma.masked_where(basins!=4, d15n_obs)         
d15n_obs_ArP = np.ma.masked_where(basins!=5, d15n_obs)         
d15n_obs_Arc = np.ma.masked_where(basins<4, d15n_obs)         

d15n_obs_Glox = np.ma.average(d15n_obs_Glo, axis=2, weights=vol)
d15n_obs_Atlx = np.ma.average(d15n_obs_Atl, axis=2, weights=vol)
d15n_obs_Pacx = np.ma.average(d15n_obs_Pac, axis=2, weights=vol)
d15n_obs_Indx = np.ma.average(d15n_obs_Ind, axis=2, weights=vol)

# put back onto 1d arrays for scattering
llats, ddeps = np.meshgrid(lat, dep)
maskobsGlo = np.ma.getmask(d15n_obs_Glox)
maskobsAtl = np.ma.getmask(d15n_obs_Atlx)
maskobsPac = np.ma.getmask(d15n_obs_Pacx)
maskobsInd = np.ma.getmask(d15n_obs_Indx)

ddepsGlo = np.ma.masked_where(maskobsGlo, ddeps); llatsGlo = np.ma.masked_where(maskobsGlo, llats)
ddepsAtl = np.ma.masked_where(maskobsAtl, ddeps); llatsAtl = np.ma.masked_where(maskobsAtl, llats)
ddepsPac = np.ma.masked_where(maskobsPac, ddeps); llatsPac = np.ma.masked_where(maskobsPac, llats)
ddepsInd = np.ma.masked_where(maskobsInd, ddeps); llatsInd = np.ma.masked_where(maskobsInd, llats)

d15n_obs_Glox_1d = np.ma.MaskedArray.compressed(d15n_obs_Glox); ddepsGbl_1d = np.ma.MaskedArray.compressed(ddepsGlo); llatsGlo_1d = np.ma.MaskedArray.compressed(llatsGlo)
d15n_obs_Atlx_1d = np.ma.MaskedArray.compressed(d15n_obs_Atlx); ddepsAtl_1d = np.ma.MaskedArray.compressed(ddepsAtl); llatsAtl_1d = np.ma.MaskedArray.compressed(llatsAtl)
d15n_obs_Pacx_1d = np.ma.MaskedArray.compressed(d15n_obs_Pacx); ddepsPac_1d = np.ma.MaskedArray.compressed(ddepsPac); llatsPac_1d = np.ma.MaskedArray.compressed(llatsPac)
d15n_obs_Indx_1d = np.ma.MaskedArray.compressed(d15n_obs_Indx); ddepsInd_1d = np.ma.MaskedArray.compressed(ddepsInd); llatsInd_1d = np.ma.MaskedArray.compressed(llatsInd)


print(np.min(d15n_obs_Glox_1d), np.max(d15n_obs_Glox_1d))
print(np.min(d15n_obs_Atlx_1d), np.max(d15n_obs_Atlx_1d))
print(np.min(d15n_obs_Pacx_1d), np.max(d15n_obs_Pacx_1d))
print(np.min(d15n_obs_Indx_1d), np.max(d15n_obs_Indx_1d))


##%% model zonal averages

### meridional space
d15n_mod_Glo = np.ma.masked_where(glob!=1, d15n)
d15n_mod_Atl = np.ma.masked_where(basins!=1, d15n)
d15n_mod_Pac = np.ma.masked_where(basins!=2, d15n)         
d15n_mod_Ind = np.ma.masked_where(basins!=3, d15n)         
d15n_mod_ArA = np.ma.masked_where(basins!=4, d15n)         
d15n_mod_ArP = np.ma.masked_where(basins!=5, d15n)         
d15n_mod_Arc = np.ma.masked_where(basins<4, d15n)         

d15n_mod_Glox = np.ma.average(d15n_mod_Glo, axis=2, weights=vol)
d15n_mod_Atlx = np.ma.average(d15n_mod_Atl, axis=2, weights=vol)
d15n_mod_Pacx = np.ma.average(d15n_mod_Pac, axis=2, weights=vol)
d15n_mod_Indx = np.ma.average(d15n_mod_Ind, axis=2, weights=vol)


#%% Compare model and observations

print("Global mean d15N from observations = ", np.ma.average(d15n_obs_Glo, weights=vol), "+/-", np.ma.std(d15n_obs_Glo))
print("Atlantic mean d15N from observations = ", np.ma.average(d15n_obs_Atl, weights=vol), "+/-", np.ma.std(d15n_obs_Atl))
print("Pacific mean d15N from observations = ", np.ma.average(d15n_obs_Pac, weights=vol), "+/-", np.ma.std(d15n_obs_Pac))
print("Indian mean d15N from observations = ", np.ma.average(d15n_obs_Ind, weights=vol), "+/-", np.ma.std(d15n_obs_Ind))
print("Southern Ocean mean d15N from observations = ", np.ma.average(d15n_obs_Glo[:,0:50,:], weights=vol[:,0:50,:]), "+/-", np.ma.std(d15n_obs_Glo[:,0:50,:]))
print("Arctic mean d15N from observations = ", np.ma.average(d15n_obs_Arc, weights=vol), "+/-", np.ma.std(d15n_obs_Arc))
print("Arctic (Atl) mean d15N from observations = ", np.ma.average(d15n_obs_ArA, weights=vol), "+/-", np.ma.std(d15n_obs_ArA))
print("Arctic (Pac) mean d15N from observations = ", np.ma.average(d15n_obs_ArP, weights=vol), "+/-", np.ma.std(d15n_obs_ArP))
print(" ")
print("Global mean d15N from simulation = ", np.ma.average(d15n_mod_Glo, weights=vol), "+/-", np.ma.std(d15n_mod_Glo))
print("Atlantic mean d15N from simulation = ", np.ma.average(d15n_mod_Atl, weights=vol), "+/-", np.ma.std(d15n_mod_Atl))
print("Pacific mean d15N from simulation = ", np.ma.average(d15n_mod_Pac, weights=vol), "+/-", np.ma.std(d15n_mod_Pac))
print("Indian mean d15N from simulation = ", np.ma.average(d15n_mod_Ind, weights=vol), "+/-", np.ma.std(d15n_mod_Ind))
print("Southern Ocean mean d15N from simulation = ", np.ma.average(d15n_mod_Glo[:,0:50,:], weights=vol[:,0:50,:]), "+/-", np.ma.std(d15n_mod_Glo[:,0:50,:]))
print("Arctic mean d15N from simulation = ", np.ma.average(d15n_mod_Arc, weights=vol), "+/-", np.ma.std(d15n_mod_Arc))
print("Arctic (Atl) mean d15N from simulation = ", np.ma.average(d15n_mod_ArA, weights=vol), "+/-", np.ma.std(d15n_mod_ArA))
print("Arctic (Pac) mean d15N from simulation = ", np.ma.average(d15n_mod_ArP, weights=vol), "+/-", np.ma.std(d15n_mod_ArP))


#%% create non-linear colormap

levs = [3,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0,7.2,7.4,7.6,7.8,8.0,8.2,8.4,8.6,8.8,9.0,9.2,9.4,9.6,9.8,10.0]
cmap_nonlin = cmocean.tools.lighten(cmo.thermal, 0.85)
#cmap_nonlin = nlcmap(cmap_lin, levs)


# 1. create colour palettes
zGbl = cmap_nonlin(np.linspace(0,1, len(levs)))
zPac= cmap_nonlin(np.linspace(0,1, len(levs)))
zAtl = cmap_nonlin(np.linspace(0,1, len(levs)))
zInd = cmap_nonlin(np.linspace(0,1, len(levs)))

# 2. bin the data into these colours
colorsGbl = np.zeros((len(d15n_obs_Glox_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_Glox_1d):
        if o >= n:
            colorsGbl[j] = zGbl[i]
            if o < m:
                print(o)
                colorsGbl[j] = zGbl[i]

colorsAtl = np.zeros((len(d15n_obs_Atlx_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_Atlx_1d):
        if o >= n:
            colorsAtl[j] = zAtl[i]
            if o < m:
                print(o)
                colorsAtl[j] = zAtl[i]

colorsPac = np.zeros((len(d15n_obs_Pacx_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_Pacx_1d):
        if o >= n:
            colorsPac[j] = zPac[i]
            if o < m:
                print(o)
                colorsPac[j] = zPac[i]

colorsInd = np.zeros((len(d15n_obs_Indx_1d),4))
for i,n in enumerate(levs[0:-1]):
    m = levs[i+1]
    print(i,n,m)
    for j,o in enumerate(d15n_obs_Indx_1d):
        if o >= n:
            colorsInd[j] = zInd[i]
            if o < m:
                print(o)
                colorsInd[j] = zInd[i]


#%% plot 2D surfaces in meridional space (gbl)(atl)(pac)(ind)

fslab = 15
fstic = 13
alfsc = 0.7

fig = plt.figure(facecolor='w', figsize=(16,9))
gs = GridSpec(100,2)

conts = np.array([1, 3, 4, 4.5, 5, 5.5, 6, 8, 10, 12])

ax1a = plt.subplot(gs[0:12,0])
ax1a.tick_params(axis='both', which='both', labelbottom=False, bottom=False)
ax1a.spines['bottom'].set_visible(False)
plt.title('Global', family='sans-serif', fontsize=fslab)
plot1a = plt.contourf(lat,dep[0:17],d15n_mod_Glox[0:17,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS00a = plt.contour(lat,dep[0:17], d15n_mod_Glox[0:17,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
ax1a.set_facecolor('silver')
#scat = plt.scatter(llatsGbl_1d, ddepsGbl_1d, s=20, c=colorsGbl, marker='o', zorder=2, alpha=alfsc)
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(215,2)
plt.yticks(np.arange(200,0,-50), np.arange(200,0,-50), family='sans-serif', fontsize=fstic)

ax1b = plt.subplot(gs[12:45,0])
ax1b.tick_params(axis='both', which='both', labelbottom=False)
ax1b.spines['top'].set_visible(False)
plot1b = plt.contourf(lat,dep[17:31],d15n_mod_Glox[17:31,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS00b = plt.contour(lat,dep[17:31], d15n_mod_Glox[17:31,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
ax1b.set_facecolor('silver')
#scat = plt.scatter(llatsGbl_1d, ddepsGbl_1d, s=20, c=colorsGbl, marker='o', zorder=2, alpha=alfsc)
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=12)
plt.xlim(-85,88); plt.ylim(5000,260)
plt.yticks(np.arange(5000,0,-1000), np.arange(5000,0,-1000), family='sans-serif', fontsize=fstic)


ax2a = plt.subplot(gs[0:12,1])
ax2a.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labelleft=False)
ax2a.spines['bottom'].set_visible(False)
plt.title('Atlantic', family='sans-serif', fontsize=fslab)
plot2a = plt.contourf(lat,dep[0:17],d15n_mod_Atlx[0:17,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS01a = plt.contour(lat,dep[0:17], d15n_mod_Atlx[0:17,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat1a = plt.scatter(llatsAtl_1d, ddepsAtl_1d, s=20, c=colorsAtl, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax2a.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(215,2)
plt.yticks(np.arange(200,0,-50), np.arange(200,0,-50), family='sans-serif', fontsize=fstic)

ax2b = plt.subplot(gs[12:45,1])
ax2b.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
ax2b.spines['top'].set_visible(False)
plot2b = plt.contourf(lat,dep[17:31],d15n_mod_Atlx[17:31,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS01b = plt.contour(lat,dep[17:31], d15n_mod_Atlx[17:31,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat1b = plt.scatter(llatsAtl_1d, ddepsAtl_1d, s=20, c=colorsAtl, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax2b.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(5000,260)
plt.yticks(np.arange(5000,0,-1000), np.arange(5000,0,-1000), family='sans-serif', fontsize=fstic)


ax3a = plt.subplot(gs[55:67,0])
ax3a.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labelleft=True)
ax3a.spines['bottom'].set_visible(False)
plt.title('Pacific', family='sans-serif', fontsize=fslab)
plot3a = plt.contourf(lat,dep[0:17],d15n_mod_Pacx[0:17,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS10a = plt.contour(lat,dep[0:17], d15n_mod_Pacx[0:17,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat3a = plt.scatter(llatsPac_1d, ddepsPac_1d, s=20, c=colorsPac, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax3a.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(215,2)
plt.yticks(np.arange(200,0,-50), np.arange(200,0,-50), family='sans-serif', fontsize=fstic)

ax3b = plt.subplot(gs[67:100,0])
ax3b.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)
ax3b.spines['top'].set_visible(False)
plot3b = plt.contourf(lat,dep[17:31],d15n_mod_Pacx[17:31,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS10b = plt.contour(lat,dep[17:31], d15n_mod_Pacx[17:31,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat3b = plt.scatter(llatsPac_1d, ddepsPac_1d, s=20, c=colorsPac, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax3b.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(5000,260)
plt.yticks(np.arange(5000,0,-1000), np.arange(5000,0,-1000), family='sans-serif', fontsize=fstic)


ax4a = plt.subplot(gs[55:67,1])
ax4a.tick_params(axis='both', which='both', labelbottom=False, bottom=False, labelleft=False)
ax4a.spines['bottom'].set_visible(False)
plt.title('Indian', family='sans-serif', fontsize=fslab)
plot4a = plt.contourf(lat,dep[0:17],d15n_mod_Indx[0:17,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS11a = plt.contour(lat,dep[0:17], d15n_mod_Indx[0:17,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat4a = plt.scatter(llatsInd_1d, ddepsInd_1d, s=20, c=colorsInd, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax4a.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(215,2)
plt.yticks(np.arange(200,0,-50), np.arange(200,0,-50), family='sans-serif', fontsize=fstic)

ax4b = plt.subplot(gs[67:100,1])
ax4b.tick_params(axis='both', which='both', labelbottom=True, labelleft=False)
ax4b.spines['top'].set_visible(False)
plot4b = plt.contourf(lat,dep[17:31],d15n_mod_Indx[17:31,...], cmap=cmap_nonlin, corner_mask=True, levels=levs, extend='both')
CS11b = plt.contour(lat,dep[17:31], d15n_mod_Indx[17:31,...], levels=conts, colors='k', alpha=0.75, linewidths=1)
scat4b = plt.scatter(llatsInd_1d, ddepsInd_1d, s=20, c=colorsInd, marker='o', zorder=2, alpha=alfsc, edgecolor='grey', lw=0.5)
ax4b.set_facecolor('silver')
plt.xticks(np.arange(-80,90,20), lat_labels, family='sans-serif', fontsize=fstic)
plt.xlim(-85,88); plt.ylim(5000,260)
plt.yticks(np.arange(5000,0,-1000), np.arange(5000,0,-1000), family='sans-serif', fontsize=fstic)


plt.subplots_adjust(wspace=0.05, right=0.87, bottom=0.08, top=0.92, left=0.06)


#%%

cbar_ax1 = fig.add_axes([0.9, 0.1, 0.025, 0.8])
cbar1 = fig.colorbar(plot1a, cax=cbar_ax1, orientation='vertical', ticks=levs[::2])
cbar1.set_label('$\delta^{15}$N$_{NO_3}$ (\u2030)', fontsize=fslab, family='sans-serif')
plt.yticks(levs[::2], levs[::2], fontsize=fstic)


#%%

fscon = 11
col1 = 'k'
col2 = 'grey'

plt.clabel(CS00a, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS00b, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS01a, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS01b, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS10a, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS10b, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS11a, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS11b, fontsize=fscon, colors=col1, inline_spacing=4, fmt='%.1f', manual=True)

plt.clabel(CS00a, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS00b, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS01a, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS01b, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS10a, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS10b, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS11a, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)
plt.clabel(CS11b, fontsize=fscon, colors=col2, inline_spacing=4, fmt='%.1f', manual=True)


os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//figures")
fig.savefig('fig-d15n_zonalsections.pdf', dpi=300, bbox_inches='tight')
fig.savefig('fig-d15n_zonalsections.png', dpi=300, bbox_inches='tight')
os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//figures")
fig.savefig('fig-d15n_zonalsections_transparent.png', dpi=300, bbox_inches='tight')



#%% Finally, return statistical measures of fit for global ocean and different regions

def summary_stats_comparison(d1,d2,weight):
    '''
    Finds the global average and range of two dataset, and returns simple comparitative 
    statistics including the root mean square error, modeling efficiency and correlation coefficient.
    
    d1 = model dataset
    d2 = observational dataset
    
    Weights for each dataset may be supplied as options.
    
    '''
    
    d1ave = np.ma.average(d1, weights=weight)
    d2ave = np.ma.average(d2, weights=weight)
    
    d1range = [np.ma.min(d1), np.ma.max(d1)]
    d2range = [np.ma.min(d2), np.ma.max(d2)]
    
    d1std = np.ma.std(d1)
    d2std = np.ma.std(d2)
    
    d1Nstd = d1std/d2std
    d2Nstd = d2std/d2std

    # RMSE
    a = (d1-d2)**2.0
    n = np.ma.count(a+100)
    RMSE = ((np.ma.sum(a)/n)**0.5)
    
    # bias / std of obs
    bias = (np.ma.sum((d1-d2))/np.ma.count(d1))
    print(np.ma.sum(d1-d2), np.ma.count(d1))
    nbias = bias/d2std 
    
    # Reliability Index
    ri = np.exp(((1./n) * np.ma.sum(np.log(d2/d1)**2.0))**0.5)
    
    # Modeling efficiency
    a = np.ma.sum((d2-d2ave)**2.0)
    b = np.ma.sum((d1-d2)**2.0)
    mef = (a-b)/a
    
    # correlation coefficiency
    a = np.ma.sum((d2-d2ave)*(d1-d1ave))
    b = np.ma.sum((d2-d2ave)**2.0)
    c = np.ma.sum((d1-d1ave)**2.0)
    r = a/((b*c)**0.5)
    

    return d1ave, d2ave, bias, nbias, d1Nstd, RMSE, r, d1std, d2std


#%%

mask_obs_Glo = np.ma.getmask(d15n_obs_Glo)
mask_obs_Atl = np.ma.getmask(d15n_obs_Atl)
mask_obs_Pac = np.ma.getmask(d15n_obs_Pac)
mask_obs_Ind = np.ma.getmask(d15n_obs_Ind)
mask_obs_Arc = np.ma.getmask(d15n_obs_Arc)
mask_obs_ArA = np.ma.getmask(d15n_obs_ArA)
mask_obs_ArP = np.ma.getmask(d15n_obs_ArP)

d15n_mod_Glom = np.ma.masked_where(mask_obs_Glo, d15n)
d15n_mod_Atlm = np.ma.masked_where(mask_obs_Atl, d15n)
d15n_mod_Pacm = np.ma.masked_where(mask_obs_Pac, d15n)
d15n_mod_Indm = np.ma.masked_where(mask_obs_Ind, d15n)
d15n_mod_Arcm = np.ma.masked_where(mask_obs_Arc, d15n)
d15n_mod_ArAm = np.ma.masked_where(mask_obs_ArA, d15n)
d15n_mod_ArPm = np.ma.masked_where(mask_obs_ArP, d15n)


# check lengths of arrays
print(np.ma.count_masked(d15n_obs_Glo), np.ma.count_masked(d15n_mod_Glom))
print(np.ma.count_masked(d15n_obs_Atl), np.ma.count_masked(d15n_mod_Atlm))
print(np.ma.count_masked(d15n_obs_Pac), np.ma.count_masked(d15n_mod_Pacm))
print(np.ma.count_masked(d15n_obs_Ind), np.ma.count_masked(d15n_mod_Indm))
print(np.ma.count_masked(d15n_obs_Arc), np.ma.count_masked(d15n_mod_Arcm))
print(np.ma.count_masked(d15n_obs_ArA), np.ma.count_masked(d15n_mod_ArAm))
print(np.ma.count_masked(d15n_obs_ArP), np.ma.count_masked(d15n_mod_ArPm))

print("Global N = ",np.ma.count(d15n_obs_Glo), np.ma.count(d15n_mod_Glom))
print("Atlantic N = ",np.ma.count(d15n_obs_Atl), np.ma.count(d15n_mod_Atlm))
print("Pacific N = ",np.ma.count(d15n_obs_Pac), np.ma.count(d15n_mod_Pacm))
print("Indian N = ",np.ma.count(d15n_obs_Ind), np.ma.count(d15n_mod_Indm))
print("Arctic N = ",np.ma.count(d15n_obs_Arc), np.ma.count(d15n_mod_Arcm))
print("Arctic (Atl) N = ",np.ma.count(d15n_obs_ArA), np.ma.count(d15n_mod_ArAm))
print("Arctic (Pac) N = ",np.ma.count(d15n_obs_ArP), np.ma.count(d15n_mod_ArPm))


stats_glo = []
stats_glo.append(summary_stats_comparison(d15n_mod_Glom, d15n_obs_Glo, weight=vol))

stats_sth = []
stats_sth.append(summary_stats_comparison(d15n_mod_Glom[:,0:50,:], d15n_obs_Glo[:,0:50,:], weight=vol[:,0:50,:])) #-90:-40 deg S

stats_atl = []
stats_atl.append(summary_stats_comparison(d15n_mod_Atlm[:,50:180,:], d15n_obs_Atl[:,50:180,:], weight=vol[:,50:180,:]))

stats_pac = []
stats_pac.append(summary_stats_comparison(d15n_mod_Pacm[:,50:180,:], d15n_obs_Pac[:,50:180,:], weight=vol[:,50:180,:]))

stats_ind = []
stats_ind.append(summary_stats_comparison(d15n_mod_Indm[:,50:180,:], d15n_obs_Ind[:,50:180,:], weight=vol[:,50:180,:]))

stats_arc = []
stats_arc.append(summary_stats_comparison(d15n_mod_Arcm[:,:,:], d15n_obs_Arc[:,:,:], weight=vol[:,:,:]))

stats_ara = []
stats_ara.append(summary_stats_comparison(d15n_mod_ArAm[:,:,:], d15n_obs_ArA[:,:,:], weight=vol[:,:,:]))

stats_arp = []
stats_arp.append(summary_stats_comparison(d15n_mod_ArPm[:,:,:], d15n_obs_ArP[:,:,:], weight=vol[:,:,:]))

'''
remember, output is:
    average model | average obs | model bias | normalised model bias | normalised model standard deviation | root mean square error | correlation
'''

print(stats_glo)
print(stats_sth)
print(stats_atl)
print(stats_pac)
print(stats_ind)
print(stats_arc)
print(stats_ara)
print(stats_arp)


#%% save univariate measures of fit

os.chdir('C:\\Users\\pearseb\\Dropbox\\PostDOc\\my articles\\d15N and d13C in PISCES\\data_for_publication')

d15nstats = np.zeros((8,9)) 
d15nstats[0,:] = np.array(stats_glo)
d15nstats[1,:] = np.array(stats_sth)
d15nstats[2,:] = np.array(stats_atl)
d15nstats[3,:] = np.array(stats_pac)
d15nstats[4,:] = np.array(stats_ind)
d15nstats[5,:] = np.array(stats_arc)
d15nstats[6,:] = np.array(stats_ara)
d15nstats[7,:] = np.array(stats_arp)
np.savetxt('d15nstats.txt', d15nstats, fmt='%.4f', delimiter=',')




