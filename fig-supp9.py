# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

Use Ferret script "define_twilight.jnl"

@author: pearseb
"""

#%% imports

import os
import numpy as np
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sb
sb.set(style='ticks')
import cmocean
import cmocean.cm as cmo

import mpl_toolkits.basemap as bm


#%% data

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")


data = nc.Dataset('figure2D_ndep_no3_utz.nc','r')
no3_utz = data.variables['NO3_FUT_UTZ_AVE'][...]
lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]



#%% prepare things for figure


lat_labs = ['80$^{\circ}$S', '60$^{\circ}$S', '40$^{\circ}$S', '20$^{\circ}$S', '0$^{\circ}$', \
        '20$^{\circ}$N', '40$^{\circ}$N', '60$^{\circ}$N', '80$^{\circ}$N']
lon_labs = ['0$^{\circ}$E', '50$^{\circ}$E', '100$^{\circ}$E', '150$^{\circ}$E', '200$^{\circ}$E', \
        '250$^{\circ}$E', '300$^{\circ}$E', '350$^{\circ}$E']

    
domain = [-61,-340,61,20]                
domain_draw = [-60,-340,60,20]
dlat=20
dlon=60

xx,yy = np.meshgrid(lon, lat)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='c')
lonproj, latproj = proj(xx, yy)



lat_labs2 = ['40$^{\circ}$S', '30$^{\circ}$S', '20$^{\circ}$S', '10$^{\circ}$S', '0$^{\circ}$', \
        '10$^{\circ}$N', '20$^{\circ}$N', '30$^{\circ}$N', '40$^{\circ}$N']



#%% figure (EZ, UTZ)

colmap = cmocean.tools.lighten(cmo.deep_r,0.75)
levs1 = np.arange(0,31,2)

fstic = 14
fslab = 15
alf = 0.7


fig = plt.figure(figsize=(10,6), facecolor='w')

gs = GridSpec(1,1)

ax1 = plt.subplot(gs[0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, no3_utz, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

plt.text(0.5,1.05, 'Twilight zone NO$_3$', transform=ax1.transAxes, fontsize=fslab, ha='center', va='center')

plt.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.9)

cbax1 = fig.add_axes([0.2, 0.125, 0.6, 0.05])
cbar1 = plt.colorbar(p1, cax=cbax1, orientation='horizontal', ticks=levs1[::2])
cbar1.ax.set_xlabel('mmol m$^{-3}$', fontsize=fslab)
cbar1.ax.tick_params(labelsize=fstic)


#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp9.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp9.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp9_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)
