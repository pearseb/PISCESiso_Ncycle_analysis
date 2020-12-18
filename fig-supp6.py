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

data = nc.Dataset('figure_depthzones.nc','r')
pic_ez = data.variables['EUPH_PIC'][...]
pic_utz = data.variables['UTZ_PIC'][...]
pic_ltz = data.variables['LTZ_PIC'][...]
lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]


#%% mask where euph/bot = 0

pic_ez = np.ma.masked_where(pic_ez == 0.0, pic_ez)
pic_utz = np.ma.masked_where(pic_utz == 0.0, pic_utz)
pic_ltz = np.ma.masked_where(pic_ltz == 0.0, pic_ltz)


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

colmap = cmocean.tools.lighten(cmo.deep,0.75)
levs1 = np.arange(80,181,5)
levs2 = np.arange(300,451,10)
#levs2 = np.concatenate((np.array([-2,-1.5]), np.arange(-1.2,1.21,0.1), np.array([1.5, 2])))

fstic = 14
fslab = 15
alf = 0.7


fig = plt.figure(figsize=(8,6.5), facecolor='w')

gs = GridSpec(2,1)

ax1 = plt.subplot(gs[0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, pic_ez, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c1 = plt.contour(lonproj,latproj, pic_ez, levels=np.array([100, 150]))
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
#proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

plt.text(0.5,1.05, 'Euphotic zone (EZ)', transform=ax1.transAxes, fontsize=fslab, ha='center', va='center')

plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.85)
cbax1 = fig.add_axes([0.85, 0.57, 0.03, 0.37])
cbar1 = plt.colorbar(p1, cax=cbax1, orientation='vertical', ticks=levs1[::2])
cbar1.ax.set_ylabel('Depth (metres)', fontsize=fslab)
cbar1.ax.tick_params(labelsize=fstic)


ax2 = plt.subplot(gs[1])
ax2.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, pic_utz, levels=levs2, cmap=colmap, vmin=np.min(levs2), vmax=np.max(levs2), extend='both')
c2 = plt.contour(lonproj,latproj, pic_utz, levels=np.array([350, 400]))
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

plt.text(0.5,1.05, 'Twilight Zone (TZ)', transform=ax2.transAxes, fontsize=fslab, ha='center', va='center')

cbax2 = fig.add_axes([0.85, 0.11, 0.03, 0.37])
cbar2 = plt.colorbar(p2, cax=cbax2, orientation='vertical', ticks=levs2[::2])
cbar2.ax.set_ylabel('Depth (metres)', fontsize=fslab)
cbar2.ax.tick_params(labelsize=fstic)


#%%

plt.text(0.05,1.05, 'a)', transform=ax1.transAxes, fontsize=fslab+2, va='center', ha='center')
plt.text(0.05,1.05, 'b)', transform=ax2.transAxes, fontsize=fslab+2, va='center', ha='center')

plt.clabel(c1, manual=True, fontsize=fslab, fmt='%i')
plt.clabel(c2, manual=True, fontsize=fslab, fmt='%i')

#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp6.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp6.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp6_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)
