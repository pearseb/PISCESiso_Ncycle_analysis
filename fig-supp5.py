# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

used in conjuction with ncycle_changes_climatechangeonly.jnl in /users/pearseb/analysis_picontrolfuture/


@author: pearseb
"""

#%% imports

from __future__ import unicode_literals

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


#%% get data

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

data = nc.Dataset('ncycle_changes_climatechangeonly.nc','r')
delta_dep = np.ma.squeeze(data.variables['DELTA_DEP'][...])
delta_fix = np.ma.squeeze(data.variables['DELTA_FIX'][...])
delta_wcd = np.ma.squeeze(data.variables['DELTA_WCD'][...])
delta_sed = np.ma.squeeze(data.variables['DELTA_SED'][...])
delta_npp = np.ma.squeeze(data.variables['DELTA_NPP'][...])
delta_zoo = np.ma.squeeze(data.variables['DELTA_ZOO'][...])

lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]

data.close()




#%% prepare things for figure


lat_labs = ['80$^{\circ}$S', '60$^{\circ}$S', '40$^{\circ}$S', '20$^{\circ}$S', '0$^{\circ}$', \
        '20$^{\circ}$N', '40$^{\circ}$N', '60$^{\circ}$N', '80$^{\circ}$N']
lon_labs = ['0$^{\circ}$E', '50$^{\circ}$E', '100$^{\circ}$E', '150$^{\circ}$E', '200$^{\circ}$E', \
        '250$^{\circ}$E', '300$^{\circ}$E', '350$^{\circ}$E']


domain = [-51,-340,51,20]                
domain_draw = [-50,-340,50,20]
dlat=20
dlon=60

lons,lats = np.meshgrid(lon,lat)
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='c')
lonproj, latproj = proj(lons, lats)



# Tableau 20 Colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
             
# Tableau Color Blind 10
tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207)]
  
# Rescale to values between 0 and 1 
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)
for i in range(len(tableau20blind)):  
    r, g, b = tableau20blind[i]  
    tableau20blind[i] = (r / 255., g / 255., b / 255.)




#%% figure of d15N-no3 signals and significance at 95% level (EZ and UTZ)

colmap = cmocean.tools.lighten(cmo.balance,0.75)
levs = np.arange(-1,1.01,0.05)
conts = np.array([-0.5, -0.1, 0.1, 0.5])
levs1 = np.arange(-20,20.01,1)
conts1 = np.array([-10, -2, 2, 10])

fstic = 13
fslab = 15
alf = 0.7
wid=0.5


fig = plt.figure(figsize=(16,8.5), facecolor='w')
gs = GridSpec(3,2)

ax1 = plt.subplot(gs[0,0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, delta_dep*0, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c1 = plt.contour(lonproj,latproj, delta_dep*0, colors='k', linewidths=wid, levels=conts, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ nitrogen deposition = 0.0 Tg N yr$^{-1}$', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')

ax2 = plt.subplot(gs[0,1])
ax2.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, delta_fix, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c2 = plt.contour(lonproj,latproj, delta_fix, colors='k', linewidths=wid, levels=conts, linestyles='-')
#proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ nitrogen fixation = 7.0 Tg N yr$^{-1}$', transform=ax2.transAxes, fontsize=fstic, ha='center', va='center')

ax3 = plt.subplot(gs[1,0])
ax3.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p3 = plt.contourf(lonproj,latproj, delta_wcd, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c3 = plt.contour(lonproj,latproj, delta_wcd, colors='k', linewidths=wid, levels=conts, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ water column denitrification = -1.4 Tg N yr$^{-1}$', transform=ax3.transAxes, fontsize=fstic, ha='center', va='center')

ax4 = plt.subplot(gs[1,1])
ax4.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p4 = plt.contourf(lonproj,latproj, delta_sed, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c4 = plt.contour(lonproj,latproj, delta_sed, colors='k', linewidths=wid, levels=conts, linestyles='-')
#proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ sedimentary denitrification = -12.2 Tg N yr$^{-1}$', transform=ax4.transAxes, fontsize=fstic, ha='center', va='center')

ax5 = plt.subplot(gs[2,0])
ax5.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p5 = plt.contourf(lonproj,latproj, delta_npp, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c5 = plt.contour(lonproj,latproj, delta_npp, colors='k', linewidths=wid, levels=conts1, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ net primary production = -544 Tg N yr$^{-1}$', transform=ax5.transAxes, fontsize=fstic, ha='center', va='center')

ax6 = plt.subplot(gs[2,1])
ax6.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p6 = plt.contourf(lonproj,latproj, delta_zoo, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c6 = plt.contour(lonproj,latproj, delta_zoo, colors='k', linewidths=wid, levels=conts1, linestyles='-')
#proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, '$\Delta$ zooplankton grazing = -697 Tg N yr$^{-1}$', transform=ax6.transAxes, fontsize=fstic, ha='center', va='center')

plt.subplots_adjust(wspace=0.05, left=0.04, top=0.94, bottom=0.08)


cbax1 = fig.add_axes([0.91, 0.69, 0.025, 0.25])
cbax2 = fig.add_axes([0.91, 0.385, 0.025, 0.25])
cbax3 = fig.add_axes([0.91, 0.08, 0.025, 0.25])

cbar1 = plt.colorbar(p1, cax=cbax1, orientation='vertical', ticks=levs[::4])
cbar2 = plt.colorbar(p3, cax=cbax2, orientation='vertical', ticks=levs[::4])
cbar3 = plt.colorbar(p5, cax=cbax3, orientation='vertical', ticks=levs1[::4])


cbar1.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar1.ax.tick_params(labelsize=fstic)
cbar2.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar2.ax.tick_params(labelsize=fstic)
cbar3.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar3.ax.tick_params(labelsize=fstic)


xx = 0.025; yy = 1.075
plt.text(xx,yy, 'a)', fontsize=fslab+2, va='center', ha='center', transform=ax1.transAxes)
plt.text(xx,yy, 'b)', fontsize=fslab+2, va='center', ha='center', transform=ax2.transAxes)
plt.text(xx,yy, 'c)', fontsize=fslab+2, va='center', ha='center', transform=ax3.transAxes)
plt.text(xx,yy, 'd)', fontsize=fslab+2, va='center', ha='center', transform=ax4.transAxes)
plt.text(xx,yy, 'e)', fontsize=fslab+2, va='center', ha='center', transform=ax5.transAxes)
plt.text(xx,yy, 'f)', fontsize=fslab+2, va='center', ha='center', transform=ax6.transAxes)


#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp5.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp5.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp5_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)



