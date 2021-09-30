# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

Use Ferret scripts "ncycle_change.jnl" and "sources_and_sinks.jnl" to create the netcdf files used to create figure


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

data = nc.Dataset('ncycle_changes.nc','r')
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


data = nc.Dataset('sources_and_sinks.nc','r')
budget_pi = data.variables['BUDGET_PI'][...]
budget_pindep = data.variables['BUDGET_PINDEP'][...]
budget_fut = data.variables['BUDGET_FUT'][...]
budget_futndep = data.variables['BUDGET_FUTNDEP'][...]
time = data.variables['TIME_COUNTER'][...]/86400/365+1900

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

cm = 1/2.54  # centimeters in inches
fig = plt.figure(figsize=(14,10), facecolor='w')
gs = GridSpec(20,10)

ax1 = plt.subplot(gs[0:5,0:5])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, delta_dep, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c1 = plt.contour(lonproj,latproj, delta_dep, colors='k', linewidths=wid, levels=conts, linestyles='-')

ax2 = plt.subplot(gs[0:5,5:10])
ax2.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, delta_fix, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c2 = plt.contour(lonproj,latproj, delta_fix, colors='k', linewidths=wid, levels=conts, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

ax3 = plt.subplot(gs[5:10,0:5])
ax3.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p3 = plt.contourf(lonproj,latproj, delta_npp, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c3 = plt.contour(lonproj,latproj, delta_npp, colors='k', linewidths=wid, levels=conts1, linestyles='-')

ax4 = plt.subplot(gs[5:10,5:10])
ax4.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p4 = plt.contourf(lonproj,latproj, delta_zoo, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c4 = plt.contour(lonproj,latproj, delta_zoo, colors='k', linewidths=wid, levels=conts1, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

ax5 = plt.subplot(gs[10:15,0:5])
ax5.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p5 = plt.contourf(lonproj,latproj, delta_wcd, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c5 = plt.contour(lonproj,latproj, delta_wcd, colors='k', linewidths=wid, levels=conts, linestyles='-')
proj.drawmeridians(np.arange(30, 331, 60), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)


ax6 = plt.subplot(gs[10:15,5:10])
ax6.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p6 = plt.contourf(lonproj,latproj, delta_sed, levels=levs, cmap=colmap, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c6 = plt.contour(lonproj,latproj, delta_sed, colors='k', linewidths=wid, levels=conts, linestyles='-')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(np.arange(30, 331, 60), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)


cols = ['k', 'royalblue', 'goldenrod', 'firebrick']
labs = ['preindustrial control', 'nitrogen deposition', 'climate change', 'climate change &\nnitrogen deposition']
wid=0.5

ax7 = plt.subplot(gs[16:20,0:8])
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.tick_params(labelsize=fstic)
#plt.plot(time, budget_pi, color=cols[0], linewidth=1.5, alpha=alf, label=labs[0])
plt.plot(time, budget_pindep - budget_pi, color=cols[0], linewidth=2.0, alpha=alf, label=labs[1])
plt.plot(time[49:300], budget_fut - budget_pi[49::], color=cols[2], linewidth=2.0, alpha=alf, label=labs[2])
plt.plot(time[49:300], budget_futndep - budget_pi[49::], color=cols[3], linewidth=2.0, alpha=alf, label=labs[3])
plt.plot((1790,2110), (0,0), color='k', linewidth=.5, alpha=0.5, linestyle='--', zorder=0)
plt.xlim(1795,2105); plt.ylim(-10,40)
plt.legend(loc='center right', fontsize=12, frameon=False, ncol=1, bbox_to_anchor=(1.275,0.5))
plt.xlabel('Year (Common Era)', fontsize=fslab)
plt.ylabel('$\Delta$ N budget\n(Tg N yr$^{-1}$)', fontsize=fslab)


plt.text(0.5,1.075, '$\Delta$ nitrogen deposition = 26.3 Tg N yr$^{-1}$ (+150%)', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(0.5,1.075, '$\Delta$ nitrogen fixation = -4.9 Tg N yr$^{-1}$ (-6%)', transform=ax2.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(0.5,1.075, '$\Delta$ net primary production = -297 Tg N yr$^{-1}$ (-5%)', transform=ax3.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(0.5,1.075, '$\Delta$ zooplankton grazing = -362 Tg N yr$^{-1}$ (-5%)', transform=ax4.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(0.5,1.075, '$\Delta$ water column denitrification = 1.2 Tg N yr$^{-1}$ (+9%)', transform=ax5.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(0.5,1.075, '$\Delta$ sedimentary denitrification = -7.7 Tg N yr$^{-1}$ (-7%)', transform=ax6.transAxes, fontsize=fstic, ha='center', va='center')

plt.subplots_adjust(wspace=0.3, left=0.12, top=0.97, bottom=0.1, right=0.95)


cbax1 = fig.add_axes([0.08, 0.775, 0.025, 0.18])
cbax2 = fig.add_axes([0.08, 0.555, 0.025, 0.18])
cbax3 = fig.add_axes([0.08, 0.335, 0.025, 0.18])

cbar1 = plt.colorbar(p1, cax=cbax1, orientation='vertical', ticks=levs[::5])
cbar2 = plt.colorbar(p3, cax=cbax2, orientation='vertical', ticks=levs1[::5])
cbar3 = plt.colorbar(p5, cax=cbax3, orientation='vertical', ticks=levs[::5])


cbar1.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar1.ax.tick_params(labelsize=fstic, right=False, labelright=False, left=True, labelleft=True)
cbar1.ax.yaxis.set_label_position('left')
cbar2.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar2.ax.tick_params(labelsize=fstic, right=False, labelright=False, left=True, labelleft=True)
cbar2.ax.yaxis.set_label_position('left')
cbar3.ax.set_ylabel('g N m$^{-2}$ yr$^{-1}$', fontsize=fslab)
cbar3.ax.tick_params(labelsize=fstic, right=False, labelright=False, left=True, labelleft=True)
cbar3.ax.yaxis.set_label_position('left')


xx = 0.025; yy = 1.075
plt.text(xx,yy, 'a)', fontsize=fslab+2, va='center', ha='center', transform=ax1.transAxes)
plt.text(xx,yy, 'b)', fontsize=fslab+2, va='center', ha='center', transform=ax2.transAxes)
plt.text(xx,yy, 'c)', fontsize=fslab+2, va='center', ha='center', transform=ax3.transAxes)
plt.text(xx,yy, 'd)', fontsize=fslab+2, va='center', ha='center', transform=ax4.transAxes)
plt.text(xx,yy, 'e)', fontsize=fslab+2, va='center', ha='center', transform=ax5.transAxes)
plt.text(xx,yy, 'f)', fontsize=fslab+2, va='center', ha='center', transform=ax6.transAxes)
plt.text(xx,yy, 'g)', fontsize=fslab+2, va='center', ha='center', transform=ax7.transAxes)


#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-main1.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-main1.eps', format='eps', dpi=300)
fig.savefig('fig-main1.pdf', dpi=300)
fig.savefig('fig-main1_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)



