# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

Use Ferret script "isotopic_trends_depthzones.jnl", "decomposition_no3.jnl" and "figure2D_cc_din_e15n.jnl" to make figure


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

data = nc.Dataset('figure2D_cc_d15npom_signal_usingPAR.nc','r')
diff_d15npom_cc_ez = data.variables['D15N_POM_EZ_FUTHISTDIF'][...]
diff_d15npom_cc_utz = data.variables['D15N_POM_UTZ_FUTHISTDIF'][...]

lon = data.variables['ETOPO60X'][...]
lon -= 360.0
lat = data.variables['ETOPO60Y'][...]

lons,lats = np.meshgrid(lon,lat)


data = nc.Dataset('ETOPO_fluxanalysis_results.nc')
enrich_pic = np.ma.squeeze(data.variables['ENRICH_PIC'][...])
enrich_fut = np.ma.squeeze(data.variables['ENRICH_FUT'][...])
enrich = enrich_fut - enrich_pic


data = nc.Dataset('figure2D_cc_din_e15n.nc','r')
delta_e15nr = data.variables['E15NR_FUT_DIF'][...]
delta_e15nn = data.variables['E15NN_FUT_DIF'][...]
delta_e15nt = data.variables['E15NT_FUT_DIF'][...]
delta_din = data.variables['DIN_FUT_DIF'][...]

'''
data = nc.Dataset('ETOPO_futdiffs_d15Nno3_UTZ_corrected_for_isopycnal_heave.nc','r')
diff_d15nno3_utz_isos = data.variables['D15N_DEPDIF_ISOPYCNALS_UTZ'][...]
diff_d15nno3_cc_utz_isos = data.variables['D15N_FUTDIF_ISOPYCNALS_UTZ'][...]
'''

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

    
#%% plot

colmap = cmocean.tools.lighten(cmo.balance,0.75)
levs1 = np.arange(-1.0,1.01,0.1)
levs2 = np.arange(-4,4.01,0.2)

fstic = 13
fslab = 15
alf = 0.7
wid = 0.5

fig = plt.figure(figsize=(8,8), facecolor='w')
gs = GridSpec(3,1)

ax1 = plt.subplot(gs[0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, diff_d15npom_cc_ez, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c1 = plt.contour(lonproj,latproj, diff_d15npom_cc_ez, colors='k', linewidths=wid, levels=levs1[::4])
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, 'Euphotic zone $\Delta$ $\delta^{15}$N$_{POM}$', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')

ax2 = plt.subplot(gs[1])
ax2.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, enrich, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c2 = plt.contour(lonproj,latproj, enrich, colors='k', linewidths=wid, levels=levs1[::4])
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, 'Twilight zone $\Delta$ $\delta^{15}$N$_{NO_3}$ due to biogeochemical fluxes', transform=ax2.transAxes, fontsize=fstic, ha='center', va='center')

ax3 = plt.subplot(gs[2])
ax3.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p3 = plt.contourf(lonproj,latproj, delta_din, levels=levs2, cmap=colmap, vmin=np.min(levs2), vmax=np.max(levs2), extend='both')
#c3 = plt.contour(lonproj,latproj, delta_e15nt, colors='k', linewidths=wid+0.5, levels=np.arange(-1,1.1,0.25))
#h3 = plt.contourf(lonproj,latproj, delta_e15nt, colors='none', levels=[-10,0], hatches=['...'])
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(np.arange(30, 331, 60), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, 'Euphotic zone $\Delta$ bioavailable nitrogen', transform=ax3.transAxes, fontsize=fstic, ha='center', va='center')


plt.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.85)


cbax1 = fig.add_axes([0.86, 0.7, 0.03, 0.25])
cbax2 = fig.add_axes([0.86, 0.4, 0.03, 0.25])
cbax3 = fig.add_axes([0.86, 0.1, 0.03, 0.25])

cbar1 = plt.colorbar(p1, cax=cbax1, orientation='vertical', ticks=levs1[::2])
cbar1.ax.set_ylabel('\u2030', fontsize=fslab)
cbax1.tick_params(labelsize=fstic)

cbar2 = plt.colorbar(p2, cax=cbax2, orientation='vertical', ticks=levs1[::2])
cbar2.ax.set_ylabel('\u2030', fontsize=fslab)
cbax2.tick_params(labelsize=fstic)

cbar3 = plt.colorbar(p3, cax=cbax3, orientation='vertical', ticks=levs2[::4])
cbar3.ax.set_ylabel('mmol m$^{-3}$', fontsize=fslab)
cbax3.tick_params(labelsize=fstic)


xx = 0.025; yy = 1.075
plt.text(xx,yy, 'a)', transform=ax1.transAxes, fontsize=fslab+2, ha='center', va='center')
plt.text(xx,yy, 'b)', transform=ax2.transAxes, fontsize=fslab+2, ha='center', va='center')
plt.text(xx,yy, 'c)', transform=ax3.transAxes, fontsize=fslab+2, ha='center', va='center')



#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-main3.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-main3.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-main3_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)


