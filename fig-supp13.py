# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

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


#%% trend data

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

# NO3

data = nc.Dataset('figure2D_ndep_d15nno3_signal_usingPAR.nc','r')
d15nno3_utz = data.variables['D15N_NO3_UTZ_DEPDIF'][...]
lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]


data = nc.Dataset('figure2D_cc_d15nno3_signal_usingPAR.nc','r')
d15nno3_cc_utz = data.variables['D15N_NO3_UTZ_FUTDIF'][...]


data = nc.Dataset('ETOPO_futdiffs_d15Nno3_UTZ_corrected_for_isopycnal_heave.nc','r')
diff_d15nno3_utz_isos = data.variables['D15N_DEPDIF_ISOPYCNALS_UTZ'][...]
diff_d15nno3_cc_utz_isos = data.variables['D15N_FUTDIF_ISOPYCNALS_UTZ'][...]


data = nc.Dataset('BGCP_ETOPO_merged_alt.nc','r')
provs = np.rint(data.variables['BGCP'][...])
land = np.ma.getmask(provs)
uniq_provs = np.unique(provs)[0:-1]
provs = np.ma.masked_where(provs==11, provs)

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


#%% figure of d15N-no3 signals and significance at 95% level (EZ and UTZ)

colmap = cmocean.tools.lighten(cmo.balance,0.75)
levs1 = np.concatenate((np.array([-2,-1.8,-1.6,-1.4]), np.arange(-1.2,1.21,0.1), np.array([1.4,1.6,1.8, 2])))
levs1 = np.arange(-1.0,1.01,0.1)

fstic = 13
fslab = 15
alf = 0.7
wid=0.5


fig = plt.figure(figsize=(10,6), facecolor='w')

gs = GridSpec(2,1)

ax1 = plt.subplot(gs[0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, d15nno3_cc_utz, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c1 = plt.contour(lonproj,latproj, d15nno3_cc_utz, colors='k', linewidths=wid, levels=levs1[::2])
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, 'Twilight zone $\Delta$ $\delta^{15}$N$_{NO_3}$ due to climate change', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')

ax2 = plt.subplot(gs[1])
ax2.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, diff_d15nno3_cc_utz_isos, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c2 = plt.contour(lonproj,latproj, diff_d15nno3_cc_utz_isos, colors='k', linewidths=wid, levels=levs1[::2])
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[True,False,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawmeridians(range(domain_draw[1], domain_draw[3]+1, dlon), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
plt.text(0.5,1.075, 'Twilight zone $\Delta$ $\delta^{15}$N$_{NO_3}$ due to climate change on common isopycnal surfaces', transform=ax2.transAxes, fontsize=fstic, ha='center', va='center')

plt.subplots_adjust(left=0.08, top=0.92, bottom=0.08)

cbax1 = fig.add_axes([0.86, 0.15, 0.03, 0.7])
cbar1 = plt.colorbar(p1, cax=cbax1, orientation='vertical', ticks=levs1[::2])
cbar1.ax.set_ylabel('$\Delta$ $\delta^{15}$N$_{NO3}$ (\u2030)', fontsize=fslab)

xx = 0.025; yy = 1.05
plt.text(xx,yy, 'a)', transform=ax1.transAxes, fontsize=fslab+2, ha='center', va='center')
plt.text(xx,yy, 'b)', transform=ax2.transAxes, fontsize=fslab+2, ha='center', va='center')



#%% savefig

plt.clabel(c1, manual=True, fmt='%.1f', fontsize=10, colors='k', inline=6)
plt.clabel(c2, manual=True, fmt='%.1f', fontsize=10, colors='k', inline=6)


os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp12.png', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp12.eps', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp12_trans.png', dpi=300, bbox_inches='tight', transparent=True)



