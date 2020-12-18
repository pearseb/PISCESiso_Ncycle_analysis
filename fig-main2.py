# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

Use Ferret script "isotopic_trends_depthzones.jnl" to create trend data
Use Python script "process_compute_ToE.py" and Ferret script "calculate_toe_percentcover.jnl" to create ToE data 


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
diff_d15nno3_ez = data.variables['D15N_NO3_EZ_DEPHISTDIF'][...]
diff_d15nno3_utz = data.variables['D15N_NO3_UTZ_DEPHISTDIF'][...]
lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]

data = nc.Dataset('figure2D_ndep_d15npom_signal_usingPAR.nc','r')
diff_d15npom_ez = data.variables['D15N_POM_EZ_DEPHISTDIF'][...]
diff_d15npom_utz = data.variables['D15N_POM_UTZ_DEPHISTDIF'][...]


data = nc.Dataset('figure2D_cc_d15nno3_signal_usingPAR.nc','r')
diff_d15nno3_cc_ez = data.variables['D15N_NO3_EZ_FUTHISTDIF'][...]
diff_d15nno3_cc_utz = data.variables['D15N_NO3_UTZ_FUTHISTDIF'][...]

data = nc.Dataset('figure2D_cc_d15npom_signal_usingPAR.nc','r')
diff_d15npom_cc_ez = data.variables['D15N_POM_EZ_FUTHISTDIF'][...]
diff_d15npom_cc_utz = data.variables['D15N_POM_UTZ_FUTHISTDIF'][...]


data = nc.Dataset('figure2D_picdep_d15nno3_signal_usingPAR.nc','r')
diff_d15nno3_pic_ez = data.variables['D15N_NO3_EZ_PICDEPHISTDIF'][...]
diff_d15nno3_pic_utz = data.variables['D15N_NO3_UTZ_PICDEPHISTDIF'][...]

data = nc.Dataset('figure2D_picdep_d15npom_signal_usingPAR.nc','r')
diff_d15npom_pic_ez = data.variables['D15N_POM_EZ_PICDEPHISTDIF'][...]
diff_d15npom_pic_utz = data.variables['D15N_POM_UTZ_PICDEPHISTDIF'][...]


data = nc.Dataset('BGCP_ETOPO_merged_alt.nc','r')
provs = np.rint(data.variables['BGCP'][...])
land = np.ma.getmask(provs)
uniq_provs = np.unique(provs)[0:-1]
provs = np.ma.masked_where(provs==11, provs)

data.close()


#%% ToE data

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

data = nc.Dataset('ETOPO_ToE_futndep_depthzones.nc','r')
toe_futndep_fix = data.variables['fix'][...]
toe_futndep_npp = data.variables['npp'][...]
toe_futndep_no3_ez = data.variables['no3_ez'][...]
toe_futndep_no3_utz = data.variables['no3_utz'][...]
toe_futndep_nst_ez = data.variables['nst_ez'][...]
toe_futndep_nst_utz = data.variables['nst_utz'][...]
toe_futndep_d15nno3_ez = data.variables['d15n_no3_ez'][...]
toe_futndep_d15nno3_utz = data.variables['d15n_no3_utz'][...]
toe_futndep_d15npom_ez = data.variables['d15n_pom_ez'][...]
toe_futndep_d15npom_utz = data.variables['d15n_pom_utz'][...]

data = nc.Dataset('ETOPO_ToE_fut_depthzones.nc','r')
toe_fut_fix = data.variables['fix'][...]
toe_fut_npp = data.variables['npp'][...]
toe_fut_temp_ez = data.variables['temp_ez'][...]
toe_fut_temp_utz = data.variables['temp_utz'][...]
toe_fut_no3_ez = data.variables['no3_ez'][...]
toe_fut_no3_utz = data.variables['no3_utz'][...]
toe_fut_nst_ez = data.variables['nst_ez'][...]
toe_fut_nst_utz = data.variables['nst_utz'][...]
toe_fut_d15nno3_ez = data.variables['d15n_no3_ez'][...]
toe_fut_d15nno3_utz = data.variables['d15n_no3_utz'][...]
toe_fut_d15npom_ez = data.variables['d15n_pom_ez'][...]
toe_fut_d15npom_utz = data.variables['d15n_pom_utz'][...]

data = nc.Dataset('ETOPO_ToE_picndep_depthzones.nc','r')
toe_picndep_fix = data.variables['fix'][...]
toe_picndep_npp = data.variables['npp'][...]
toe_picndep_no3_ez = data.variables['no3_ez'][...]
toe_picndep_no3_utz = data.variables['no3_utz'][...]
toe_picndep_nst_ez = data.variables['nst_ez'][...]
toe_picndep_nst_utz = data.variables['nst_utz'][...]
toe_picndep_d15nno3_ez = data.variables['d15n_no3_ez'][...]
toe_picndep_d15nno3_utz = data.variables['d15n_no3_utz'][...]
toe_picndep_d15npom_ez = data.variables['d15n_pom_ez'][...]
toe_picndep_d15npom_utz = data.variables['d15n_pom_utz'][...]


data = nc.Dataset('BGCP_ETOPO_merged_alt.nc','r')
provs = np.rint(data.variables['BGCP'][...])
land = np.ma.getmask(provs)
uniq_provs = np.unique(provs)[0:-1]
provs = np.ma.masked_where(provs==11, provs)

data.close()


# set years > 2100 (which are masked) to 2100

toe_picndep_fix = np.ma.getdata(toe_picndep_fix)
toe_picndep_npp = np.ma.getdata(toe_picndep_npp)

toe_picndep_no3_ez = np.ma.getdata(toe_picndep_no3_ez)
toe_picndep_no3_utz = np.ma.getdata(toe_picndep_no3_utz)

toe_picndep_nst_ez = np.ma.getdata(toe_picndep_nst_ez)
toe_picndep_nst_utz = np.ma.getdata(toe_picndep_nst_utz)

toe_picndep_d15nno3_ez = np.ma.getdata(toe_picndep_d15nno3_ez)
toe_picndep_d15nno3_utz = np.ma.getdata(toe_picndep_d15nno3_utz)

toe_picndep_d15npom_ez = np.ma.getdata(toe_picndep_d15npom_ez)
toe_picndep_d15npom_utz = np.ma.getdata(toe_picndep_d15npom_utz)


toe_futndep_fix = np.ma.getdata(toe_futndep_fix)
toe_futndep_npp = np.ma.getdata(toe_futndep_npp)

toe_futndep_no3_ez = np.ma.getdata(toe_futndep_no3_ez)
toe_futndep_no3_utz = np.ma.getdata(toe_futndep_no3_utz)

toe_futndep_nst_ez = np.ma.getdata(toe_futndep_nst_ez)
toe_futndep_nst_utz = np.ma.getdata(toe_futndep_nst_utz)

toe_futndep_d15nno3_ez = np.ma.getdata(toe_futndep_d15nno3_ez)
toe_futndep_d15nno3_utz = np.ma.getdata(toe_futndep_d15nno3_utz)

toe_futndep_d15npom_ez = np.ma.getdata(toe_futndep_d15npom_ez)
toe_futndep_d15npom_utz = np.ma.getdata(toe_futndep_d15npom_utz)


toe_fut_fix = np.ma.getdata(toe_fut_fix)
toe_fut_npp = np.ma.getdata(toe_fut_npp)

toe_fut_temp_ez = np.ma.getdata(toe_fut_temp_ez)
toe_fut_temp_utz = np.ma.getdata(toe_fut_temp_utz)

toe_fut_no3_ez = np.ma.getdata(toe_fut_no3_ez)
toe_fut_no3_utz = np.ma.getdata(toe_fut_no3_utz)

toe_fut_nst_ez = np.ma.getdata(toe_fut_nst_ez)
toe_fut_nst_utz = np.ma.getdata(toe_fut_nst_utz)

toe_fut_d15nno3_ez = np.ma.getdata(toe_fut_d15nno3_ez)
toe_fut_d15nno3_utz = np.ma.getdata(toe_fut_d15nno3_utz)

toe_fut_d15npom_ez = np.ma.getdata(toe_fut_d15npom_ez)
toe_fut_d15npom_utz = np.ma.getdata(toe_fut_d15npom_utz)


toe_fut_fix[toe_fut_fix > 1e20] = 2100
toe_futndep_fix[toe_futndep_fix > 1e20] = 2100
toe_picndep_fix[toe_picndep_fix > 1e20] = 2100
toe_fut_npp[toe_fut_npp > 1e20] = 2100
toe_futndep_npp[toe_futndep_npp > 1e20] = 2100
toe_picndep_npp[toe_picndep_npp > 1e20] = 2100

toe_picndep_no3_ez[toe_picndep_no3_ez > 1e20] = 2100
toe_picndep_no3_utz[toe_picndep_no3_utz > 1e20] = 2100

toe_picndep_nst_ez[toe_picndep_nst_ez > 1e20] = 2100
toe_picndep_nst_utz[toe_picndep_nst_utz > 1e20] = 2100

toe_picndep_d15nno3_ez[toe_picndep_d15nno3_ez > 1e20] = 2100
toe_picndep_d15nno3_utz[toe_picndep_d15nno3_utz > 1e20] = 2100

toe_picndep_d15npom_ez[toe_picndep_d15npom_ez > 1e20] = 2100
toe_picndep_d15npom_utz[toe_picndep_d15npom_utz > 1e20] = 2100

toe_futndep_no3_ez[toe_futndep_no3_ez > 1e20] = 2100
toe_futndep_no3_utz[toe_futndep_no3_utz > 1e20] = 2100

toe_futndep_nst_ez[toe_futndep_nst_ez > 1e20] = 2100
toe_futndep_nst_utz[toe_futndep_nst_utz > 1e20] = 2100

toe_futndep_d15nno3_ez[toe_futndep_d15nno3_ez > 1e20] = 2100
toe_futndep_d15nno3_utz[toe_futndep_d15nno3_utz > 1e20] = 2100

toe_futndep_d15npom_ez[toe_futndep_d15npom_ez > 1e20] = 2100
toe_futndep_d15npom_utz[toe_futndep_d15npom_utz > 1e20] = 2100

toe_fut_temp_ez[toe_fut_temp_ez > 1e20] = 2100
toe_fut_temp_utz[toe_fut_temp_utz > 1e20] = 2100

toe_fut_no3_ez[toe_fut_no3_ez > 1e20] = 2100
toe_fut_no3_utz[toe_fut_no3_utz > 1e20] = 2100

toe_fut_nst_ez[toe_fut_nst_ez > 1e20] = 2100
toe_fut_nst_utz[toe_fut_nst_utz > 1e20] = 2100

toe_fut_d15nno3_ez[toe_fut_d15nno3_ez > 1e20] = 2100
toe_fut_d15nno3_utz[toe_fut_d15nno3_utz > 1e20] = 2100

toe_fut_d15npom_ez[toe_fut_d15npom_ez > 1e20] = 2100
toe_fut_d15npom_utz[toe_fut_d15npom_utz > 1e20] = 2100


toe_fut_fix = np.ma.masked_where(land, toe_fut_fix)
toe_futndep_fix = np.ma.masked_where(land, toe_futndep_fix)
toe_picndep_fix = np.ma.masked_where(land, toe_picndep_fix)
toe_fut_npp = np.ma.masked_where(land, toe_fut_npp)
toe_futndep_npp = np.ma.masked_where(land, toe_futndep_npp)
toe_picndep_npp = np.ma.masked_where(land, toe_picndep_npp)

toe_picndep_no3_ez = np.ma.masked_where(land, toe_picndep_no3_ez)
toe_picndep_no3_utz = np.ma.masked_where(land, toe_picndep_no3_utz)

toe_picndep_nst_ez = np.ma.masked_where(land, toe_picndep_nst_ez)
toe_picndep_nst_utz = np.ma.masked_where(land, toe_picndep_nst_utz)

toe_picndep_d15nno3_ez = np.ma.masked_where(land, toe_picndep_d15nno3_ez)
toe_picndep_d15nno3_utz = np.ma.masked_where(land, toe_picndep_d15nno3_utz)

toe_picndep_d15npom_ez = np.ma.masked_where(land, toe_picndep_d15npom_ez)
toe_picndep_d15npom_utz = np.ma.masked_where(land, toe_picndep_d15npom_utz)

toe_futndep_no3_ez = np.ma.masked_where(land, toe_futndep_no3_ez)
toe_futndep_no3_utz = np.ma.masked_where(land, toe_futndep_no3_utz)

toe_futndep_nst_ez = np.ma.masked_where(land, toe_futndep_nst_ez)
toe_futndep_nst_utz = np.ma.masked_where(land, toe_futndep_nst_utz)

toe_futndep_d15nno3_ez = np.ma.masked_where(land, toe_futndep_d15nno3_ez)
toe_futndep_d15nno3_utz = np.ma.masked_where(land, toe_futndep_d15nno3_utz)

toe_futndep_d15npom_ez = np.ma.masked_where(land, toe_futndep_d15npom_ez)
toe_futndep_d15npom_utz = np.ma.masked_where(land, toe_futndep_d15npom_utz)

toe_fut_temp_ez = np.ma.masked_where(land, toe_fut_temp_ez)
toe_fut_temp_utz = np.ma.masked_where(land, toe_fut_temp_utz)

toe_fut_no3_ez = np.ma.masked_where(land, toe_fut_no3_ez)
toe_fut_no3_utz = np.ma.masked_where(land, toe_fut_no3_utz)

toe_fut_nst_ez = np.ma.masked_where(land, toe_fut_nst_ez)
toe_fut_nst_utz = np.ma.masked_where(land, toe_fut_nst_utz)

toe_fut_d15nno3_ez = np.ma.masked_where(land, toe_fut_d15nno3_ez)
toe_fut_d15nno3_utz = np.ma.masked_where(land, toe_fut_d15nno3_utz)

toe_fut_d15npom_ez = np.ma.masked_where(land, toe_fut_d15npom_ez)
toe_fut_d15npom_utz = np.ma.masked_where(land, toe_fut_d15npom_utz)



#%%  get ToE curve data instead of separating things into arbitrary provinces

import pandas as pd

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

names = ['Year', r'N$_2$ fix', 'NPP', r'NO$_3^{EZ}$', r'NO$_3^{TZ}$', r'N*$^{EZ}$', r'N*$^{TZ}$', r'$\delta^{15}$N$_{NO_3}^{EZ}$', r'$\delta^{15}$N$_{NO_3}^{TZ}$', r'$\delta^{15}$N$_{POM}^{EZ}$', r'$\delta^{15}$N$_{POM}^{TZ}$']

curve_fut = pd.read_csv('ToE_fut_curves.txt', names=names, delimiter="\t")
curve_futdep = pd.read_csv('ToE_futndep_curves.txt', names=names, delimiter="\t")
curve_picdep = pd.read_csv('ToE_picndep_curves.txt', names=names, delimiter="\t")

curve_fut[curve_fut < 0] = 0
curve_futdep[curve_futdep < 0] = 0
curve_picdep[curve_picdep < 0] = 0

curve_fut = curve_fut*100
curve_futdep = curve_futdep*100
curve_picdep = curve_picdep*100


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

fstic = 13
fslab = 15
alf = 0.7
wid=0.7

colmap = cmocean.tools.lighten(cmo.balance,0.75)
colmap_toe = cmocean.tools.lighten(cmo.tempo_r,0.75)

fig = plt.figure(figsize=(18,7), facecolor='w')
gs = GridSpec(24,3)


levs1 = np.concatenate((np.array([-2,-1.8,-1.6,-1.4]), np.arange(-1.2,1.21,0.1), np.array([1.4,1.6,1.8, 2])))
levs1 = np.arange(-1,1.01,0.05)

ax1 = plt.subplot(gs[0:8,0])
ax1.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p1 = plt.contourf(lonproj,latproj, diff_d15nno3_utz, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c1 = plt.contour(lonproj,latproj, diff_d15nno3_utz, colors='k', linewidths=wid, levels=np.array([-0.2,0.2]))
#cc = plt.contour(lonproj,latproj, provs, colors='k', linewidths=wid, levels=uniq_provs, linestyles='--')
#plt.text(0.5,1.075, '$\Delta$ $\delta^{15}$N$_{NO_3}$ in Upper Twilight Zone (UTZ)', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

ax4 = plt.subplot(gs[8:16,0])
ax4.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p4 = plt.contourf(lonproj,latproj, diff_d15nno3_pic_utz, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c4 = plt.contour(lonproj,latproj, diff_d15nno3_pic_utz, colors='k', linewidths=wid, levels=np.array([-0.2,0.2]))
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)

ax7 = plt.subplot(gs[16:24,0])
ax7.tick_params(labelsize=fstic)
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p7 = plt.contourf(lonproj,latproj, diff_d15nno3_cc_utz, levels=levs1, cmap=colmap, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
c7 = plt.contour(lonproj,latproj, diff_d15nno3_cc_utz, colors='k', linewidths=wid, levels=np.array([-0.2,0.2]))
proj.drawmeridians(np.arange(30, 331, 60), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)
proj.drawparallels(range(domain_draw[0], domain_draw[2]+1, dlat), labels=[False,True,False,False], color=(.3,.3,.3), linewidth=0, fontsize=fstic)




levs1 = np.concatenate((np.array([1930]), np.arange(1940,2091,10), np.array([2100])))
hatc = ['', '', '', '', '', '', '', '', '', '....', '....', '....', '....', '....', '', '', '']
levs2 = np.concatenate((np.array([-100,-80,-60]), np.arange(-50,51,5), np.array([60,80,100])))
conts1 = np.array([2010])
conts2 = np.array([0])


ax2 = plt.subplot(gs[0:8,1])
ax2.tick_params(labelsize=fstic)
# annotate
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p2 = plt.contourf(lonproj,latproj, toe_futndep_d15nno3_utz, hatches=hatc, levels=levs1, cmap=colmap_toe, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
#plt.contourf(lonproj,latproj, toe_futndep_d15nno3_utz, colors='None', hatches=['....'], levels=np.array([2020,2060]))

ax5 = plt.subplot(gs[8:16,1])
ax5.tick_params(labelsize=fstic)
# annotate
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p5 = plt.contourf(lonproj,latproj, toe_picndep_d15nno3_utz, hatches=hatc, levels=levs1, cmap=colmap_toe, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
#plt.contourf(lonproj,latproj, toe_picndep_d15nno3_utz, colors='None', hatches=['....'], levels=np.array([2020,2060]))

ax8 = plt.subplot(gs[16:24,1])
ax8.tick_params(labelsize=fstic)
# annotate
proj.drawcoastlines(linewidth=0.5, color='k')
proj.fillcontinents(color='grey')
p8 = plt.contourf(lonproj,latproj, toe_fut_d15nno3_utz, hatches=hatc, levels=levs1, cmap=colmap_toe, vmin=np.min(levs1), vmax=np.max(levs1), extend='both')
#plt.contourf(lonproj,latproj, toe_fut_d15nno3_utz, colors='None', hatches=['....'], levels=np.array([2020,2060]))
proj.drawmeridians(np.arange(30, 331, 60), labels=[True,False,False,True], color=(.3,.3,.3), linewidth=0, fontsize=fstic)



alf1 = 0.75
wid = 2.5; sty = ['-', '--']
years = np.arange(1850,2101,1)


ax3 = plt.subplot(gs[1:7,2])
ax3.tick_params(labelsize=fstic, left=False, labelleft=False, right=True, labelright=True, labelbottom=False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.plot(years, curve_futdep["NPP"], color='k', linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[2])
plt.plot(years, curve_futdep["N$_2$ fix"], color='grey', linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[1])
plt.plot(years, curve_futdep["NO$_3^{EZ}$"], color=tableau20[2], linestyle=sty[1], alpha=alf1, linewidth=wid, label=names[3])
plt.plot(years, curve_futdep["NO$_3^{TZ}$"], color=tableau20[2], linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[4])
plt.plot(years, curve_futdep["N*$^{EZ}$"], color=tableau20[0], linestyle=sty[1], alpha=alf1, linewidth=wid, label=names[5])
plt.plot(years, curve_futdep["N*$^{TZ}$"], color=tableau20[0], linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[6])
plt.plot(years, curve_futdep["$\delta^{15}$N$_{POM}^{EZ}$"], color=tableau20[8], linestyle=sty[1], alpha=alf1, linewidth=wid, label=names[9])
plt.plot(years, curve_futdep["$\delta^{15}$N$_{POM}^{TZ}$"], color=tableau20[8], linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[10])
plt.plot(years, curve_futdep["$\delta^{15}$N$_{NO_3}^{EZ}$"], color=tableau20[6], linestyle=sty[1], alpha=alf1, linewidth=wid, label=names[7])
plt.plot(years, curve_futdep["$\delta^{15}$N$_{NO_3}^{TZ}$"], color=tableau20[6], linestyle=sty[0], alpha=alf1, linewidth=wid, label=names[8])

plt.axvspan(1986,2005,facecolor='grey', alpha=0.2)
plt.axvspan(2081,2100,facecolor='grey', alpha=0.2)

plt.plot((1850,2100),(50,50),'k', alpha=1, linestyle=':', linewidth=1)

plt.ylim(-1,100); plt.xlim(1900,2100)
plt.ylabel('% Ocean', fontsize=fslab)
ax3.yaxis.set_label_position('right')

plt.legend(frameon=False, loc='upper center', ncol=5, fontsize=10, bbox_to_anchor=(0.5,1.6), columnspacing=0.5)


ax6 = plt.subplot(gs[9:15,2])
ax6.tick_params(labelsize=fstic, left=False, labelleft=False, right=True, labelright=True, labelbottom=False)
ax6.spines['top'].set_visible(False)
ax6.spines['left'].set_visible(False)

plt.plot(years, curve_picdep["NPP"], color='k', linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["N$_2$ fix"], color='grey', linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["NO$_3^{EZ}$"], color=tableau20[2], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["NO$_3^{TZ}$"], color=tableau20[2], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["N*$^{EZ}$"], color=tableau20[0], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["N*$^{TZ}$"], color=tableau20[0], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["$\delta^{15}$N$_{POM}^{EZ}$"], color=tableau20[8], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["$\delta^{15}$N$_{POM}^{TZ}$"], color=tableau20[8], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["$\delta^{15}$N$_{NO_3}^{EZ}$"], color=tableau20[6], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_picdep["$\delta^{15}$N$_{NO_3}^{TZ}$"], color=tableau20[6], linestyle=sty[0], alpha=alf1, linewidth=wid)

plt.axvspan(1986,2005,facecolor='grey', alpha=0.2)
plt.axvspan(2081,2100,facecolor='grey', alpha=0.2)

plt.plot((1850,2100),(50,50),'k', alpha=1, linestyle=':', linewidth=1)

plt.ylim(-1,100); plt.xlim(1900,2100)
plt.ylabel('% Ocean', fontsize=fslab)
ax6.yaxis.set_label_position('right')


ax9 = plt.subplot(gs[17:23,2])
ax9.tick_params(labelsize=fstic, left=False, labelleft=False, right=True, labelright=True)
ax9.spines['top'].set_visible(False)
ax9.spines['left'].set_visible(False)

plt.plot(years, curve_fut["NPP"], color='k', linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["N$_2$ fix"], color='grey', linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["NO$_3^{EZ}$"], color=tableau20[2], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["NO$_3^{TZ}$"], color=tableau20[2], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["N*$^{EZ}$"], color=tableau20[0], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["N*$^{TZ}$"], color=tableau20[0], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["$\delta^{15}$N$_{POM}^{EZ}$"], color=tableau20[8], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["$\delta^{15}$N$_{POM}^{TZ}$"], color=tableau20[8], linestyle=sty[0], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["$\delta^{15}$N$_{NO_3}^{EZ}$"], color=tableau20[6], linestyle=sty[1], alpha=alf1, linewidth=wid)
plt.plot(years, curve_fut["$\delta^{15}$N$_{NO_3}^{TZ}$"], color=tableau20[6], linestyle=sty[0], alpha=alf1, linewidth=wid)

plt.axvspan(1986,2005,facecolor='grey', alpha=0.2)
plt.axvspan(2081,2100,facecolor='grey', alpha=0.2)

plt.plot((1850,2100),(50,50),'k', alpha=1, linestyle=':', linewidth=1)

plt.ylim(-1,100); plt.xlim(1900,2100)
plt.ylabel('% Ocean', fontsize=fslab)
ax9.yaxis.set_label_position('right')
plt.xlabel('Year', fontsize=fslab)


plt.subplots_adjust(wspace=0.11, left=0.08, bottom=0.075, top=0.875, right=0.95)


levs1 = np.arange(-1,1.01,0.05)

cbax1 = fig.add_axes([0.1, 0.91, 0.23, 0.03])
cbar1 = plt.colorbar(p1, cax=cbax1, orientation='horizontal', ticks=levs1[::4])
cbar1.ax.set_xlabel('21$^{st}$ century $\Delta$ $\delta^{15}$N$^{TZ}_{NO3}$ (\u2030)', fontsize=fslab, labelpad=10)
cbar1.ax.tick_params(labelsize=fstic)
cbar1.ax.xaxis.set_label_position('top')

levs1 = np.concatenate((np.array([1930]), np.arange(1940,2091,10), np.array([2100])))

cbax2 = fig.add_axes([0.4, 0.91, 0.23, 0.03])
cbar2 = plt.colorbar(p2, cax=cbax2, orientation='horizontal', ticks=levs1[::3])
cbar2.ax.set_xlabel('Time of Emergence', fontsize=fslab, labelpad=10)
cbar2.ax.tick_params(labelsize=fstic)
cbar2.ax.xaxis.set_label_position('top')



col='whitesmoke'
xx=0.2;yy=0.85
plt.text(xx,yy, 'a)', fontsize=fslab+2, ha='center', va='center', transform=ax1.transAxes, color=col)
plt.text(xx,yy, 'b)', fontsize=fslab+2, ha='center', va='center', transform=ax2.transAxes, color=col)
plt.text(xx,yy, 'd)', fontsize=fslab+2, ha='center', va='center', transform=ax4.transAxes, color=col)
plt.text(xx,yy, 'e)', fontsize=fslab+2, ha='center', va='center', transform=ax5.transAxes, color=col)
plt.text(xx,yy, 'g)', fontsize=fslab+2, ha='center', va='center', transform=ax7.transAxes, color=col)
plt.text(xx,yy, 'h)', fontsize=fslab+2, ha='center', va='center', transform=ax8.transAxes, color=col)

xx=0.05;yy=0.95
plt.text(xx,yy, 'c)', fontsize=fslab+2, ha='center', va='center', transform=ax3.transAxes)
plt.text(xx,yy, 'f)', fontsize=fslab+2, ha='center', va='center', transform=ax6.transAxes)
plt.text(xx,yy, 'i)', fontsize=fslab+2, ha='center', va='center', transform=ax9.transAxes)


plt.text(-0.15,0.5, 'climate change\n&\nnitrogen\ndeposition', transform=ax1.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(-0.15,0.5, 'nitrogen\ndeposition', transform=ax4.transAxes, fontsize=fstic, ha='center', va='center')
plt.text(-0.15,0.5, 'climate change', transform=ax7.transAxes, fontsize=fstic, ha='center', va='center')


#%% savefig

plt.clabel(c1, manual=True, fontsize=11, colors='k', inline=6, fmt='%.1f')
plt.clabel(c4, manual=True, fontsize=11, colors='k', inline=6, fmt='%.1f')
plt.clabel(c7, manual=True, fontsize=11, colors='k', inline=6, fmt='%.1f')


os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-main2.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-main2.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-main2_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)

