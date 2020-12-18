# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:45:58 2020

Use Ferret script "assess_direct_indirect_effects.jnl" to make figure

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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb
sb.set(style='ticks')
import cmocean
import cmocean.cm as cmo

import mpl_toolkits.basemap as bm



#%% get data for bar graphs

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

data = nc.Dataset('ETOPO_direct_indirect_effects.nc','r')
delta_no3 = np.ma.squeeze(data.variables['NO3_FUT_DIF'][...])
delta_d15nno3 = np.ma.squeeze(data.variables['D15NNO3_FUT_DIF'][...])
delta_d15npom = np.ma.squeeze(data.variables['D15NPOM_FUT_DIF'][...])
delta_fix = np.ma.squeeze(data.variables['FIX_FUT_DIF'][...])
delta_wcd = np.ma.squeeze(data.variables['DEN_FUT_DIF'][...])
delta_sed = np.ma.squeeze(data.variables['SDEN_FUT_DIF'][...])
delta_npp = np.ma.squeeze(data.variables['NPP_FUT_DIF'][...])
delta_zoo = np.ma.squeeze(data.variables['GRA_FUT_DIF'][...])
#delta_e15n = np.ma.squeeze(data.variables['DELTA_E15N'][...])

lon = data.variables['ETOPO60X'][...]
lon -= 360
lat = data.variables['ETOPO60Y'][...]

no3_dir = np.ma.squeeze(data.variables['NO3_DIR_DIF'][...])
d15nno3_dir = np.ma.squeeze(data.variables['D15NNO3_DIR_DIF'][...])
d15npom_dir = np.ma.squeeze(data.variables['D15NPOM_DIR_DIF'][...])
fix_dir = np.ma.squeeze(data.variables['FIX_DIR_DIF'][...])
wcd_dir = np.ma.squeeze(data.variables['DEN_DIR_DIF'][...])
sed_dir = np.ma.squeeze(data.variables['SDEN_DIR_DIF'][...])
npp_dir = np.ma.squeeze(data.variables['NPP_DIR_DIF'][...])
zoo_dir = np.ma.squeeze(data.variables['GRA_DIR_DIF'][...])
#e15n_dir = np.ma.squeeze(data.variables['E15N_DIR_DIF'][...])

no3_ind = np.ma.squeeze(data.variables['NO3_IND_DIF'][...])
d15nno3_ind = np.ma.squeeze(data.variables['D15NNO3_IND_DIF'][...])
d15npom_ind = np.ma.squeeze(data.variables['D15NPOM_IND_DIF'][...])
fix_ind = np.ma.squeeze(data.variables['FIX_IND_DIF'][...])
wcd_ind = np.ma.squeeze(data.variables['DEN_IND_DIF'][...])
sed_ind = np.ma.squeeze(data.variables['SDEN_IND_DIF'][...])
npp_ind = np.ma.squeeze(data.variables['NPP_IND_DIF'][...])
zoo_ind = np.ma.squeeze(data.variables['GRA_IND_DIF'][...])
#e15n_ind = np.ma.squeeze(data.variables['GRA_IND_DIF'][...])


data.close()


#%% calculate correlation coefficients

mask = np.ma.getmask(d15nno3_dir)
delta_d15nno3 = np.ma.masked_where(mask, delta_d15nno3)
mask = np.ma.getmask(delta_d15nno3)
d15nno3_dir = np.ma.masked_where(mask, d15nno3_dir)
d15nno3_ind = np.ma.masked_where(mask, d15nno3_ind)

mask = np.ma.getmask(d15npom_dir)
delta_d15npom = np.ma.masked_where(mask, delta_d15npom)
mask = np.ma.getmask(delta_d15npom)
d15npom_dir = np.ma.masked_where(mask, d15npom_dir)
d15npom_ind = np.ma.masked_where(mask, d15npom_ind)


r_dir = np.array( [ np.corrcoef(np.ma.compressed(delta_no3), np.ma.compressed(no3_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_d15nno3), np.ma.compressed(d15nno3_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_d15npom), np.ma.compressed(d15npom_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_npp), np.ma.compressed(npp_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_zoo), np.ma.compressed(zoo_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_fix), np.ma.compressed(fix_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_wcd), np.ma.compressed(wcd_dir))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_sed), np.ma.compressed(sed_dir))[0,1] ] )

r_ind = np.array( [ np.corrcoef(np.ma.compressed(delta_no3), np.ma.compressed(no3_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_d15nno3), np.ma.compressed(d15nno3_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_d15npom), np.ma.compressed(d15npom_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_npp), np.ma.compressed(npp_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_zoo), np.ma.compressed(zoo_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_fix), np.ma.compressed(fix_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_wcd), np.ma.compressed(wcd_ind))[0,1], \
                    np.corrcoef(np.ma.compressed(delta_sed), np.ma.compressed(sed_ind))[0,1] ] )


from scipy import stats as scs

### test for normality

norm_del = np.array( [ scs.normaltest(np.ma.compressed(delta_no3)), \
                    scs.normaltest(np.ma.compressed(delta_d15nno3)), \
                    scs.normaltest(np.ma.compressed(delta_d15npom)), \
                     scs.normaltest(np.ma.compressed(delta_npp)), \
                    scs.normaltest(np.ma.compressed(delta_zoo)), \
                    scs.normaltest(np.ma.compressed(delta_fix)), \
                    scs.normaltest(np.ma.compressed(delta_wcd)), \
                    scs.normaltest(np.ma.compressed(delta_sed)) ] )

norm_dir = np.array( [ scs.normaltest(np.ma.compressed(no3_dir)), \
                    scs.normaltest(np.ma.compressed(d15nno3_dir)), \
                    scs.normaltest(np.ma.compressed(d15npom_dir)), \
                    scs.normaltest(np.ma.compressed(npp_dir)), \
                    scs.normaltest(np.ma.compressed(zoo_dir)), \
                    scs.normaltest(np.ma.compressed(fix_dir)), \
                    scs.normaltest(np.ma.compressed(wcd_dir)), \
                    scs.normaltest(np.ma.compressed(sed_dir)) ] )

norm_ind = np.array( [ scs.normaltest(np.ma.compressed(no3_ind)), \
                    scs.normaltest(np.ma.compressed(d15nno3_ind)), \
                    scs.normaltest(np.ma.compressed(d15npom_ind)), \
                    scs.normaltest(np.ma.compressed(npp_ind)), \
                    scs.normaltest(np.ma.compressed(zoo_ind)), \
                    scs.normaltest(np.ma.compressed(fix_ind)), \
                    scs.normaltest(np.ma.compressed(wcd_ind)), \
                    scs.normaltest(np.ma.compressed(sed_ind)) ] )

    
### calculation of correlation
    
spr_dir = np.array( [ scs.spearmanr(np.ma.compressed(delta_no3), np.ma.compressed(no3_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_d15nno3), np.ma.compressed(d15nno3_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_d15npom), np.ma.compressed(d15npom_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_npp), np.ma.compressed(npp_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_zoo), np.ma.compressed(zoo_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_fix), np.ma.compressed(fix_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_wcd), np.ma.compressed(wcd_dir)), \
                    scs.spearmanr(np.ma.compressed(delta_sed), np.ma.compressed(sed_dir)) ] )
    
spr_ind = np.array( [ scs.spearmanr(np.ma.compressed(delta_no3), np.ma.compressed(no3_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_d15nno3), np.ma.compressed(d15nno3_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_d15npom), np.ma.compressed(d15npom_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_npp), np.ma.compressed(npp_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_zoo), np.ma.compressed(zoo_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_fix), np.ma.compressed(fix_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_wcd), np.ma.compressed(wcd_ind)), \
                    scs.spearmanr(np.ma.compressed(delta_sed), np.ma.compressed(sed_ind)) ] )
    

print(np.min(spr_dir[:,0]), np.max(spr_dir[:,0]))
print(np.min(spr_ind[:,0]), np.max(spr_ind[:,0]))




    
#%% plot


fstic = 13
fslab = 15
alf = 0.7
wid = 0.8
lw = 2

cols = ['firebrick', 'k']
labs = ['direct effects (warming on rates)', 'indirect effects (circulation changes)']
xlabs = ['NO$^{EZ}_3$', '$\delta^{15}$N$^{TZ}_{NO_3}$', '$\delta^{15}$N$^{EZ}_{POM}$', 'NPP', 'Grazing', 'N$_2$ fix', 'WC$_{den}$', 'Sed$_{den}$', ' ', \
         'NO$^{EZ}_3$', '$\delta^{15}$N$^{TZ}_{NO_3}$', '$\delta^{15}$N$^{EZ}_{POM}$', 'NPP', 'Grazing', 'N$_2$ fix', 'WC$_{den}$', 'Sed$_{den}$', ' ']


fig = plt.figure(figsize=(10,5), facecolor='w')
gs = GridSpec(1,1)



ax1 = plt.subplot(gs[0])
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=fstic, bottom=False, labelbottom=False, top=False, labeltop=True)

'''
name = "Pearson's Correlation"
plt.bar(np.arange(0,8), r_dir, color=cols[0], label=labs[0], alpha=alf, linewidth=lw, edgecolor='k', width=wid)
plt.bar(np.arange(9,17), r_ind, color=cols[1], label=labs[1], alpha=alf, linewidth=lw, edgecolor='k', width=wid)
'''
name = "Spearman's Correlation"
plt.bar(np.arange(0,8), spr_dir[:,0], color=cols[0], label=labs[0], alpha=alf, linewidth=lw, edgecolor='k', width=wid)
plt.bar(np.arange(9,17), spr_ind[:,0], color=cols[1], label=labs[1], alpha=alf, linewidth=lw, edgecolor='k', width=wid)

plt.plot((-1,17),(0,0),'k-', linewidth=2)
plt.plot((-1,17),(-0.5,-0.5),'k--', linewidth=0.5)
plt.plot((-1,17),(0.5,0.5),'k--', linewidth=0.5)
plt.plot((-1,17),(0.9,0.9),'k--', linewidth=0.5)

plt.text(17.1,0.5, '0.5', fontsize=fstic, va='center', ha='left')
plt.text(17.1,0.9, '0.9', fontsize=fstic, va='center', ha='left')


plt.ylim(-0.5,1); plt.xlim(-1,17)
plt.xticks(np.arange(0,18), xlabs, rotation=90)
xticks = ax1.xaxis.get_major_ticks()
xticks[8].set_visible(False)

plt.ylabel(name, fontsize=fslab)
plt.legend(loc='upper center', frameon=False, ncol=5, bbox_to_anchor=(0.51,.15), columnspacing=4, fontsize=fstic)

plt.subplots_adjust(bottom=0.05, top=0.8)

#%% savefig

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-main4.png', dpi=500, bbox_inches='tight')
fig.savefig('fig-main4_trans.png', dpi=500, bbox_inches='tight', transparent=True)
fig.savefig('fig-main4.pdf', dpi=500, bbox_inches='tight')


