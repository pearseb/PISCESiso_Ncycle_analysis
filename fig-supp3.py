# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:13:16 2020

@author: pearseb
"""

#%% imports

import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sb
sb.set(style='ticks')

from matplotlib.gridspec import GridSpec


#%% get data

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")

data = nc.Dataset('ndep_Tg_yr.nc')
print(data)
ndep = data.variables['NDEP_TGYR'][...]
years = np.arange(1801,2101,1)

data.close()


#%% plotting aeolian Nr dep

fstic = 14
fslab = 16
alf = 0.7

fig = plt.figure(figsize=(10,6))

gs = GridSpec(1,1)

ax1 = plt.subplot(gs[0])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=fstic)

plt.plot(years, ndep, color='k', alpha=alf, linewidth=2)
plt.xlim(1800,2101)
plt.ylabel('nitrogen deposition over ocean (Tg N yr$^{-1}$)', fontsize=fslab, labelpad=12)
plt.xlabel('Year (Common Era)', fontsize=fslab, labelpad=6)


#%%

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp3.png',dpi=300,bbox_to_inches='tight')
fig.savefig('fig-supp3.eps',dpi=300,bbox_to_inches='tight')
fig.savefig('fig-supp3_trans.png',dpi=300,bbox_to_inches='tight', transparent=True)




