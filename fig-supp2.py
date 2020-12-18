# -*- coding: utf-8 -*-
"""
Created on Wed May 03 08:56:59 2017


Purpose
-------
    To create taylor diagram of improvement caused by COM bgc model in other
    biogeochemical fields
    

@author: pearseb
"""

#%% imports

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
from taylor import TaylorDiagram
import cmocean.cm as cmo
from matplotlib.gridspec import GridSpec
import seaborn as sb
sb.set(style='ticks')


#%%
def create_TaylorDee(Nstd, r2, Nbias, std_range=(0.0,1.5), labels=[' '], fields=['o'], markers=['$1$'], title=' ', text=' '):
    '''
    Creates a Taylor Diagram from the data that is given to the function.
    
    The datasets to be compared must be given as a list composed of two arrays
    with equal dimensions.
    
    eg. create_TaylorDee([d1,d2], weights=weight)
    
    d1 = data for comparion (model results)
    d2 = observations
    
    If newfig is set to True (default), then the function creates a new Taylor
    Diagram. If it si False, then it adds the previous Taylor Diagram.

    '''
    # plot taylor diagram
    fig = plt.figure(figsize=(10,7),facecolor='w')
    fig.suptitle(title, y=0.95, family='serif', fontsize=16)
    
    for m in np.arange(len(markers)):
        if (m == 0):    
            td = TaylorDiagram(1.0, fig=fig, label='Reference', srange=std_range)
            
        colour = Nbias[m]
        colour = (Nbias[m]+1.0)/2.0  # forcing values to fit on colormap, such that 0 = white
    
        td.add_sample(Nstd[m], r2[m], marker=fields[m], ms=20, mfc=cmo.balance(colour), mec='grey', zorder=1, linestyle='None')
        td.ax.plot(np.arccos(r2[m]), Nstd[m], marker=markers[m], linestyle='None',color='k', ms=15, mfc='w', zorder=2, label=labels[m])
    td.ax.text(np.arccos(0.01),std_range[1]+std_range[1]/12, text, fontsize=15, color='firebrick', fontweight='bold', family='sans-serif', ha='left', va='center')
            
    #fig.legend(td.samplePoints[1::], labels, loc='lower left', numpoints=1, scatterpoints=None, \
    #           bbox_to_anchor=(0.25,0.675))

    
    # Add RMS contours, and label them
    contours = td.add_contours(levels=5, colors='0.5', zorder=0) # 5 levels
    td.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    td.ax.plot((np.arccos([0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99]),np.arccos([0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99])), \
               ([std_range[0],std_range[0],std_range[0],std_range[0],std_range[0],std_range[0]] ,\
                [std_range[1],std_range[1],std_range[1],std_range[1],std_range[1],std_range[1]]), \
               linestyle=':',color='grey', zorder=0, alpha=0.75)
    
    plt.subplots_adjust(left=0.01)
    
    cbax = fig.add_axes([0.8,0.15,0.03,0.7])
    cmap = cmo.balance
    #bounds = np.arange(-1,1.1,0.2)
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #cbar = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='vertical', extend='both')
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cbar = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, orientation='vertical', extend='both')
    cbar.set_label('Normalised bias', fontsize=15)
    cbax.tick_params(labelsize=13)


#%% d15N data from ORCA1 simulation by Laurent

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//data_for_publication")
data = np.genfromtxt('d15nstats.txt', delimiter=',')
r2 = data[:,6]
Nbias = data[:,3] 
Nstd = data[:,4]


marks = ['$G$', '$S$', '$A$', '$P$', '$I$']
fields = ['o', 'o', 'o', 'o', 'o']
labels = ['Global', 'Southern Ocean', 'Atlantic', 'Pacific', 'Indian']

create_TaylorDee(Nstd,r2,Nbias, std_range=(0.0,1.5), labels=labels, fields=fields, markers=marks, title='PISCES$_{iso}$', text='$\delta^{15}$N$_{NO_3}$')
os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
plt.savefig('fig-supp2.png',dpi=300,bbox_inches='tight')
plt.savefig('fig-supp2.pdf',dpi=300,bbox_inches='tight')
plt.savefig('fig-supp2_trans.png',dpi=300,bbox_inches='tight',transparent=True)


