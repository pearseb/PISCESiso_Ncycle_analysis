# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:41:56 2020

     - explore isotopic fractionation of nitrogen by phytoplankton following water parcel since upwelling

State variables
    no3     - bioavailable N                                                                    mmol N m-3
    no3_15  - 15-N of bioavailable N                                                            mmol 15N m-3
    phy     - N in phytoplankton biomass                                                        mmol N m-3
    phy_15  - 15-N in phytoplankton biomass                                                     mmol 15N m-3
    tot     - total N in model                                                                  mmol N m-3
    tot_15  - total 15-N in model                                                               mmol 15N m-3

Fluxes
    N_upt   - N loss due to uptake by phytoplankton (dependent on growth rate(T) and fixed N availability)          mmol N m-3 day-1
    N_rec   - N gain due to recycling of phytoplankton biomass (inversely proportional to phytoplankton biomass)    mmol N m-3 day-1
    N_exp   - N loss due to export of phytoplankton as PON (equal to 1 - N_rec)                                     mmol N m-3 day-1

Defined variables
    no3_i       - initial no3 concentration
    phy_i       - initial phy concentration
    no3_15_i    - initial no3_15 concentration
    phy_15_i    - initial phy_15 concentration
    e_phy       - fractionation factor of phytoplankton assimilation 
    T_c         - temperature of water
    k_n         - nitrogen uptake half-saturation coefficient
    L_lim       - mean light limitation in the euphotic zone
    Fe_lim      - mean iron limitation term in the euphotic zone
    
@author: pearseb
"""

#%% imports

from __future__ import unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean
import cmocean.cm as cmo
import seaborn as sb
sb.set(style='ticks')

from tqdm import tqdm


#%% set initial concentrations of state variables and other constants

# time conditions
maxt = 100            # days
dt = 96               # timesteps per day

no3_i = 20.0          # upwelled nitrate into euphotic zone of equatorial region
d15n_no3_i = 6.0      # initial d15N signature of upwelled nitrate
phy_i = no3_i*0.02    # concentration of total phytoplankton
d15n_phy_i = 1.0      # initial d15N signature of equatorial phytoplankton

# growth conditions
T_c = 18.0            # temperature
k_n = 1.0             # nitrogen half saturation constant for michaelis menton kinetic uptake
L_lim = 0.15          # mean light limitation term over the euphotic zone in picontrol run
Fe_lim = 0.4          # mean iron limitation term over the euphotic zone (stops excessive uptake in high nitrate zone)

# isotopic constants
e_npp = 5.0         # fractionation factor associated with primary production


#%% model


def phyto_frac_0D(maxt, dt,
                  T_c, no3_i, phy_i, d15n_no3_i, d15n_phy_i, e_npp, 
                  L_lim, Fe_lim, k_n, 
                  plot, print_solutions):

    # recycling and export
    p_mort = 0.01       # quadratic mortality coefficient for phytoplankton (mmol N m-3)-1 day-1
    p_resp = 0.01       # linear respiration coefficient for phytoplankton  day-1
    k_mort = 0.2        # half-saturation constant for phytoplankton mortality
    f_rec = 0.4         # minimum fraction of detritus that is recycled
    T_rec = 0.035       # temperature-dependent scaler on recycling
    
    
    # initialise arrays 
    tot = np.zeros((maxt*dt))  # total nitrogen in system
    tot_15 = np.zeros((maxt*dt))  # total 15-nitrogen in system
    no3 = np.zeros((maxt*dt))
    phy = np.zeros((maxt*dt))
    no3_15 = np.zeros((maxt*dt))
    phy_15 = np.zeros((maxt*dt))
    N_upt = np.zeros((maxt*dt))
    N_rec = np.zeros((maxt*dt))
    N_exp = np.zeros((maxt*dt))
    N_fra = np.zeros((maxt*dt))
    N15_upt = np.zeros((maxt*dt))
    N15_rec = np.zeros((maxt*dt))
    N15_exp = np.zeros((maxt*dt))
    N_lim = np.zeros((maxt*dt))
    d15N_no3 = np.zeros((maxt*dt))
    d15N_phy = np.zeros((maxt*dt))
    d15N_exp = np.zeros((maxt*dt))
    

    # calculate maximum potential growth of phytoplankton using Temp
    tgfunc = np.exp(0.063913*T_c)             # temperature power function describing max potential biological rates
    u_max = 0.6 * tgfunc                      # maximum potential growth rate of phytoplantkon set by temperature (units of per day)

    
    # run forward timestepping model
    t1 = 0; t2 = maxt*dt
    timesteps = np.arange(t1,t2,1)
        
    for i,t in enumerate(timesteps[t1:t2]):
        
        dtr = 1./dt
    
        # initialise arrays if at first timestep
        if i == 0:
            # nitrate and phytoplankton
            no3[i] = no3_i
            phy[i] = no3[i]*0.02
            # retrieve no3_15 and phy_15
            no3_15[i] = no3_i*(d15n_no3_i*1e-3+1.0)
            phy_15[i] = phy[i]*(d15n_phy_i*1e-3+1.0)
            # get delta-15N values of nitrate and phytoplankton
            d15N_no3[i] = (no3_15[i]/no3[i]-1)*1000
            d15N_phy[i] = (phy_15[i]/phy[i]-1)*1000
            # keep an eye on total budget of N and N15 to ensure conservation of mass
            tot[i] = no3[i] + phy[i]
            tot_15[i] = no3_15[i] + phy_15[i]
        
        else: # update state variables at new timestep
            # nitrate and phytoplankton
            no3[i] = no3_n
            phy[i] = phy_n
            # 15 nitrate and phytos
            no3_15[i] = no3_15_n
            phy_15[i] = phy_15_n
            # calculate delta-15N values of nitrate and phytoplankton
            d15N_no3[i] = (no3_15_n/no3_n-1)*1000
            d15N_phy[i] = (phy_15_n/phy_n-1)*1000
            # calculate delta-15N values of nitrate and phytoplankton
            tot[i] = no3[i] + phy[i] + np.sum(N_exp)
            tot_15[i] = no3_15[i] + phy_15[i] + np.sum(N15_exp)
        
        
        # 1. N_upt
        N_lim[i] = no3[i] / (no3[i] + k_n)
        N_upt[i] = u_max * L_lim * min(Fe_lim, N_lim[i]) * phy[i] * dtr
        N_fra[i] = e_npp * N_lim[i]
        N15_upt[i] = N_upt[i] * no3_15[i]/no3[i] * (1.0 - 1e-3*N_fra[i])
        
        # 2. recylcing and export
            # quadratic terms for phytoplankton respiration and mortality losses
        phy_min = max(phy[i] - 0.01, 0.0)
        phy_mort = p_mort * phy_min * phy[i]
        phy_resp = p_resp * phy_min/(phy[i] + k_mort) * phy[i]
        phy_loss = phy_resp + phy_mort
            # recycling and export based on phytoplankton losses
        rec_exp = f_rec + T_rec * tgfunc   # 0.4 is base recycling:export ratio and is increased at higher temperatures
        N_rec[i] = phy_loss * rec_exp * dtr
        N_exp[i] = phy_loss * (1.0 - rec_exp) * dtr
        N15_rec[i] = N_rec[i] * (phy_15[i]/phy[i])
        N15_exp[i] = N_exp[i] * (phy_15[i]/phy[i])
        
        # uptake state variables
        no3_n = no3[i] - N_upt[i] + N_rec[i]
        phy_n = phy[i] + N_upt[i] - N_rec[i] - N_exp[i]
        no3_15_n = no3_15[i] - N15_upt[i] + N15_rec[i]
        phy_15_n = phy_15[i] + N15_upt[i] - N15_rec[i] - N15_exp[i]
        
        
    # calcualte d15N of exported matter
    d15N_exp = (np.cumsum(N15_exp)/np.cumsum(N_exp) - 1.0)*1000
    
    # calcualte d15N of total matter
    d15N_tot = (tot_15/tot - 1.0)*1000
    
    
    
    ### plot 
    if plot == True:
        fslab = 15
        lab1 = ['DIN', 'POM', 'ExpM', 'Total N']
        lab2 = ['DIN uptake', 'POM recycled', 'POM exported']
        lab3 = ['$\epsilon^{15}_{phy}$']
        lab4 = ['$\delta^{15}$N$_{DIN}$', '$\delta^{15}$N$_{POM}$', '$\delta^{15}$N$_{ExpM}$', '$\delta^{15}$N$_{total}$']
        
        fig = plt.figure(figsize=(10,9.5))
        gs = GridSpec(4,1)
        
        ax1 = plt.subplot(gs[0])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelbottom=False)
        plt.plot(timesteps[t1:t2], no3[t1:t2], linestyle='-', label=lab1[0])
        plt.plot(timesteps[t1:t2], phy[t1:t2], linestyle='--', label=lab1[1])
        plt.plot(timesteps[t1:t2], np.cumsum(N_exp[t1:t2]), linestyle=':', label=lab1[2])
        plt.plot(timesteps[t1:t2], tot[t1:t2], linestyle='-.', label=lab1[3])
        plt.ylim(-1,25)
        plt.ylabel('mmol m$^{-3}$')
        plt.legend()
        
        ax2 = plt.subplot(gs[1])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelbottom=False)
        plt.plot(timesteps[t1:t2], N_upt[t1:t2]/dtr, linestyle='-', label=lab2[0])
        plt.plot(timesteps[t1:t2], N_rec[t1:t2]/dtr, linestyle='--', label=lab2[1])
        plt.plot(timesteps[t1:t2], N_exp[t1:t2]/dtr, linestyle=':', label=lab2[2])
        #plt.ylim(0,0.02)
        plt.ylabel('mmol m$^{-3}$ day$^{-1}$')
        plt.legend()
        
        ax3 = plt.subplot(gs[2])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(labelbottom=False)
        plt.plot(timesteps[t1:t2], N_fra[t1:t2], linestyle='-', label=lab3[0])
        plt.ylim(0,5.1)
        plt.ylabel('\u2030')
        plt.legend()
        
        # mask d15N when no3 is below 0.3 mmol m-3
        #d15N_no3 = np.ma.masked_where(no3 < 0.3, d15N_no3)
        ax4 = plt.subplot(gs[3])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(labelbottom=True)
        plt.title('$\delta^{15}$N')
        plt.plot(timesteps[t1:t2], d15N_no3[t1:t2], linestyle='-', label=lab4[0])
        plt.plot(timesteps[t1:t2], d15N_phy[t1:t2], linestyle='--', label=lab4[1])
        plt.plot(timesteps[t1:t2], d15N_exp[t1:t2], linestyle=':', label=lab4[2])
        plt.plot(timesteps[t1:t2], d15N_tot[t1:t2], linestyle='-.', label=lab4[3])
        plt.ylim(0,15)
        plt.ylabel('\u2030')
        plt.xlabel('time/distance')
        plt.legend()
        
        
        plt.subplots_adjust(top=0.95, hspace=0.3)
        
        
        xx = 0.025; yy=1.025
        plt.text(xx,yy,'a', transform=ax1.transAxes, va='center', ha='center', fontsize=fslab+2, fontweight='bold')
        plt.text(xx,yy,'b', transform=ax2.transAxes, va='center', ha='center', fontsize=fslab+2, fontweight='bold')
        plt.text(xx,yy,'c', transform=ax3.transAxes, va='center', ha='center', fontsize=fslab+2, fontweight='bold')
        plt.text(xx,yy,'d', transform=ax4.transAxes, va='center', ha='center', fontsize=fslab+2, fontweight='bold')
    
    
    if print_solutions == True:
        print('no3 = ', no3[i], 'd15N-no3 = ', d15N_no3[i], 'max d15N-no3 = ', np.max(d15N_no3))
        print('pon = ', phy[i], 'd15N-pon = ', d15N_phy[i])
        print('exp = ', np.sum(N_exp), 'd15N-exp = ', d15N_exp[i])
        print('tot = ', tot[i], 'd15N-tot = ', d15N_tot[i])
        print('d15N-no3 minus d15N-pon = ', d15N_no3[i] - d15N_phy[i])

    
    return d15N_no3, d15N_exp, fig


#%% produce figures informing schematic (Figure 3d)

no3_i = 20
(out1_d15N_no3, out1_d15N_exp, fig) = phyto_frac_0D(maxt, dt, 
                                               T_c, no3_i, phy_i, d15n_no3_i, d15n_phy_i, e_npp,
                                               L_lim, Fe_lim, k_n,
                                               plot=True, print_solutions=True)

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-supp10.png', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp10.eps', dpi=300, bbox_to_inches='tight')
fig.savefig('fig-supp10_trans.png', dpi=300, bbox_to_inches='tight', transparent=True)


no3_i = 10
(out2_d15N_no3, out2_d15N_exp, fig) = phyto_frac_0D(maxt, dt, 
                                               T_c, no3_i, phy_i, d15n_no3_i, d15n_phy_i, e_npp,
                                               L_lim, Fe_lim, k_n,
                                               plot=True, print_solutions=True)

fig = plt.figure(figsize=(8,3))
plt.plot(np.arange(maxt*dt), out1_d15N_no3, 'b-')
plt.plot(np.arange(maxt*dt), out1_d15N_exp, 'k-')
plt.plot(np.arange(maxt*dt), out2_d15N_no3, 'b--')
plt.plot(np.arange(maxt*dt), out2_d15N_exp, 'k--')
plt.xlim(9600,0)

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//figures")
fig.savefig('fig-inset_for_schematic.png', dpi=300, bbox_to_inches='tight')


#%% explore changes to initial no3

# set constants for model
maxt=100; dt=24

# create array of initial nitrogen concentrations
no3_i = 20
no3 = np.linspace(no3_i*0.5, no3_i, maxt)

# initialise arrays for saving model output
out_d15N_no3 = np.zeros(len(no3))
out_d15N_exp = np.zeros(len(no3))


# run model
for xx,nit in tqdm(enumerate(no3)):
    (tmp_d15N_no3, tmp_d15N_exp) = phyto_frac_0D(maxt, dt, T_c, 
                                                 nit, phy_i, d15n_no3_i, d15n_phy_i, e_npp,
                                                 L_lim, Fe_lim, k_n,
                                                 plot=False, print_solutions=False)
    out_d15N_no3[xx] = tmp_d15N_no3[-1]
    out_d15N_exp[xx] = tmp_d15N_exp[-1]


### find linear fit of d15N to initial nitrogen concentration
from scipy.optimize import curve_fit
def func(x,a,b):
    return a*x + b

xx = (1-no3/no3[-1])*100
yy = out_d15N_exp-out_d15N_exp[-1]
popt, pcov = curve_fit(func, xx, yy)

# save output
residuals = yy - func(xx, popt[0], popt[1])
ss_res = np.sum(residuals**2); ss_tot = np.sum( (yy - np.mean(yy))**2 ); r2 = 1 - (ss_res/ss_tot)
slope = popt[0]; interc = popt[1]; error = np.sqrt(np.diag(pcov))[0]

# save arrays for plotting
xx1 = (1-no3/no3[-1])*100
yy1 = out_d15N_exp-out_d15N_exp[-1]


#%% explore effect of temperature as initial nitrogen changes

# set constants for model
maxt=100; dt=24

# create array of initial nitrogen concentrations
no3_i = 20
no3 = np.linspace(no3_i*0.5, no3_i, 21)

# create array of temperature
tem = np.linspace(16, 20, 21)

# initialise arrays for saving model output
out_d15N_no3 = np.zeros((len(no3), len(tem)))
out_d15N_exp = np.zeros((len(no3), len(tem)))

# run model
for xx,nit in tqdm(enumerate(no3)):
    for yy,t_c in enumerate(tem):
        (tmp_d15N_no3, tmp_d15N_exp) = phyto_frac_0D(maxt, dt, t_c, 
                                                     nit, phy_i, d15n_no3_i, d15n_phy_i, e_npp,
                                                     L_lim, Fe_lim, k_n,
                                                     plot=False, print_solutions=False)
        out_d15N_no3[xx,yy] = tmp_d15N_no3[-1]
        out_d15N_exp[xx,yy] = tmp_d15N_exp[-1]


### find linear fit of d15N to initial nitrogen concentration
from scipy.optimize import curve_fit
def func(x,a,b):
    return a*x + b

slopes = []
for i in np.arange(len(tem)):
    xx = (1-no3/no3[-1])*100
    yy = out_d15N_exp[:,i]-out_d15N_exp[-1,i]
    popt, pcov = curve_fit(func, xx, yy)
    slopes.append(popt[0])

slopes = np.array(slopes)
print(np.mean(slopes), np.min(slopes), np.max(slopes))

# save arrays for plotting
xx2 = (1-no3/no3[-1])*100
tint = np.int(np.where(tem==18.0)[0])
yy2 = out_d15N_exp[:,:] - np.max(out_d15N_exp[:,tint])



#%% plot output

fstic = 13
fslab = 15
alf = 0.8
size = 100
wid = 2

fig = plt.figure(figsize=(7,9.5))
gs = GridSpec(2,1)

ax1 = plt.subplot(gs[0])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=fstic)

plt.scatter(xx1, yy1, s=size, alpha=alf, color='k', marker='o', edgecolor='none')
plt.plot(xx1, func(xx1, slope, interc), color='firebrick', alpha=alf, linewidth=wid) 

plt.text(0.5,0.8, 'r$^2$ = %.3f'%(r2), transform=ax1.transAxes, ha='left', va='center', fontsize=fstic)
plt.text(0.5,0.7, '$\Delta$ $\delta^{15}$N$_{POM}$ = %.2f $\cdot$ $\Delta$N%%'%(slope), transform=ax1.transAxes, ha='left', va='center', fontsize=fstic)

plt.ylabel('$\Delta$ $\delta^{15}$N$_{POM}$ (\u2030)', fontsize=fslab)
#plt.xlabel('% decrease in bioavailable nitrogen', fontsize=fslab)

plt.xlim(0,50)
plt.ylim(-1,0)

colmap = cmocean.tools.lighten(cmo.amp, alpha=0.8)
zmin = np.round(np.min(yy2),1)
zmax = np.round(np.max(yy2),1)
levs = np.arange(zmin, zmax+0.1, 0.1)

ax2 = plt.subplot(gs[1])
ax2.tick_params(labelsize=fstic)

p1 = plt.contourf(xx2, tem - T_c, np.transpose(yy2), cmap=colmap, levels=levs, vmin=np.min(levs), vmax=np.max(levs), extend='both')
c1 = plt.contour(xx2, tem - T_c, np.transpose(yy2), colors='k', levels=[0])

plt.ylabel('$\Delta$ T ($^{\circ}$)', fontsize=fslab)
plt.xlabel('% decrease in bioavailable nitrogen', fontsize=fslab)
plt.yticks(np.arange(-2,2.1,1), np.arange(-2,2.1,1))

plt.subplots_adjust(top=0.925, bottom=0.18, hspace=0.25, right=0.92, left=0.15)

x = 0.025; y = 1.05
plt.text(x,y, 'a', transform=ax1.transAxes, fontsize=fslab+2, va='center', ha='center', fontweight='bold')
plt.text(x,y, 'b', transform=ax2.transAxes, fontsize=fslab+2, va='center', ha='center', fontweight='bold')

cbax1 = fig.add_axes([0.2, 0.075, 0.675, 0.03])
cbar1 = plt.colorbar(p1, cax=cbax1, orientation='horizontal', ticks=levs[::2])
cbax1.tick_params(labelsize=fstic)
cbar1.ax.invert_xaxis()
cbar1.set_label('$\Delta$ $\delta^{15}$N$_{POM}$ (\u2030)', fontsize=fslab)


#%% save figure

plt.clabel(c1, manual=True, fontsize=fstic, fmt='%i')

os.chdir("C://Users//pearseb//Dropbox//PostDoc//my articles//d15N and d13C in PISCES//scripts_for_publication//supplementary_figures")
fig.savefig('fig-supp11.png', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp11.eps', dpi=300, bbox_inches='tight')
fig.savefig('fig-supp11_trans.png', dpi=300, bbox_inches='tight', transparent=True)