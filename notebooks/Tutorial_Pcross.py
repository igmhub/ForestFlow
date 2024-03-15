# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: lace
# ---

# # Tutorial for how to calculate $P_\times$

#
# This notebook should be run in an environment that contains both LaCE and ForestFlow.

import numpy as np
from scipy import special
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
# import P3D theory
from lace.cosmo import camb_cosmo
from forestflow.model_p3d_arinyo import get_linP_interp
from forestflow.model_p3d_arinyo import ArinyoModel
import time
# %load_ext autoreload
# %autoreload 2
from forestflow.pcross import Px_Mpc
import hankl

# First, choose a redshift and $k$ range. Initialize an instance of the Arinyo class for this redshift given cosmology calculations from Camb.

zs = np.array([2,2.5]) # set target redshift
cosmo = camb_cosmo.get_cosmology() # set default cosmo
camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, camb_kmax_Mpc=200) # set default cosmo
arinyo = ArinyoModel(cosmo=cosmo, camb_results=camb_results, zs=zs, camb_kmax_Mpc=200) # set model
arinyo.default_params

# ## Plot the 3D power spectrum

# +
nn_k = 200 # number of k bins
nn_mu = 10 # number of mu bins
k = np.logspace(-1.5, 2, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D

plin = arinyo.linP_Mpc(zs[0], k) # get linear power spectrum at target z
p3d = arinyo.P3D_Mpc(zs[0], k2d, mu2d, arinyo.default_params) # get P3D at target z
p1d = arinyo.P1D_Mpc(zs[0], kpar, parameters=arinyo.default_params) # get P1D at target z
# -

for ii in range(p3d.shape[1]):
    plt.loglog(k, p3d[:, ii]/plin, label=r'$<\mu>=$'+str(np.round(mu[ii], 2)))
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.xlim([10**-1,10**8])
plt.ylim([10**-10,10])
plt.legend()

# we can compute Px from within the Arinyo class:
rperp, Px_per_kpar = arinyo.Px_Mpc(zs[0], kpar, arinyo.default_params)

# we could have also done it outside of the class with the function Px_Mpc:
rperp2, Px_per_kpar2 = Px_Mpc(
    kpar,
    arinyo.P3D_Mpc,
    zs[0],
    P3D_mode='pol',
    **{'pp':arinyo.default_params})

np.sum(Px_per_kpar == Px_per_kpar2), Px_per_kpar.size # these will be equal if the result is the same

# # Calculate $P_\times$ for a series of $k_\parallel$.
#
# Observationally, $P_\times$ is a measurement made between two sightlines separated by the angle $\theta$ on the sky. As such, it contains 3D information about correlations. It is relevant because we have distinct sightlines measured by spectroscopic instruments like DESI.
#
# In theory, $P_\times$ is an alternative expression of 3D power using a combination of real space and $k$ space. The overdensity of line-of-sight modes $k_\parallel$ at two sky positions have correlation $2\pi P_\times$ if the $k_\parallel$ modes are the same (and none otherwise). If the 'two' sky positions are equal, this becomes a 1D measurement, only measuring power along the line-of-sight, called P1D.
#
# In real space, one can think of this as examining the 3D correlation between overdensities separated by a given $\theta$, at redshift $z$, and Fourier transforming *only* along the line-of-sight to move from "real" (wavelength) space to $k$ space.
#
# $$P_\times (z, \theta, k_{\parallel}) \equiv \int d \Delta \lambda e^{i \Delta \lambda k_{\parallel}}\xi_\mathrm{3D}(z,\theta,\Delta \lambda) $$
#
# If one considers instead the 3D power spectrum $P_\mathrm{3D}$, as a function of transverse scalar mode $k_\perp$ and line-of-sight $k_\parallel$, one rather needs to (inverse) 2D Fourier transform the power spectrum *only* in the perpendicular modes, to go from $k_{\perp}$ space to $\theta$ space in the transverse direction while continuing to express the power in Fourier space along the line-of-sight.
# $$ P_\times (z, \theta, k_{\parallel}) = \int \frac{d^2 k_{\perp}}{(2\pi)^2} e^{i\boldsymbol{\theta} \cdot \boldsymbol{k_{\perp}}} P_\mathrm{3D} (z, k_{\perp}, k_\parallel)$$
#
# This integral is adding up all of the different $k_{\perp}$ modes that enter into the real-space delta fluctuations at a given $\theta$ separation.
#
# To solve, it can be rearranged to become:
# $$ P_{\times}(z,k_{\parallel}, r_\perp) = \frac{1}{2\pi} \int_{k_{\parallel}}^{\infty} k dk J_0 (k_\perp r_\perp) P_\mathrm{3D}(z, k, \mu)$$
#
# The Px_Mpc function performs the Hankel transform to integrate P3D.
#
# Going the other way around, if one wanted to compute $P_\mathrm{3D}$ from a $P_\times$ measurement, we would do
# $$ P_\mathrm{3D}(z,k_{\perp}, k_{\parallel}) = 2\pi \int_{0}^{\infty} d\theta J_0 (k_\perp \theta) \theta P_\times (z, \theta, k_{\parallel})$$
#
# <!-- $$ P_{\times}(z,k_{\parallel}, \theta) = \frac{1}{2\pi} \int_{k_{\parallel}}^{\infty} k dk J_0 (k_\perp \theta) P(z, k, \mu)$$ -->
#

# choose some values of k parallel to compute
kpars_Px  = np.logspace(-3, np.log10(20), 100)
rperp,Px_per_kpar = Px_Mpc(kpars_Px,
    arinyo.P3D_Mpc,
    zs[0],
    P3D_mode='pol',
    **{'pp':arinyo.default_params})


# # Plot $P_\times$ as a function of $r_\perp$
# At very low rperp, $P_\times$ should match with P1D. We can plot the P1D predictions for the same model for few values of $k_\parallel$, to test the integration.

p1d_comparison = arinyo.P1D_Mpc(zs[0], kpars_Px, parameters=arinyo.default_params) # get the P1D comparison

# +
# first, check if the fiducial matches P1D at the low end
fig,ax = plt.subplots(nrows=2,ncols=1, figsize=[8,5], gridspec_kw={'height_ratios':[3,1]}, sharex=True)
cmap = mpl.colormaps['Set1']
delta = 0

kpars_to_plot  = np.arange(0,len(kpars_Px),10).astype(int)
kpar_plot = kpars_Px[kpars_to_plot]
for ik, Px in enumerate(Px_per_kpar[kpars_to_plot]):
    ax[0].plot(rperp, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap(ik/len(kpars_to_plot)))
    ax[0].plot(rperp,np.full(len(rperp), p1d_comparison[kpars_to_plot][ik]), c=cmap(ik/len(kpars_to_plot)), linestyle='--')

    pctdiff = ((np.full(len(rperp), p1d_comparison[kpars_to_plot][ik])-Px)/Px)*100
    ax[1].plot(rperp, pctdiff, c=cmap(ik/len(kpars_to_plot)))

ax[0].legend()
ax[1].set_xlabel(r"$r_\perp$ [Mpc]")
ax[0].set_ylabel(r"$P_\times$ [Mpc]")
ax[1].set_ylabel("% diff")
plt.xscale("log")
# plt.yscale("log")
ax[0].set_xlim([.01,1])
ax[0].set_ylim([0,np.amax(Px_per_kpar)+.1])
ax[1].set_ylim([-1,1])
plt.suptitle(r"$P_\times$ vs P1D, default settings")

# -

# # Now let us look at $P_\times$ in a different way, as a function of $k_\parallel$ for different $r_\perp$ values

# series of rperp we're interested in
rperps_toplot = np.array([0, 0.2,0.972,2.204,3.444,5.941])/cosmo.h # Example, from Abdul-Karim et al 2023
print("Trying to plot r_perp =", rperps_toplot, "Mpc")
rperp_sel,Px_per_kpar_sel = Px_Mpc(kpars_Px,
    arinyo.P3D_Mpc,
    zs[0],
    rperp_choice=rperps_toplot,
    P3D_mode='pol',
    **{'pp':arinyo.default_params})
Px_per_rperp_sel = Px_per_kpar_sel.T # transpose the array to get the correct ordering
print("Going to plot:", rperp_sel, "Mpc")

# +
# check that the first one has no fractional difference
fig,ax = plt.subplots(nrows=2,ncols=1, figsize=[8,5], gridspec_kw={'height_ratios':[3,1]}, sharex=True)
delta = 0
ax[0].plot(kpars_Px, p1d_comparison, label='arinyo model 1D', color='k')
ax[0].plot(kpars_Px, Px_per_rperp_sel[0], linestyle='dashed', color='yellow', label='first Px')


pctdiff = (Px_per_rperp_sel[0]-p1d_comparison)/p1d_comparison*100
ax[1].plot(kpars_Px, pctdiff)

ax[0].legend()
ax[1].set_xlim([0.1,20])
ax[0].set_ylim([10**-7,1.1])
ax[0].set_yscale("log")
ax[0].set_xscale("log")
ax[0].set_ylabel(r"$P_\times$ [Mpc/$h$]")
ax[1].set_xlabel(r"$k_{\parallel}$ [$h$ Mpc$^{-1}$]")

# -

# These match perfectly as they should, since our first rperp is very close to 0.

# Now, let's try to reproduce the plot from Abdul-Karim et al 2023
#

# +
fig,ax = plt.subplots(1,1)
colors=['blue','orange','green','red','yellow','purple']
plt.title(r"$P_\times$ from FFTLog integration")
print(rperp_sel)
for r, Px in enumerate(Px_per_rperp_sel):
    plt.loglog(kpars_Px/cosmo.h, Px*cosmo.h, 'o', label=f"$r_{{\perp}}=${round(rperp_sel[r],3)*0.675} Mpc/h", ms=5, c=colors[r])

ax.set_xlim([0.4,20])
ax.set_ylim([10**-7,10**1])
ax.loglog(kpars_Px/cosmo.h, p1d_comparison*cosmo.h, label='arinyo model 1D', color='0.8')
ax.set_ylabel(r"$P_\times$ [Mpc/$h$]")
ax.set_xlabel(r"$k_{\parallel}$ [$h$ Mpc$^{-1}$]")
ax.set_yticks([10**-7,10**-5,10**-3,10**-1,10])
ax.grid()
plt.legend()
# -

# # If you want to use the pcross function with several variations, we can use Px_detailed

from forestflow.pcross import Px_Mpc_detailed

# +
rperp_full,Px_per_kpar_full = Px_Mpc_detailed(
    kpars_Px,
    arinyo.P3D_Mpc,
    zs[0],
    P3D_mode='pol',
    min_rperp=10**-20,
    max_rperp=1000,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    nkperp=2**16,
    trans_to_p1d=True,
    interpmin = 0.005,
    interpmax = 0.2,
    fast_transition=False, 
    **{'pp':arinyo.default_params}
)

rperp_alt,Px_per_kpar_alt = Px_Mpc_detailed(
    kpars_Px,
    arinyo.P3D_Mpc,
    zs[0],
    P3D_mode='pol',
    min_rperp=10**-20,
    max_rperp=1000,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    nkperp=2**14,
    trans_to_p1d=False, 
    **{'pp':arinyo.default_params}
)

# +
# check accuracy with respect to fiducial

same_rperp = [np.argmin(abs(rperp_full-rp)) for rp in rperp_alt]


fig,ax = plt.subplots(nrows=3,ncols=1, figsize=[8,8], gridspec_kw={'height_ratios':[3,1,1]}, sharex=True)
delta = 0

kpars_to_plot  = np.arange(0,len(kpars_Px),10).astype(int)
kpar_plot = kpars_Px[kpars_to_plot]
for ik, Px in enumerate(Px_per_kpar_full[kpars_to_plot]):
    ax[0].plot(rperp_full, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap(ik/len(kpars_to_plot)))
    ax[0].plot(rperp_alt, Px_per_kpar_alt[kpars_to_plot[ik]], c='k', linestyle='dotted')
    pctdiff = (Px_per_kpar_alt[kpars_to_plot[ik]]-Px[same_rperp])/Px[same_rperp]*100
    absdiff = Px_per_kpar_alt[kpars_to_plot[ik]]-Px[same_rperp]
    ax[1].plot(rperp_alt, pctdiff, c=cmap(ik/len(kpars_to_plot)))
    ax[2].plot(rperp_alt, absdiff, c=cmap(ik/len(kpars_to_plot)))
    # add a tolerance
    if len(rperp_alt[rperp_alt<80][pctdiff[rperp_alt<80]>0.1])>0:
        print("minimum", np.amin(rperp_alt[pctdiff>.1]), "tolerance exceeded.")
        print("maximum", np.amax(rperp_alt[rperp_alt<80][pctdiff[rperp_alt<80]>.1]), "tolerance exceeded.")

ax[0].legend()
ax[1].set_xlabel(r"$r_\perp$ [Mpc]")
ax[0].set_ylabel(r"$P_\times$ [Mpc]")
ax[1].set_ylabel("% diff")
ax[2].set_ylabel("abs. diff")
plt.xscale("log")
# plt.yscale("log")
ax[0].set_xlim([.001,1000])
ax[0].set_ylim([0,0.9])
ax[1].set_ylim([-1,1])
plt.suptitle(r"$N_\mathrm{steps} = 2^{14}, k_\perp^\mathrm{min}=10^{-20}, k_\perp^\mathrm{max}=10^3$")


# -

# ## We can see that by lowering nkperp from 2^16 to 2^14, the integration accuracy has worsened

# There are some strange discrete jumps when looking at the differences, e.g., at rperp~8. However, when we investigate further this appears to be very minor (see below plots). Therefore lowering Nkperp can be safe, but is worth testing depending on the accuracy to time tradeoff an analysis needs.

for ik, Px in enumerate(Px_per_kpar_full[kpars_to_plot][:3]):
    plt.plot(rperp_full, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap(ik/len(kpars_to_plot)))
    plt.plot(rperp_alt, Px_per_kpar_alt[kpars_to_plot[ik]], c='k', linestyle='dotted')
plt.xlim([5,10])
plt.ylim([.125,.23])
plt.legend()

for ik, Px in enumerate(Px_per_kpar_full[kpars_to_plot][3:7]):
    plt.plot(rperp_full, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap((ik+3)/len(kpars_to_plot)))
    plt.plot(rperp_alt, Px_per_kpar_alt[kpars_to_plot[ik+3]], c='k', linestyle='dotted')
plt.xlim([5,10])
plt.ylim([.01,.3])
plt.legend()

for ik, Px in enumerate(Px_per_kpar_full[kpars_to_plot][7:9]):
    plt.plot(rperp_full, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap((ik+7)/len(kpars_to_plot)))
    plt.plot(rperp_alt, Px_per_kpar_alt[kpars_to_plot[ik+7]], c='k', linestyle='dotted')
plt.xlim([5,10])
plt.ylim([10**-13,.01])
plt.legend()
plt.yscale('log')


