# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
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
from ForestFlow.model_p3d_arinyo import get_linP_interp
from ForestFlow.model_p3d_arinyo import ArinyoModel
import timeit
sys.path.append("../scripts")
sys.path.append("../forestflow")
from pcross import get_Px
import hankl

# First, choose a redshift and $k$ range. Initialize an instance of the Arinyo class for this redshift given cosmology calculations from Camb.

from lya_pk.model_p3d_arinyo import get_linP_interp
from lya_pk.model_p3d_arinyo import ArinyoModel
zs = np.array([2, 2.5])
cosmo = camb_cosmo.get_cosmology()
camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, camb_kmax_Mpc=100)
arinyo = ArinyoModel(cosmo=cosmo, camb_results=camb_results, zs=zs, camb_kmax_Mpc=100)
params = arinyo.default_params

# ## Plot the 3D power spectrum

# +
nn_k = 200
nn_mu = 10
k = np.logspace(-4, 8, nn_k)
kpar = np.logspace(-4, 3, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu)
mu2d = np.tile(mu[:, np.newaxis], nn_k).T

plin = arinyo.linP_Mpc(zs[0], k)
p3d = arinyo.P3D_Mpc(zs[0], k2d, mu2d, params)
p1d = arinyo.P1D_Mpc(zs[0], kpar, parameters=params)
# -

for ii in range(p3d.shape[1]):
    plt.loglog(k, p3d[:, ii]/plin, label=r'$<\mu>=$'+str(np.round(mu[ii], 2)))
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.xlim([10**-1,10**8])
plt.ylim([10**-10,10])
plt.legend()

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
# $$ P_{\times}(z,k_{\parallel}, r_\perp) = \frac{1}{2\pi} \int_{k_{\parallel}}^{\infty} k dk j_0 (k_\perp r_\perp) P_\mathrm{3D}(z, k, \mu)$$
#
# The get_Px function performs the Hankel transform to integrate P3D.
#
# Going the other way around, if one wanted to compute $P_\mathrm{3D}$ from a $P_\times$ measurement, we would do
# $$ P_\mathrm{3D}(z,k_{\perp}, k_{\parallel}) = 2\pi \int_{0}^{\infty} d\theta J_0 (k_\perp \theta) \theta P_\times (z, \theta, k_{\parallel})$$
#
# <!-- $$ P_{\times}(z,k_{\parallel}, \theta) = \frac{1}{2\pi} \int_{k_{\parallel}}^{\infty} k dk j_0 (k_\perp \theta) P(z, k, \mu)$$ -->
#

kpars_Px  = np.logspace(-3, np.log10(20), 100)
rperp_fid,Px_per_kpar_fid = get_Px(kpars_Px, arinyo, zs[1], min_kperp=10.**-20, max_kperp=10.**3, Nsteps_kperp=10000, trans_to_p1d=False)


# # Plot $P_\times$ as a function of $r_\perp$
# At very low rperp, $P_\times$ should match with P1D. We can plot the P1D predictions for the same model for few values of $k_\parallel$, to test the integration.

p1d_comparison = arinyo.P1D_Mpc(zs[1], kpars_Px, parameters=params)

# +
# first, check if the fiducial matches P1D at the low end
fig,ax = plt.subplots(nrows=2,ncols=1, figsize=[8,5], gridspec_kw={'height_ratios':[3,1]}, sharex=True)
cmap = mpl.colormaps['Set1']
delta = 0

kpars_to_plot  = np.arange(0,len(kpars_Px),10).astype(int)
kpar_plot = kpars_Px[kpars_to_plot]
for ik, Px in enumerate(Px_per_kpar_fid[kpars_to_plot]):
    ax[0].plot(rperp_fid, Px, label=f'$k_\parallel$={round(kpar_plot[ik],3)}', c=cmap(ik/len(kpars_to_plot)))
    ax[0].plot(rperp_fid,np.full(len(rperp_fid), p1d_comparison[kpars_to_plot][ik]), c=cmap(ik/len(kpars_to_plot)), linestyle='--')

    pctdiff = ((np.full(len(rperp_fid), p1d_comparison[kpars_to_plot][ik])-Px)/Px)*100
    ax[1].plot(rperp_fid, pctdiff, c=cmap(ik/len(kpars_to_plot)))

ax[0].legend()
ax[1].set_xlabel(r"$r_\perp$ [Mpc]")
ax[0].set_ylabel(r"$P_\times$ [Mpc]")
ax[1].set_ylabel("% diff")
plt.xscale("log")
# plt.yscale("log")
ax[0].set_xlim([.001,30])
ax[0].set_ylim([0,0.6])
ax[1].set_ylim([-1,1])
# plt.suptitle(r"$N_\mathrm{steps} = 5000, k_\perp^\mathrm{min}=10^{-20}, k_\perp^\mathrm{max}=10^3$")

# -


