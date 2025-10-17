# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
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
from forestflow.pcross import Px_Mpc, Px_Mpc_detailed
import hankl

# First, choose a redshift and $k$ range. Initialize an instance of the Arinyo class for this redshift given cosmology calculations from Camb.

zs = np.array([2, 2.5])  # set target redshift
cosmo = camb_cosmo.get_cosmology()  # set default cosmo
camb_results = camb_cosmo.get_camb_results(
    cosmo, zs=zs, camb_kmax_Mpc=200
)  # set default cosmo
arinyo = ArinyoModel(
    cosmo=cosmo, camb_results=camb_results, zs=zs, camb_kmax_Mpc=200
)  # set model
arinyo.default_params

# ## Plot the 3D power spectrum

# +
nn_k = 200  # number of k bins
nn_mu = 10  # number of mu bins
k = np.logspace(-1.5, 2, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu)  # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T  # mu grid for P3D

kpar = np.logspace(-1, np.log10(5), nn_k)  # kpar for P1D

plin = arinyo.linP_Mpc(zs[0], k)  # get linear power spectrum at target z
p3d = arinyo.P3D_Mpc(
    zs[0], k2d, mu2d, arinyo.default_params
)  # get P3D at target z
p1d = arinyo.P1D_Mpc(
    zs[0], kpar, parameters=arinyo.default_params
)  # get P1D at target z
# -

for ii in range(p3d.shape[1]):
    plt.loglog(
        k, p3d[:, ii] / plin, label=r"$<\mu>=$" + str(np.round(mu[ii], 2))
    )
plt.xlabel(r"$k$ [Mpc]")
plt.ylabel(r"$P/P_{\rm lin}$")
plt.xlim([10**-1, 10**8])
plt.ylim([10**-10, 10])
plt.legend()

rperp = np.logspace(-2,2,100) # use the same rperp for each z. We could also input this as a list of [rperp, rperp] for each z.

arinyo.default_params

if arinyo.default_params:
    print("true")

# +
# we can compute Px from within the Arinyo class using default parameters,
Px_Mpc_1 = arinyo.Px_Mpc(z=zs[0], kpar_iMpc = kpar, rperp_Mpc = rperp, parameters=arinyo.default_params)

# we could have also done it outside of the class with the function Px_Mpc:
Px_Mpc_2 = Px_Mpc(
    zs[0], kpar, rperp, arinyo.P3D_Mpc, P3D_mode="pol", P3D_params=arinyo.default_params
)
print("Detailed method is equal to previous method:", np.allclose(Px_Mpc_1, Px_Mpc_2, atol=1e-15))
# -

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

# choose some values of k parallel to compute. We can do this for two redshifts at once, but we will evaluate
# the same kpar and rperp for both redshifts. It is also possible to input a list of different kpar and rperp
# for each redshift, e.g. [kpar1, kpar2] and [rperp1, rperp2].
kpars_Px = np.logspace(-3, np.log10(20), 100)
# add a 0 to kpars_Px to make sure that kpar=0 works fine
kpars_Px = np.append(0, kpars_Px)
Px_per_theta_perz = Px_Mpc(
    zs,
    kpars_Px,
    rperp,
    arinyo.P3D_Mpc,
    P3D_mode="pol",
    P3D_params=[arinyo.default_params, arinyo.default_params], # use the same parameters for both redshifts
)

# # Plot $P_\times$ as a function of $r_\perp$
# At very low rperp, $P_\times$ should match with P1D. We can plot the P1D predictions for the same model for few values of $k_\parallel$, to test the integration.

p1d_comparison = []
for iz, z in enumerate(zs):
    p1d_comparison.append(arinyo.P1D_Mpc(
        zs[iz], kpars_Px, parameters=arinyo.default_params
    ))  # get the P1D comparison

# +
# first, check if the fiducial matches P1D at the low end
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=[8, 5],
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)
cmap = mpl.colormaps["Set1"]
delta = 0

kpars_to_plot = np.arange(0, len(kpars_Px), 10).astype(int)
kpar_plot = kpars_Px[kpars_to_plot]
# just work with first z
iz = 0
for ik, Px in enumerate(Px_per_theta_perz[iz].T[kpars_to_plot]):
    ax[0].plot(
        rperp,
        Px,
        label=f"$k_\parallel$={round(kpar_plot[ik],3)}",
        c=cmap(ik / len(kpars_to_plot)),
    )
    ax[0].plot(
        rperp,
        np.full(len(rperp), p1d_comparison[iz][kpars_to_plot][ik]),
        c=cmap(ik / len(kpars_to_plot)),
        linestyle="--",
    )

    pctdiff = (
        (np.full(len(rperp), p1d_comparison[iz][kpars_to_plot][ik]) - Px) / Px
    ) * 100
    ax[1].plot(rperp, pctdiff, c=cmap(ik / len(kpars_to_plot)))

ax[0].legend()
ax[1].set_xlabel(r"$r_\perp$ [Mpc]")
ax[0].set_ylabel(r"$P_\times$ [Mpc]")
ax[1].set_ylabel("% diff")
plt.xscale("log")
# plt.yscale("log")
ax[0].set_xlim([0.01, 1])
ax[0].set_ylim([0, np.amax(Px_per_theta_perz) + 0.1])
ax[1].set_ylim([-1, 1])
plt.suptitle(r"$P_\times$ vs P1D, default settings")

# -

# # Now let us look at $P_\times$ in a different way, as a function of $k_\parallel$ for different $r_\perp$ values

# +
# series of rperp we're interested in
rperp = (
    np.array([0, 0.2, 0.972, 2.204, 3.444, 5.941]) / cosmo.h
)  
# the following will give a warning because we are inputting the same Arinyo parameter values for each redshift.
# If you want to input different values for the different redshifts, these should be input in format:
# {'bias': [b1,b2], 'beta': [beta1,beta2], ...} for each redshift.

Px_sel = Px_Mpc(
    zs,
    kpars_Px,
    rperp,
    arinyo.P3D_Mpc,
    P3D_mode="pol",
    P3D_params=arinyo.default_params,
)

# +

# check that the first one has no fractional difference
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=[8, 5],
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)
delta = 0
ax[0].plot(kpars_Px, p1d_comparison[0], label="arinyo model 1D, z=0", color="k")
ax[0].plot(kpars_Px, p1d_comparison[1], label="arinyo model 1D, z=1", color="grey")
ax[0].plot(
    kpars_Px,
    Px_sel[0][0],
    linestyle="dashed",
    color="yellow",
    label=f"first Px, z={zs[0]}"
)
ax[0].plot(
    kpars_Px,
    Px_sel[1][0],
    linestyle="dotted",
    color="green",
    label=f"first Px, z={zs[1]}"
)


pctdiff_z0 = (Px_sel[0][0] - p1d_comparison[0]) / p1d_comparison[0] * 100
pctdiff_z1 = (Px_sel[1][0] - p1d_comparison[1]) / p1d_comparison[1] * 100
if np.allclose(pctdiff_z0, 0, atol=1e-15):
    print("First Px matches P1D at z=0")
if np.allclose(pctdiff_z1, 0, atol=1e-15):
    print("First Px matches P1D at z=1")
ax[1].plot(kpars_Px, pctdiff_z0, color="yellow")
ax[1].plot(kpars_Px, pctdiff_z1, color="green", linestyle="--")

ax[0].legend()
ax[1].set_xlim([0.1, 20])
ax[1].set_ylim([-.005, .005])
ax[0].set_ylim([10**-7, 1.1])
ax[0].set_yscale("log")
ax[0].set_xscale("log")
ax[0].set_ylabel(r"$P_\times$ [Mpc/$h$]")
ax[1].set_xlabel(r"$k_{\parallel}$ [$h$ Mpc$^{-1}$]")

# -

# Do the same in linear plot

# +

# check that the first one has no fractional difference
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=[8, 5],
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)
delta = 0
ax[0].plot(kpars_Px, p1d_comparison[0], label="arinyo model 1D, z=0", color="k")
ax[0].plot(kpars_Px, p1d_comparison[1], label="arinyo model 1D, z=1", color="grey")
ax[0].plot(
    kpars_Px,
    Px_sel[0][0],
    linestyle="dashed",
    color="yellow",
    label=f"first Px, z={zs[0]}"
)
ax[0].plot(
    kpars_Px,
    Px_sel[1][0],
    linestyle="dotted",
    color="green",
    label=f"first Px, z={zs[1]}"
)


pctdiff_z0 = (Px_sel[0][0] - p1d_comparison[0]) / p1d_comparison[0] * 100
pctdiff_z1 = (Px_sel[1][0] - p1d_comparison[1]) / p1d_comparison[1] * 100
if np.allclose(pctdiff_z0, 0, atol=1e-15):
    print("First Px matches P1D at z=0")
if np.allclose(pctdiff_z1, 0, atol=1e-15):
    print("First Px matches P1D at z=1")
ax[1].plot(kpars_Px, pctdiff_z0, color="yellow")
ax[1].plot(kpars_Px, pctdiff_z1, color="green", linestyle="--")

ax[0].legend()
ax[1].set_xlim([-0.05, 5])
ax[1].set_ylim([-.005, .005])
ax[0].set_ylim([10**-7, 1.1])
# ax[0].set_yscale("log")
# ax[0].set_xscale("log")
ax[0].set_ylabel(r"$P_\times$ [Mpc/$h$]")
ax[1].set_xlabel(r"$k_{\parallel}$ [$h$ Mpc$^{-1}$]")

# -

# These match perfectly as they should, since our first rperp is very close to 0, where the code transitions to using the P1D result.

# Now, let's try to reproduce the plot from Abdul-Karim et al 2023
#

colors = ["blue", "orange", "green", "red", "yellow", "purple"]
for iz, z in enumerate(zs):
    fig, ax = plt.subplots(1, 1)
    
    print("Plotting redshift", z)
    for r, Px in enumerate(Px_sel[iz]):
        plt.loglog(
            kpars_Px / cosmo.h,
            Px * cosmo.h,
            "o",
            label=f"$r_{{\perp}}=${round(rperp[r],3)*0.675} Mpc/h",
            ms=5,
            c=colors[r],
        )

    ax.set_xlim([0.4, 20])
    ax.set_ylim([10**-7, 10**1])
    ax.loglog(
        kpars_Px / cosmo.h,
        p1d_comparison[iz] * cosmo.h,
        label="arinyo model 1D",
        color="0.8",
    )
    ax.set_ylabel(r"$P_\times$ [Mpc/$h$]")
    ax.set_xlabel(r"$k_{\parallel}$ [$h$ Mpc$^{-1}$]")
    ax.set_yticks([10**-7, 10**-5, 10**-3, 10**-1, 10])
    ax.grid()
    plt.legend()
    plt.title(rf"$P_\times$ for z = {z}")
    plt.show()
    plt.clf()

# # If you want to use the pcross function with several variations, we can use Px_detailed

rperp = np.logspace(-4,3, 1000)

# +
# change the value of nkerp to a very high value to get a 'perfect' integration via Hankel transform:

Px_Mpc_full = Px_Mpc_detailed(
    zs[0],
    kpars_Px,
    rperp,
    arinyo.P3D_Mpc,
    P3D_mode="pol",
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    nkperp=2**16,
    interpmin=0.005,
    interpmax=0.2,
    fast_transition=False,
    P3D_params =arinyo.default_params,
)

# compare with a lower value of nkperp to see the difference:
Px_Mpc_lownkperp = Px_Mpc_detailed(
    zs[0],
    kpars_Px,
    rperp,
    arinyo.P3D_Mpc,
    P3D_mode="pol",
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    nkperp=2**14,
    P3D_params =arinyo.default_params,
)

# +
# check accuracy with respect to fiducial


fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=[8, 8],
    gridspec_kw={"height_ratios": [3, 1, 1]},
    sharex=True,
)
delta = 0

kpars_to_plot = np.arange(0, len(kpars_Px), 10).astype(int)
kpar_plot = kpars_Px[kpars_to_plot]
for ik, Px in enumerate(Px_Mpc_full.T[kpars_to_plot]):
    ax[0].plot(
        rperp,
        Px,
        label=f"$k_\parallel$={round(kpar_plot[ik],3)}",
        c=cmap(ik / len(kpars_to_plot)),
    )
    ax[0].plot(
        rperp, Px_Mpc_lownkperp.T[kpars_to_plot[ik]], c="k", linestyle="dotted"
    )
    pctdiff = (
        (Px_Mpc_lownkperp.T[kpars_to_plot[ik]] - Px)
        / Px
        * 100
    )
    absdiff = Px_Mpc_lownkperp.T[kpars_to_plot[ik]] - Px
    ax[1].plot(rperp, pctdiff, c=cmap(ik / len(kpars_to_plot)))
    ax[2].plot(rperp, absdiff, c=cmap(ik / len(kpars_to_plot)))
    # add a tolerance
    if len(rperp[rperp < 80][pctdiff[rperp < 80] > 0.1]) > 0:
        print(
            "minimum", np.amin(rperp[pctdiff > 0.1]), "tolerance exceeded."
        )
        print(
            "maximum",
            np.amax(rperp[rperp < 80][pctdiff[rperp < 80] > 0.1]),
            "tolerance exceeded.",
        )
# import mline
import matplotlib.lines as mlines
black_dotted = mlines.Line2D([], [], color="k", linestyle="dotted")
solid_line = mlines.Line2D([], [], color="k", linestyle="solid")
# add to existing legend
handles, labels = ax[0].get_legend_handles_labels()
handles.append(black_dotted)
handles.append(solid_line)
labels.append(r"$N_\mathrm{kperp}=2^{14}$")
labels.append(r"$N_\mathrm{kperp}=2^{16}$")
ax[0].legend(handles, labels)

ax[1].set_xlabel(r"$r_\perp$ [Mpc]")
ax[0].set_ylabel(r"$P_\times$ [Mpc]")
ax[1].set_ylabel("% diff")
ax[2].set_ylabel("abs. diff")
plt.xscale("log")
# plt.yscale("log")
ax[0].set_xlim([0.001, 1000])
ax[0].set_ylim([0, 0.9])
ax[1].set_ylim([-1, 1])
plt.suptitle(
    r"$N_\mathrm{steps} = 2^{14}, k_\perp^\mathrm{min}=10^{-20}, k_\perp^\mathrm{max}=10^3$"
)


# -

# ## We can see that by lowering nkperp from 2^16 to 2^14, the integration accuracy has worsened

# There are some strange discrete jumps when looking at the differences, e.g., at rperp~8. However, when we investigate further this appears to be very minor (see below plots). Therefore lowering Nkperp can be safe, but is worth testing depending on the accuracy to time tradeoff an analysis needs.

for ik, Px in enumerate(Px_Mpc_full.T[kpars_to_plot][:3]):
    plt.plot(
        rperp,
        Px,
        label=f"$k_\parallel$={round(kpar_plot[ik],3)}",
        c=cmap(ik / len(kpars_to_plot)),
    )
    plt.plot(
        rperp, Px_Mpc_lownkperp.T[kpars_to_plot[ik]], c="k", linestyle="dotted"
    )
plt.xlim([5, 10])
plt.ylim([0.125, 0.23])
plt.legend()

for ik, Px in enumerate(Px_Mpc_full.T[kpars_to_plot][3:7]):
    plt.plot(
        rperp,
        Px,
        label=f"$k_\parallel$={round(kpar_plot[ik],3)}",
        c=cmap((ik + 3) / len(kpars_to_plot)),
    )
    plt.plot(
        rperp,
        Px_Mpc_lownkperp.T[kpars_to_plot[ik + 3]],
        c="k",
        linestyle="dotted",
    )
plt.xlim([5, 10])
plt.ylim([0.01, 0.3])
plt.legend()

for ik, Px in enumerate(Px_Mpc_full.T[kpars_to_plot][7:9]):
    plt.plot(
        rperp,
        Px,
        label=f"$k_\parallel$={round(kpar_plot[ik],3)}",
        c=cmap((ik + 7) / len(kpars_to_plot)),
    )
    plt.plot(
        rperp,
        Px_Mpc_lownkperp.T[kpars_to_plot[ik + 7]],
        c="k",
        linestyle="dotted",
    )
plt.xlim([5, 10])
plt.ylim([10**-13, 0.01])
plt.legend()
plt.yscale("log")
