# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial on how to use the Arinyo model

# %% [markdown]
# In this notebook we explain how to compute P3D and P1D from a particular Arinyo model

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.model_p3d_arinyo import ArinyoModel


# %% [markdown]
# ## Call Arinyo model for a particular cosmology
#
# For more details about the Arinyo model see Eq. 4.5 from Givans+22 (https://arxiv.org/abs/2205.00962)

# %%
cosmo = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    'ns': 0.9665,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
model_Arinyo = ArinyoModel(cosmo)

# %% [markdown]
# ### Compute P3D & P1D
#
# for the same cosmology

# %%
zs = 3. # redshift

# P3D
nn_k = 200 # number of k bins
nn_mu = 10 # number of mu bins
k = np.logspace(-1.5, 1, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

#P1D
kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D

arinyo_pars = {
    'bias': -0.18,
    'beta': 1.3,
    'q1': 0.4,
    'q2': 0.0,
    'kvav': 0.58,
    'av': 0.29,
    'bv': 1.55,
    'kp': 10.5
}

plin = model_Arinyo.linP_Mpc(zs, k) # get linear power spectrum at target zmodel_Arinyo
p3d = model_Arinyo.P3D_Mpc(zs, k2d, mu2d, arinyo_pars) # get P3D at target z
p1d = model_Arinyo.P1D_Mpc(zs, kpar, arinyo_pars) # get P1D at target z

# %% [markdown]
# #### Plot P3D

# %%
for ii in range(p3d.shape[1]):
    col = 'C'+str(ii)
    if ii % 3 == 0:
        lab = r'$<\mu>=$'+str(np.round(mu[ii], 2))
    else:
        lab = None
    plt.loglog(k, p3d[:, ii]/plin, col, label=lab)
    plt.plot(k, p3d[0, ii]/plin[0]+k[:]*0, col+'--')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.legend(loc='upper left')

# %% [markdown]
# #### Plot P1D

# %%
plt.plot(kpar, kpar * p1d/np.pi)
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}(k)$')
plt.xscale('log')

# %% [markdown]
# ## Call same model for a different cosmology
#
# Very useful during inference, much faster than creating more instances of the same model

# %%
cosmo_new = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    'ns': 0.8665, # different
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}

plin_new = model_Arinyo.linP_Mpc(zs, k, cosmo_new=cosmo_new) # get linear power spectrum at target zmodel_Arinyo
p3d_new = model_Arinyo.P3D_Mpc(zs, k2d, mu2d, arinyo_pars, cosmo_new=cosmo_new) # get P3D at target z
p1d_new = model_Arinyo.P1D_Mpc(zs, kpar, arinyo_pars, cosmo_new=cosmo_new) # get P1D at target z

# %%
plt.loglog(k, plin)
plt.loglog(k, plin_new)

# %% [markdown]
# #### Typically, we only change As and ns during fits, and it takes almost the same time

# %%
cosmo_new = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    'ns': 0.8665, # different
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}

# %%
# %%time
for ii in range(100):
    model_Arinyo.P3D_Mpc(zs, k2d, mu2d, arinyo_pars, cosmo_new=cosmo_new)

# %%
# %%time
for ii in range(100):
    model_Arinyo.P3D_Mpc(zs, k2d, mu2d, arinyo_pars)

# %% [markdown]
# #### Much slower when changing other parameters since we need to call camb every time

# %%
cosmo_new = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.219, # different
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    'ns': 0.9665,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}

# %% [markdown]
# Note we are calling the function 10 times

# %%
# %%time
for ii in range(10):
    model_Arinyo.P3D_Mpc(zs, k2d, mu2d, arinyo_pars, cosmo_new=cosmo_new)

# %% [markdown]
# ## Arinyo model from emulator

# %%
import forestflow
from forestflow.P3D_cINN import P3DEmulator

# %%
path_repo = os.path.dirname(forestflow.__path__[0]) + '/'
emulator = P3DEmulator(
    model_path = path_repo + "/data/emulator_models/forest_mpg",
)

# %%
emulator.

# %% [markdown]
# #### Ratio with best-fitting model

# %%
mask = k3d_Mpc[:,0] < 4
for ii in range(0, p3d_Mpc.shape[1]):
    lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
    plt.plot(k3d_Mpc[mask, ii], p3d_pred[mask, ii]/model_p3d[mask, ii]-1, label=lab)
plt.plot(k3d_Mpc[mask, 0], k3d_Mpc[mask, 0]*0, 'k--')
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.legend()

# %%
# mask = k1d_Mpc < 4
# plt.plot(k1d_Mpc[mask], p1d_pred[mask]/model_p1d[mask]-1, '-', label='Sim/Model-1')
# plt.plot(k1d_Mpc[mask], k1d_Mpc[mask]*0, 'k--')
# plt.xscale('log')
# plt.xlabel(r'$k$ [Mpc]')
# plt.ylabel(r'$P_{\rm 1D}$')
# plt.legend()

# %%
