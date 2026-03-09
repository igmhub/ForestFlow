# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # ForestFlow tutorial

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from lace.cosmo import cosmology
import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator

# %% [markdown]
# ## Load emulator
#
# Here to directly load the emulator

# %%
emulator = P3DEmulator(
    model_path=os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "forest_mpg")
)

# %% [markdown]
# ## Evaluate emulator to get Arinyo parameters

# %% [markdown]
# #### You can provide multiple inputs at once

# %%
list_input_params = [
    {'Delta2_p': 0.18489945277410613,
      'n_p': -2.331713201486465,
      'mF': 0.23475637218289533,
      'sigT_Mpc': 0.10040737452608385,
      'gamma': 1.2115605945334802,
      'kF_Mpc': 14.191866950067904},
     {'Delta2_p': 0.20276666703485943,
      'n_p': -2.3317132064538915,
      'mF': 0.310236058401032,
      'sigT_Mpc': 0.10751395885731446,
      'gamma': 1.2059890102644482,
      'kF_Mpc': 13.177851268715806},
]

# %%
# %%time
coeffs_mean = emulator.predict_Arinyos(emu_params=list_input_params)
coeffs_mean

# %% [markdown]
# #### Or just one

# %%
input_params = {
    'Delta2_p': 0.18489945277410613,
    'n_p': -2.331713201486465,
    'mF': 0.23475637218289533,
    'sigT_Mpc': 0.10040737452608385,
    'gamma': 1.2115605945334802,
    'kF_Mpc': 14.191866950067904
}

# %%
# %%time
coeffs_mean = emulator.predict_Arinyos(
    emu_params=input_params,
)
coeffs_mean

# %% [markdown]
# #### return numpy array for compatibility with old version

# %%
# %%time
coeffs_mean = emulator.predict_Arinyos(
    emu_params=input_params,
    return_dict=False
)
coeffs_mean

# %% [markdown]
# ## Get P3D and P1D
#
# See Tutorial_Arinyo for more info about the ArinyoModel class

# %%
from forestflow.model_p3d_arinyo import ArinyoModel
#from lace.cosmo import camb_cosmo, fit_linP

# %%
# target cosmology
cosmo_params = {
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
cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params)

# set Arinyo model
model_Arinyo = ArinyoModel(fid_cosmo=cosmo)

# Compute compressed parameters for the target cosmology
z = 4.
kp_Mpc = 0.7
linP_zs = cosmo.get_linP_Mpc_params(z, kp_Mpc)

# define input parameters to emulator
input_emu = {
    "Delta2_p": linP_zs["Delta2_p"],
    "n_p": linP_zs["n_p"],
    'mF': 0.23475637218289533,
    'sigT_Mpc': 0.10040737452608385,
    'gamma': 1.2115605945334802,
    'kF_Mpc': 14.191866950067904
}


# %% [markdown]
# #### Predict Arinyo with emulator

# %%
par_ari = emulator.predict_Arinyos(input_emu, return_dict=True)
par_ari

# %% [markdown]
# ### Get power

# %%
# %%time
# P3D
nn_k = 200 # number of k bins
nn_mu = 10 # number of mu bins
k = np.logspace(-1.5, 1, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

#P1D
kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D



plin = model_Arinyo.linP_Mpc(z, k) # get linear power spectrum at target zmodel_Arinyo
p3d = model_Arinyo.P3D_Mpc_k_mu(z, k2d, mu2d, par_ari) # get P3D at target z
p1d = model_Arinyo.P1D_Mpc(z, kpar, par_ari) # get P1D at target z

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

# %%
plt.plot(kpar, kpar * p1d/np.pi)
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}(k)$')
plt.xscale('log')

# %%

# %% [markdown]
# ## For developers, train emulator

# %%
# %%time
Archive3D = GadgetArchive3D()

# %%
train = True
if train:
    emulator = P3DEmulator(
        training_data=Archive3D.training_data,
        emu_input_names=Archive3D.emu_params,
        training_type='Arinyo_min',
        train=True,
        nepochs=4000,
        batch_size=20,
        step_size=200,
        weight_decay=0.01,
        Nrealizations=6000,
        save_path=os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test")
        # save_path=os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "forest_mpg")
    )

# %%
arr_loss = np.array(emulator.loss_arr)
plt.plot(-arr_loss)
plt.ylim(20, 41)
plt.axvline(4000)
# plt.xscale("log")

# %%

# %%
