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
# # ForestFlow tutorial
#
# In this tutorial we explain how to:
# - Access best-fitting P3D model parameters to simulation measurements
# - Use ForestFlow

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_test_p3d
from forestflow.P3D_cINN import P3DEmulator

# %%
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program

# %% [markdown]
# ### LOAD P3D ARCHIVE

# %%
# %%time
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ### Extract best-fitting parameters at z=3

# %%
params = "Arinyo_min"
ztarget = 3
val_params = []
for sim in Archive3D.training_data:
    if(sim["z"] == ztarget):
    # if(sim["z"] >= 2) & (sim["z"] < 3):
        val_params.append(sim[params])

# %%
nsims = len(val_params)
name_params = val_params[0].keys()
nparams = len(name_params)
arr_val_params = np.zeros((nsims, nparams))

for ii in range(nsims):
    for jj, pname in enumerate(name_params):
        arr_val_params[ii, jj] = val_params[ii][pname]

# %%
fig, ax = plt.subplots(4, 2)
ax = ax.reshape(-1)
for jj, pname in enumerate(name_params): 
    ax[jj].hist(arr_val_params[:,jj], bins=20);
    ax[jj].set_xlabel(pname)
plt.tight_layout()

# %%
# plt.hist(arr_val_params[:,2], bins=30);
# plt.xlabel("q1")
# plt.tight_layout()
# plt.savefig("prior_q1_z2_z275.png")

# %% [markdown]
# ## Load Emulator
#
# We have pre-trained emulators, no need for training

# %%
train_emu = False

if train_emu:
    p3d_emu = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        training_type='Arinyo_min',
        save_path=path_program+"/data/emulator_models/new_emu.pt",
    )
else:
    p3d_emu = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        Nrealizations=50000,
        training_type='Arinyo_min',
        model_path=path_program+"/data/emulator_models/mpg_hypercube.pt",
    )

# %%
Archive3D.emu_params

# %% [markdown]
# ### Evaluate emulator

# %% [markdown]
# Only get value of P3D model parameters

# %%
# target redshift
z_test = 3

# target cosmology
cosmo = {
    'H0': 67.0,
    'omch2': 0.12,
    'ombh2': 0.022,
    'mnu': 0.0,
    'omk': 0,
    'As': 2.2e-09,
    'ns': 0.94,
    'nrun': 0.0,
    'w': -1.0
}

# Cosmological and IGM input parameters. No need to specify 
# cosmological parameters when target cosmology is provided
# IGM parameters from random simulation, access via sim.keys()
input_params = {
    # 'Delta2_p': 0., # not used if you provide cosmology
    # 'n_p': 0., # not used if you provide cosmology
    'mF': 0.66,
    'sigT_Mpc': 0.13,
    'gamma': 1.5,
    'kF_Mpc': 10.5
}

info_power = {
    "cosmo": cosmo,
    "z": z_test,
}

out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    # Nrealizations=100
)
out.keys()

# %%
for par in input_params:
    print(par, Archive3D.training_data[0][par])

# %%
# %%time
out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    # Nrealizations=100
)

# %%
# %%time
coeffs_mean = p3d_emu.predict_Arinyos(
    Archive3D.training_data[0],
    # return_all_realizations=False,
    Nrealizations=10000,
)
# coeffs_mean/coeffs_mean2-1

# %%
# %%time
coeffs_mean = p3d_emu.predict_Arinyos(
    Archive3D.training_data[0],
    return_all_realizations=False,
    # Nrealizations=100000,
    plot=True
)
# coeffs_mean/coeffs_mean2-1

# %%
# value of parameters
print(out["coeffs_Arinyo"])
# and emulation error
print(out["coeffs_Arinyo_std"])

# for the target cosmology:
# small-scale amplitude, slope, and running of linear power spectrum
# f_p provides the logarithmic growth factor
print(out["linP_zs"])

# %% [markdown]
# Return b_eta instead of beta

# %%
out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    Nrealizations=10000
)

# value of parameters
print(out["coeffs_Arinyo"])
# and emulation error
# print(out["coeffs_Arinyo_std"])

# %% [markdown]
# Also return P3D 

# %%
# ks at which compute P3D
k = np.logspace(-2, 1, 100)
# mu's at which compute P3D
mu = np.zeros_like(k)

k3d_Mpc = np.concatenate([k, k])
mu3d = np.concatenate([mu, mu+1])

info_power = {
    "cosmo": cosmo,
    "z": z_test,
    "k3d_Mpc": k3d_Mpc,
    "mu": mu3d,
    "return_p3d": True,
}

out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    Nrealizations=10000
)
out.keys()

# %%
_ = out["mu"] == 0
plt.plot(out["k_Mpc"][_], out["p3d"][_]/out["Plin"][_], label="mu=0")
_ = out["mu"] == 1
plt.plot(out["k_Mpc"][_], out["p3d"][_]/out["Plin"][_], label="mu=1")
plt.ylabel("P3D/Plin")
plt.xlabel("k")
plt.legend()
plt.xscale("log")

# %% [markdown]
# Now get P1D

# %%
# ks at which compute P3D
k1d_Mpc = np.logspace(-1, 1, 100)

info_power = {
    "cosmo": cosmo,
    "z": z_test,
    "k1d_Mpc": k1d_Mpc,
    "return_p1d": True,
}

out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    Nrealizations=100
)
out.keys()

# %%
plt.plot(out["k1d_Mpc"], out["k1d_Mpc"]/np.pi*out["p1d"])
plt.ylabel("kpar*P1D/pi")
plt.xlabel("kpar")
plt.xscale("log")

# %%

# %%
