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

# %% [markdown]
# ## Load P3D archive and train emulator 
#
# There is a trained version already stored, jump to load emulator if do not want to wait

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

# %%
# %%time

p3d_emu = P3DEmulator(
    training_data=Archive3D.training_data,
    emu_input_names=Archive3D.emu_params,
    training_type='Arinyo_min',
    train=True,
    nepochs=1001,
    step_size=500,
    Nrealizations=5000,
    save_path=path_program+"/data/emulator_models/new_emu.pt",
)

# %%
arr_loss = np.array(p3d_emu.loss_arr)
plt.plot(-arr_loss)
plt.ylim(30, 41)

# %% [markdown]
# ### Load emulator

# %%
p3d_emu = P3DEmulator(
    model_path=path_program+"/data/emulator_models/new_emu.pt",
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
coeffs_all, coeffs_mean = p3d_emu.predict_Arinyos(
    emu_params=list_input_params,
    return_all_realizations=True,
    Nrealizations=1000,
)
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
coeffs_all, coeffs_mean = p3d_emu.predict_Arinyos(
    emu_params=input_params,
    return_all_realizations=True,
    Nrealizations=1000,
)
coeffs_mean

# %% [markdown]
# #### Convergence
#
# The emulator draws random samples internally, and their number
# is characterize by Nrealizations. The precision of the parameters
# as a function of the number of realizations is

# %%
# %%time

Ntot = 100000
coeffs_all, coeffs_mean = p3d_emu.predict_Arinyos(
    emu_params=input_params,
    return_all_realizations=True,
    Nrealizations=Ntot,
)
coeffs_mean

# %%
nx = np.geomspace(10, Ntot, 10)

for ii in range(coeffs_all.shape[1]):
    plt.loglog(
        nx, 
        np.std(coeffs_all[:, ii])/np.mean(coeffs_all[:, ii])/np.sqrt(nx),
        label=p3d_emu.Arinyo_params[ii]
    )
# 1% precision for 2000 realizations
plt.axhline(0.01, color="k")
plt.axvline(2e3, color="k")
plt.legend()

# %% [markdown]
# ## Get P3D and P1D
#
# You can also compute these quantities, but need to specify the cosmology. This is because we need Plin to
# evaluate the Arinyo model

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
    'As': 2.1e-09,
    'ns': 0.96,
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



# %%
# get only Arinyo params
info_power = {
    "cosmo": cosmo,
    "z": z_test,
}


out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    return_bias_eta=True,
    Nrealizations=2000
)
out

# %%
# %%time
Nrea = 1000
out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    Nrealizations=Nrea,
    seed=0,
    return_all_realizations=True
)

# %% [markdown]
# Also return P3D 

# %%
# ks at which compute P3D
k = np.logspace(-2, 1, 100)
# mu's at which compute P3D
mu = np.zeros_like(k)

k3d_Mpc = np.concatenate([k, k])
mu3d = np.concatenate([mu, mu+1])
Ntot = 100000

info_power = {
    "cosmo": cosmo,
    "z": z_test,
    "k3d_Mpc": k3d_Mpc,
    "mu": mu3d,
    "return_p3d": True,
    "return_cov": True,
}

out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    Nrealizations=Ntot
)
out.keys()

# %%
ii = 0
_ = out["mu"] == 0
plt.errorbar(
    out["k_Mpc"][_], 
    out["p3d"][_]/out["Plin"][_], 
    out["p3d_std"][_]/out["Plin"][_]/np.sqrt(Ntot), 
    label="mu=0",
    color = "C"+str(ii)
)
_ = out["mu"] == 1
plt.errorbar(
    out["k_Mpc"][_], 
    out["p3d"][_]/out["Plin"][_], 
    out["p3d_std"][_]/out["Plin"][_]/np.sqrt(Ntot), 
    label="mu=1",
    ls="--",
    color = "C"+str(ii)
)
plt.ylabel("P3D/Plin")
plt.xlabel("k")
plt.legend()
plt.xscale("log")

# %% [markdown]
# #### Convergence P3D

# %%
nx = np.geomspace(10, Ntot, 8)

for ii in range(8):
    # _ = out["mu"] == 0
    # plt.loglog(
    #     out["k_Mpc"][_], 
    #     out["p3d_std"][_]/out["Plin"][_]/np.sqrt(nx[ii]), 
    #     label="mu=0",
    #     ls="--",
    #     color = "C"+str(ii)
    # )
    # only look at mu=1, bigger error
    _ = out["mu"] == 1
    plt.loglog(
        out["k_Mpc"][_], 
        out["p3d_std"][_]/out["p3d"][_]/np.sqrt(nx[ii]), 
        label= str(int(nx[ii])),
        ls="-",
        color = "C"+str(ii)
    )


plt.axhline(1e-3, color="k")
plt.ylabel("std_p3d/p3d/sqrt(Nrea)")
plt.xlabel("k3d")
plt.legend()
plt.xscale("log")

# %% [markdown]
# Now get P1D

# %%
# ks at which compute P3D
k1d_Mpc = np.geomspace(0.01, 4, 100)

info_power = {
    "cosmo": cosmo,
    "z": z_test,
    "k1d_Mpc": k1d_Mpc,
    "return_p1d": True,
    "return_cov": True,
}
Ntot = 10000

out = p3d_emu.evaluate(
    emu_params=input_params,
    info_power=info_power,
    # natural_params=True,
    Nrealizations=Ntot
)
out.keys()

# %%
plt.plot(out["k1d_Mpc"], out["k1d_Mpc"]/np.pi*out["p1d"])
plt.ylabel("kpar*P1D/pi")
plt.xlabel("kpar")
plt.xscale("log")

# %%
nx = np.geomspace(10, Ntot, 8)

for ii in range(8):
    plt.loglog(
        out["k1d_Mpc"], 
        out["p1d_std"]/out["p1d"]/np.sqrt(nx[ii]), 
        label= str(int(nx[ii])),
        ls="-",
        color = "C"+str(ii)
    )


plt.axhline(4.5e-4, color="k")
plt.ylabel("std_P1D/P1D/sqrt(Nrea)")
plt.xlabel("kpar")
plt.legend()
plt.xscale("log")

# %%
out.keys()

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
