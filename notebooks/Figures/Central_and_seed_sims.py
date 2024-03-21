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
# # Central vs Seed figure (old)

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.plots_v0 import plot_err_uncertainty

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder

path_program = ls_level(os.getcwd(), 1)
print(path_program)
sys.path.append(path_program)

# %%

# %% [markdown]
# ## LOAD P3D ARCHIVE

# %%
# %%time
folder_lya_data = path_program +  "/data/best_arinyo/"
#folder_interp = path_program+"/data/plin_interp/"
folder_chains='/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/'

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1], 
    folder_data=folder_lya_data, 
    force_recompute_plin=False,
    average='both'
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## TRAIN EMULATOR

# %%
# p3d_emu = P3DEmulator(
#     Archive3D.training_data,
#     Archive3D.emu_params,
#     nepochs=300,
#     lr=0.001,#0.005
#     batch_size=20,
#     step_size=200,
#     gamma=0.1,
#     weight_decay=0,
#     adamw=True,
#     nLayers_inn=12,#15
#     Archive=Archive3D,
#     Nrealizations=1000,
#     model_path='../data/emulator_models/mpg_hypercube.pt'
# )

p3d_emu = P3DEmulator(
        Archive3D.training_data,
        Archive3D.emu_params,
        nepochs=1,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        model_path="../data/emulator_models/mpg_hypercube.pt",
    )

# %% [markdown]
# ## PLOT TEST SIMULATION AT z=z_test

# %%
sim_label = ['mpg_central','mpg_seed']
z_test = 3.0

Nsim = 30
Nz = 11
zs = np.flip(np.arange(2, 4.6, 0.25))

k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

k_mask = (k_Mpc < 4) & (k_Mpc > 0)

k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]

k_p1d_Mpc = Archive3D.training_data[0]["k_Mpc"]
k1d_mask = (k_p1d_Mpc < 5) & (k_p1d_Mpc > 0)
k_p1d_Mpc = k_p1d_Mpc[k1d_mask]
norm = k_p1d_Mpc / np.pi

# %%

# %%
sim_label = 'mpg_central'
z_test = 3.0
val_scaling=1.0

# %%
plot_err_uncertainty(
    archive = Archive3D,
    emulator=p3d_emu, 
    sim_labels=['mpg_central','mpg_seed'],
    mu_lims_p3d = [0.31,0.38],
    z=z_test, 
    val_scaling=1.0, 
    colors = ['deepskyblue', 'goldenrod']
)


# %%
